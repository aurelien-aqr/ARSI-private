"""Job runner: batch execution with per-frame isolation, retries, masking and
structured logging (docs/SPEC.md "Error taxonomy" - behaviour is the spec)."""
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import APP_DATA
from . import localizers
from .adapters import NEEDS_REFERENCE, default_prompt, get_module, run_frame
from .cache import VerdictCache
from .errors import FrameError, ParseError, VLMCallError
from .masking import MaskSpec
from .ollama_client import OllamaClient
from .schema import FrameResult, JobResult, JobSummary

JOBS_DIR = APP_DATA / "jobs"

FORMAT_REMINDER = ("\n\nREMINDER: your previous answer did not follow the "
                   "required output format. Answer in EXACTLY the format "
                   "specified above, with no extra text.")


@dataclass
class JobConfig:
    script: str
    frames: list                        # image paths
    model: str = None                   # None -> script default
    prompt: str = None                  # None -> script default
    prompt_name: str = "default"        # preset label for the report
    reference: str = None
    # Per-frame reference, parallel to `frames` (None entry -> `reference`). A
    # benchmark protocol can pair each frame with its OWN clean reference: the
    # 39T dataset has four cameras, so scoring it in one run needs four
    # references. Left None for the ordinary one-camera case.
    frame_references: list = None
    mask: str = None                    # path to a MaskSpec JSON, or None
    localizer: str = None               # vlm_05 region proposal; None -> localizers.DEFAULT
    params: dict = field(default_factory=dict)   # timeout_s, max_retries + UPPER_CASE module overrides
    bench: dict = None                  # {run_id, dataset} when this job scores a benchmark
    job_id: str = None
    job_dir: str = None

    def resolved(self):
        self.job_id = self.job_id or datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.job_dir = Path(self.job_dir) if self.job_dir else JOBS_DIR / self.job_id
        if self.frame_references is not None \
                and len(self.frame_references) != len(self.frames):
            raise FrameError(
                f"frame_references has {len(self.frame_references)} entries for "
                f"{len(self.frames)} frames - they must line up, or a frame "
                f"would be diffed against the wrong camera")
        return self

    def references_for_frames(self):
        """One reference per frame, resolved. `reference` is the fallback so the
        two spellings never have to be handled separately downstream."""
        if self.frame_references is None:
            return [self.reference] * len(self.frames)
        return [r or self.reference for r in self.frame_references]

    def public_dict(self):
        d = {"script": self.script, "model": self.model,
             "prompt_name": self.prompt_name, "prompt": self.prompt,
             "reference": self.reference, "mask": self.mask,
             "localizer": self.localizer, "n_frames": len(self.frames),
             "params": self.params}
        if self.frame_references is not None:
            # the count, not the list: what a reader needs is "this run used more
            # than one reference", and the per-frame value is on each FrameResult
            d["n_references"] = len(set(self.references_for_frames()))
        if self.bench:
            d["bench"] = self.bench
        return d


class _JobLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def __call__(self, event: str, **fields):
        rec = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "event": event, **fields}
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()
        return rec

    def close(self):
        self._fh.close()


def _materialize_mask(cfg: JobConfig, emit) -> tuple:
    """Render masked copies of every reference and every frame into the job dir.
    The mask must hit BOTH sides identically or the diff pipelines would
    detect the mask itself as change. Returns (frames, references, mask_hash),
    where `references` has one entry per frame.

    Everything downstream - FrameResult.image, the events, the report - then
    carries the masked paths, so what the UI shows is what the VLM saw."""
    if not cfg.mask:
        return cfg.frames, cfg.references_for_frames(), ""
    spec = MaskSpec.load(cfg.mask)
    masked_dir = Path(cfg.job_dir) / "masked"
    done = {}

    def mask_one(src):
        src = str(src)
        if src not in done:
            # disambiguate identical basenames coming from different directories
            name = Path(src).name
            if any(Path(v).name == name for v in done.values()):
                name = f"{Path(src).stem}-{len(done)}{Path(src).suffix}"
            done[src] = str(spec.apply_file(src, masked_dir / name))
        return done[src]

    frames = [mask_one(f) for f in cfg.frames]
    references = [mask_one(r) if r else None for r in cfg.references_for_frames()]
    emit("mask_applied", mask=spec.name, hash=spec.hash, n_images=len(done),
         reference=references[0] if references else None)
    return frames, references, spec.hash


def run_job(cfg: JobConfig, on_event=None, client=None, cache=None,
            stop=None) -> JobResult:
    """Execute the batch. Job-fatal errors (Ollama down, model missing) raise
    BEFORE any frame runs; per-frame errors never stop the batch. `stop` is an
    optional callable checked between frames AND, for the crop-judging pipelines,
    between regions inside a frame - a busy frame is 20-30 VLM calls, so a
    frame-only check made cancellation take minutes. When it returns True the job
    ends with status "cancelled", keeping the partial results (including the
    regions already judged in the frame that was interrupted)."""
    cfg.resolved()
    log = _JobLog(Path(cfg.job_dir) / "job.log")

    def emit(event, **fields):
        rec = log(event, **fields)
        if on_event:
            on_event(rec)

    params = cfg.params or {}
    max_retries = int(params.get("max_retries", 2))
    client = client or OllamaClient(timeout=float(params.get("timeout_s", 120)))
    cfg.model = cfg.model or get_module(cfg.script).MODEL_NAME
    client.ensure_model(cfg.model)          # raises ModelMissing / OllamaUnreachable
    if cache is None and cfg.script == "vlm_05":
        cache = VerdictCache()
    if NEEDS_REFERENCE.get(cfg.script) \
            and not all(cfg.references_for_frames()):
        raise FrameError(f"{cfg.script} needs a reference image "
                         f"(every frame needs one, not just the job)")
    if cfg.script == "vlm_05":
        # validate and load the region proposer BEFORE the loop: an unavailable
        # backbone must fail the job once, with one message, not per frame
        cfg.localizer = cfg.localizer or localizers.DEFAULT
        localizers.check(cfg.localizer)
        note = localizers.warmup(cfg.localizer, cfg.references_for_frames())
        if note:
            emit("localizer_ready", localizer=cfg.localizer, note=note)

    prompt = cfg.prompt or default_prompt(cfg.script)
    result = JobResult(job_id=cfg.job_id, config=cfg.public_dict(),
                       started=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    emit("job_started", job_id=cfg.job_id, script=cfg.script,
         model=cfg.model or "(script default)", n_frames=len(cfg.frames),
         localizer=cfg.localizer or "")
    t_job = time.time()

    frames, references, mask_hash = _materialize_mask(cfg, emit)
    if cache is not None:
        # the verdict key names a file, it does not fingerprint it: an image
        # rebuilt in place would otherwise be scored against the old pixels'
        # verdicts, silently and with no failure to notice
        for stale in cache.drop_changed(list(frames) + list(references)):
            emit("cache_invalidated", **stale)
    if mask_hash:
        # the masked reference is what the pipeline compared against - the UI
        # must show that one, not the untouched file the user picked
        result.config["reference_masked"] = references[0] if references else None
        result.config["mask_hash"] = mask_hash

    def snapshot(status):
        """Persist what exists so far, atomically. Called after EVERY frame: the
        file used to be written once at the end, so a server killed mid-job left
        no record at all - the job vanished from the history and its finished
        frames were lost. `status` is "running" until the job actually ends, which
        is also what lets the history spot an interrupted job (a file that says
        running with no live worker behind it)."""
        okf = [f for f in result.frames if f.status == "ok"]
        result.summary = JobSummary(
            n_frames=len(result.frames), n_ok=len(okf),
            n_anomalous=sum(1 for f in okf if f.anomaly),
            n_failed=sum(1 for f in result.frames if f.status == "failed"),
            wall_seconds=round(time.time() - t_job, 2))
        result.status = status
        out = Path(cfg.job_dir) / "results.json"
        tmp = out.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=1, ensure_ascii=False)
        tmp.replace(out)          # a reader never sees a half-written file

    cancelled = False
    for i, frame in enumerate(frames):
        if stop and stop():
            cancelled = True
            emit("job_cancelled", after_frames=i)
            break
        emit("frame_start", index=i, frame=str(frame),
             frame_id=Path(frame).stem, n_frames=len(frames))
        t0 = time.time()
        fr = None
        attempt = 0
        format_failed = False
        while True:
            attempt += 1
            try:
                # The reminder is appended only after a ParseError: appending it
                # on transport retries too would change the prompt fingerprint
                # and silently invalidate the vlm_05 verdict cache for the frame.
                fr = run_frame(cfg.script, frame, reference=references[i],
                               model=cfg.model,
                               prompt=prompt + (FORMAT_REMINDER if format_failed else ""),
                               params=params, client=client, cache=cache,
                               mask_hash=mask_hash, localizer=cfg.localizer,
                               stop=stop)
                fr.attempts = attempt
                break
            except (ParseError, VLMCallError) as exc:
                format_failed = format_failed or isinstance(exc, ParseError)
                emit("frame_retry", index=i, frame=str(frame), attempt=attempt,
                     error=f"{type(exc).__name__}: {exc}")
                if attempt > max_retries:
                    fr = FrameResult(frame_id=Path(frame).stem, image=str(frame),
                                     status="failed", attempts=attempt, anomaly=None,
                                     raw_response=getattr(exc, "raw", ""),
                                     error=f"{type(exc).__name__}: {exc}")
                    break
            except FrameError as exc:
                fr = FrameResult(frame_id=Path(frame).stem, image=str(frame),
                                 status="failed", attempts=attempt, anomaly=None,
                                 error=f"FrameError: {exc}")
                break
        fr.seconds = round(time.time() - t0, 2)
        result.frames.append(fr)
        if fr.status == "cancelled":
            # the pipeline stopped mid-frame on the cancel flag: no point
            # starting the next frame
            cancelled = True
        # snapshot BEFORE the event: a client that reacts to frame_done by
        # fetching results.json must not read a file that predates the frame
        snapshot("running")
        emit("frame_done", index=i, frame_id=fr.frame_id, status=fr.status,
             anomaly=fr.anomaly, n_detections=len(fr.detections),
             attempts=fr.attempts, seconds=fr.seconds,
             # image + detections let the run screen show the verdict on the
             # frame live, without waiting for results.json
             frame=fr.image, detections=[asdict(d) for d in fr.detections],
             error=fr.error)
        if cancelled:
            emit("job_cancelled", after_frames=i + 1)
            break

    result.finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if cancelled:
        result.status = "cancelled"
    else:
        n_failed = sum(1 for f in result.frames if f.status == "failed")
        result.status = "completed" if n_failed < len(result.frames) \
            or not result.frames else "failed"

    snapshot(result.status)
    emit("job_finished", status=result.status, **result.summary.__dict__)
    log.close()
    return result
