"""Job runner: batch execution with per-frame isolation, retries, masking and
structured logging (docs/SPEC.md "Error taxonomy" — behaviour is the spec)."""
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import APP_DATA
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
    mask: str = None                    # path to a MaskSpec JSON, or None
    params: dict = field(default_factory=dict)   # timeout_s, max_retries + UPPER_CASE module overrides
    job_id: str = None
    job_dir: str = None

    def resolved(self):
        self.job_id = self.job_id or datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.job_dir = Path(self.job_dir) if self.job_dir else JOBS_DIR / self.job_id
        return self

    def public_dict(self):
        return {"script": self.script, "model": self.model,
                "prompt_name": self.prompt_name, "prompt": self.prompt,
                "reference": self.reference, "mask": self.mask,
                "n_frames": len(self.frames), "params": self.params}


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


def _materialize_mask(cfg: JobConfig, log) -> tuple:
    """Render masked copies of the reference and every frame into the job dir.
    The mask must hit BOTH sides identically or the diff pipelines would
    detect the mask itself as change. Returns (frames, reference, mask_hash)."""
    if not cfg.mask:
        return cfg.frames, cfg.reference, ""
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
    reference = mask_one(cfg.reference) if cfg.reference else None
    log("mask_applied", mask=spec.name, hash=spec.hash, n_images=len(done))
    return frames, reference, spec.hash


def run_job(cfg: JobConfig, on_event=None, client=None, cache=None) -> JobResult:
    """Execute the batch. Job-fatal errors (Ollama down, model missing) raise
    BEFORE any frame runs; per-frame errors never stop the batch."""
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
    if cfg.reference is None and NEEDS_REFERENCE.get(cfg.script):
        raise FrameError(f"{cfg.script} needs a reference image")

    prompt = cfg.prompt or default_prompt(cfg.script)
    result = JobResult(job_id=cfg.job_id, config=cfg.public_dict(),
                       started=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    emit("job_started", job_id=cfg.job_id, script=cfg.script,
         model=cfg.model or "(script default)", n_frames=len(cfg.frames))
    t_job = time.time()

    frames, reference, mask_hash = _materialize_mask(cfg, log)

    for i, frame in enumerate(frames):
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
                fr = run_frame(cfg.script, frame, reference=reference,
                               model=cfg.model,
                               prompt=prompt + (FORMAT_REMINDER if format_failed else ""),
                               params=params, client=client, cache=cache,
                               mask_hash=mask_hash)
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
        emit("frame_done", index=i, frame_id=fr.frame_id, status=fr.status,
             anomaly=fr.anomaly, n_detections=len(fr.detections),
             attempts=fr.attempts, seconds=fr.seconds)

    ok = [f for f in result.frames if f.status == "ok"]
    result.summary = JobSummary(
        n_frames=len(result.frames), n_ok=len(ok),
        n_anomalous=sum(1 for f in ok if f.anomaly),
        n_failed=sum(1 for f in result.frames if f.status == "failed"),
        wall_seconds=round(time.time() - t_job, 2))
    result.finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result.status = "completed" if result.summary.n_failed < len(result.frames) or not result.frames \
        else "failed"

    out_path = Path(cfg.job_dir) / "results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=1, ensure_ascii=False)
    emit("job_finished", status=result.status, **result.summary.__dict__)
    log.close()
    return result
