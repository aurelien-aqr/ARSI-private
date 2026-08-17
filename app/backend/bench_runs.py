"""Benchmark runs: score a dataset with a chosen pipeline, in one of two modes.

    full      the real thing - every case goes through the pipeline the Studio
              runs (script x localizer x model x prompt) via the ordinary job
              queue, so it gets SSE progress, cancel, the shared verdict cache
              and shows up in the Results screen like any other run. Scored at
              frame AND object level.
    localize  region proposal only, no VLM, no Ollama: seconds per case instead
              of minutes, and it answers the question threshold tuning actually
              asks ("did anything box this instance at all?"). This is
              benchmark/eval_localization.py's measurement, driven from the
              localizer registry instead of ad-hoc variant strings.

A run is a directory under benchmark/runs/<run_id>/:

    run.json                config, status, progress, job_id
    dataset_snapshot.json   THE CASES AS THEY WERE when the run started
    score.json              frozen once the run ends (live-computed before that)
    report.md               same sections as the CLI's report

The snapshot is the point. Ground truth is editable now, so a run scored against
labels that have since been corrected must keep reporting what it actually
measured, and say that the dataset moved on - `stale` in the run summary.
"""
import json
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from arsi_core import benchmarks, localizers
from arsi_core.adapters import _module_overrides, configured, get_module
from arsi_core.runner import JobConfig

RUNS_DIR = benchmarks.RUNS_DIR
MODES = ("full", "localize")


class BenchError(ValueError):
    """Bad run request (unknown dataset/mode/localizer, empty case selection)."""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", run_id or ""):
        raise BenchError(f"invalid run id '{run_id}'")
    return RUNS_DIR / run_id


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False)
    tmp.replace(path)


def _read_json(path: Path, default=None):
    if not path.is_file():
        return default
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return default


def save_run(run: dict):
    _write_json(run_dir(run["run_id"]) / "run.json", run)
    return run


def load_run(run_id: str) -> dict:
    run = _read_json(run_dir(run_id) / "run.json")
    if run is None:
        raise BenchError(f"no benchmark run '{run_id}'")
    return run


def snapshot_of(run_id: str) -> dict:
    return _read_json(run_dir(run_id) / "dataset_snapshot.json",
                      {"references": {}, "cases": []})


def list_run_ids():
    if not RUNS_DIR.exists():
        return []
    return sorted((p.parent.name for p in RUNS_DIR.glob("*/run.json")),
                  reverse=True)


# ------------------------------------------------------------------ launching

def _select_cases(doc: dict, case_ids):
    """The cases to run, with the two structural preconditions checked here.

    A dataset is only validated on save and on `summary`, so a hand-edited or
    externally built file can reach this point with a case that has no `id` or
    names a reference key that does not exist. Both used to surface as a KeyError
    (a 500); they are the caller's problem to fix, so they are BenchErrors."""
    cases = doc.get("cases") or []
    refs = doc.get("references") or {}
    bad = [c.get("id") or "(no id)" for c in cases
           if not isinstance(c, dict) or not c.get("id")
           or c.get("reference") not in refs]
    if bad:
        raise BenchError(f"the dataset has {len(bad)} unusable case(s) "
                         f"({', '.join(map(str, bad[:3]))}) - each case needs an "
                         f"'id' and a 'reference' that is a key of 'references'")
    if case_ids:
        wanted = list(dict.fromkeys(case_ids))
        by_id = {c["id"]: c for c in cases}
        missing = [c for c in wanted if c not in by_id]
        if missing:
            raise BenchError(f"unknown case(s): {', '.join(missing[:5])}")
        cases = [by_id[c] for c in wanted]
    if not cases:
        raise BenchError("no cases selected")
    return cases


def create(payload: dict, submit_job, gpu: bool = True) -> dict:
    """Validate a run request, write its snapshot, and start it.

    `submit_job(JobConfig) -> job_id` is injected rather than imported so the
    tests can drive this without the real JobManager."""
    mode = payload.get("mode") or "full"
    if mode not in MODES:
        raise BenchError(f"unknown mode '{mode}' (expected {' or '.join(MODES)})")
    ds_id, doc = benchmarks.load(payload.get("dataset") or "")
    cases = _select_cases(doc, payload.get("cases"))
    refs = doc.get("references") or {}

    localizer = payload.get("localizer") or localizers.DEFAULT
    if localizer not in localizers.names():
        raise BenchError(f"unknown localizer '{localizer}'")
    ok, why = localizers.availability(localizer)
    if not ok:
        raise BenchError(why)

    script = payload.get("script") or "vlm_05"
    if mode == "localize" and script != "vlm_05":
        raise BenchError("localization-only scoring is a vlm_05 measurement "
                         "(it scores the region proposer, which only vlm_05 has)")

    # timestamp + random suffix, like a job_id: two runs of the same mode started
    # in the same second (two tabs, a double POST, a scripted A/B) would otherwise
    # share a directory and one would score against the other's snapshot
    run_id = (time.strftime("%Y%m%d-%H%M%S") + "-" + mode[:3]
              + "-" + uuid.uuid4().hex[:4])
    run = {
        "run_id": run_id, "dataset": ds_id, "digest": benchmarks.digest(doc),
        "created": _now(), "status": "queued", "job_id": None,
        "progress": {"done": 0, "total": len(cases)},
        "wall_seconds": 0.0,
        "config": {
            "mode": mode, "script": script, "localizer": localizer,
            "model": payload.get("model") or None,
            "prompt": payload.get("prompt") or None,
            "prompt_name": payload.get("prompt_name") or "default",
            "n_cases": len(cases),
            "subset": bool(payload.get("cases")),
        },
        "params": payload.get("params") or {},
    }
    _write_json(run_dir(run_id) / "dataset_snapshot.json",
                {"dataset": ds_id, "name": doc.get("name", ds_id),
                 "references": refs, "cases": cases})
    save_run(run)

    if mode == "full":
        # If the submission is refused (model not installed, Ollama down) the
        # caller gets that error - but the run directory must not survive as a row
        # stuck at "queued" with no job behind it, which nothing could then stop.
        try:
            _submit_full(run, cases, refs, script, localizer, ds_id, submit_job)
        except Exception:
            shutil.rmtree(run_dir(run_id), ignore_errors=True)
            raise
    else:
        LOCALIZE_WORKER.submit(run_id)
    return run


def _submit_full(run, cases, refs, script, localizer, ds_id, submit_job):
    cfg = JobConfig(
        script=script,
        frames=[str(benchmarks.REPO_ROOT / c["image"]) for c in cases],
        frame_references=[str(benchmarks.REPO_ROOT / refs[c["reference"]])
                          for c in cases],
        model=run["config"]["model"], prompt=run["config"]["prompt"],
        prompt_name=run["config"]["prompt_name"],
        localizer=localizer if script == "vlm_05" else None,
        params=run["params"],
        bench={"run_id": run["run_id"], "dataset": ds_id})
    # The benchmark images are already masked (each was written through its own
    # camera's mask), so no mask is applied here - doing it twice would change the
    # verdict-cache key and throw away 4642 cached verdicts.
    run["job_id"] = submit_job(cfg)
    run["status"] = "running"
    save_run(run)


# ------------------------------------------------- localization-only execution

class LocalizeWorker:
    """One thread, one run at a time. No Ollama involved: the whole cost is the
    diff (and DINOv2 when the gate is picked), ~1-2 s per case on CPU."""

    def __init__(self):
        self._q = queue.Queue()
        self.cancel = {}                       # run_id -> threading.Event
        self._thread = None

    def _ensure_thread(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def submit(self, run_id: str):
        self.cancel[run_id] = threading.Event()
        self._q.put(run_id)
        self._ensure_thread()

    def stop(self, run_id: str) -> bool:
        ev = self.cancel.get(run_id)
        if ev is None:
            return False
        ev.set()
        return True

    def _loop(self):
        while True:
            try:
                run_id = self._q.get(timeout=300)
            except queue.Empty:
                self._thread = None            # idle: let the thread retire
                return
            try:
                self._run(run_id)
            except Exception as exc:           # never kill the worker
                traceback.print_exc()
                try:
                    run = load_run(run_id)
                    run["status"] = "failed"
                    run["error"] = f"{type(exc).__name__}: {exc}"
                    save_run(run)
                except BenchError:
                    pass

    def _run(self, run_id: str):
        run = load_run(run_id)
        snap = snapshot_of(run_id)
        cases, refs = snap["cases"], snap["references"]
        stop = self.cancel.get(run_id) or threading.Event()
        name = run["config"]["localizer"]
        localizers.check(name)
        localizers.warmup(name)

        module = get_module("vlm_05")
        overrides = _module_overrides(module, run["params"])
        run["status"] = "running"
        run["started"] = _now()
        save_run(run)

        t0 = time.time()
        rows = []
        with configured(module, **overrides):
            for i, case in enumerate(cases):
                if stop.is_set():
                    run["status"] = "cancelled"
                    break
                t_case = time.time()
                ref = str(benchmarks.REPO_ROOT / refs[case["reference"]])
                img = str(benchmarks.REPO_ROOT / case["image"])
                try:
                    regions, info = localizers.localize(name, module, ref, img,
                                                        run["params"])
                    rows.append({
                        "regions": [{"bbox": list(r["bbox"]), "area": r.get("area"),
                                     "channel": r.get("channel", "photo")}
                                    for r in regions],
                        "localization": {k: v for k, v in info.items()
                                         if not isinstance(v, (list, dict))},
                        "seconds": round(time.time() - t_case, 2)})
                except Exception as exc:
                    # one unreadable case must not lose the other 28
                    rows.append({"regions": [], "localization": {},
                                 "seconds": round(time.time() - t_case, 2),
                                 "error": f"{type(exc).__name__}: {exc}"})
                run["progress"] = {"done": i + 1, "total": len(cases)}
                run["wall_seconds"] = round(time.time() - t0, 2)
                # after every case, like run_benchmark.py: an interrupted run
                # keeps the cases it did measure
                _write_json(run_dir(run_id) / "score.json",
                            benchmarks.score_localize(cases[:len(rows)], rows))
                save_run(run)
        if run["status"] != "cancelled":
            run["status"] = "completed"
        run["finished"] = _now()
        run["wall_seconds"] = round(time.time() - t0, 2)
        run["frozen"] = True
        save_run(run)
        _freeze(run, benchmarks.score_localize(cases[:len(rows)], rows))


LOCALIZE_WORKER = LocalizeWorker()


# ------------------------------------------------------------------- reading

def _freeze(run: dict, score: dict):
    """Write the final score and report once the run is over."""
    d = run_dir(run["run_id"])
    _write_json(d / "score.json", score)
    (d / "report.md").write_text(benchmarks.report_md(run, score), encoding="utf-8")
    return score


def state(run_id: str, job_data=None, job_status=None, job_error=None) -> dict:
    """Everything the UI needs about one run: the run record, its live status and
    its score. `job_data` is the results.json of the backing job (full mode) -
    passed in so this module needs no import of the job manager.

    The score is recomputed from those results on every read while the job runs
    (free live scoring), and frozen to score.json once it ends."""
    run = load_run(run_id)
    snap = snapshot_of(run_id)
    cases = snap["cases"]
    mode = run["config"]["mode"]
    score = _read_json(run_dir(run_id) / "score.json")

    if mode == "full" and not run.get("frozen"):
        # No status and no results means nothing is working on this job and
        # nothing was saved either (server restarted, job deleted). Reporting the
        # stored "running" would leave the UI polling a run that can never move.
        run["status"] = job_status or ("interrupted" if run.get("job_id")
                                      else run["status"])
        if job_error:
            run["error"] = job_error
        frames = (job_data or {}).get("frames") or []
        run["progress"] = {"done": len(frames), "total": len(cases)}
        if job_data and job_data.get("summary"):
            run["wall_seconds"] = job_data["summary"].get("wall_seconds", 0.0)
        # No frame judged yet -> no score at all, rather than an all-zero one: a
        # queued run advertising "frame F1 0.000" reads as a measurement.
        score = benchmarks.score_full(cases, frames) if frames else None
        if run["status"] in ("completed", "failed", "cancelled", "interrupted"):
            run["frozen"] = bool(frames)
            if frames:
                score = _freeze(run, score)
        save_run(run)
    return {"run": run, "score": score,
            "stale": _is_stale(run),
            "references": snap.get("references", {}),
            "dataset_name": snap.get("name", run["dataset"])}


def _is_stale(run: dict) -> bool:
    """True when the dataset has been edited since this run was scored. Not an
    error - the run's numbers stay valid for the labels it used - but a
    comparison against a newer run would be apples to oranges."""
    try:
        _, doc = benchmarks.load(run["dataset"])
    except benchmarks.DatasetError:
        return True
    return benchmarks.digest(doc) != run.get("digest")


def summarize(run_id: str, job_data=None, job_status=None, job_error=None) -> dict:
    """One row of the runs table."""
    st = state(run_id, job_data, job_status, job_error)
    run, score = st["run"], st["score"]
    return {**{k: run[k] for k in ("run_id", "dataset", "status", "progress",
                                   "created", "job_id", "wall_seconds")},
            "config": run["config"], "params": run.get("params") or {},
            "error": run.get("error"),
            "stale": st["stale"],
            "headline": benchmarks.headline(score) if score else None}


def delete(run_id: str):
    import shutil
    d = run_dir(run_id)
    if not d.is_dir():
        raise BenchError(f"no benchmark run '{run_id}'")
    shutil.rmtree(d, ignore_errors=True)
