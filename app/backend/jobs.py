"""In-process job manager: one worker thread (Ollama is the bottleneck, jobs
run strictly one at a time), per-job event ring buffers for SSE subscribers,
results persisted by arsi_core.runner (results.json per job dir)."""
import json
import queue
import threading
import traceback
from pathlib import Path

from arsi_core.errors import ArsiError
from arsi_core.runner import JOBS_DIR, JobConfig, run_job


class Job:
    def __init__(self, cfg: JobConfig):
        cfg.resolved()
        self.cfg = cfg
        self.status = "queued"
        self.events = []                 # full history (ring-buffered)
        self.subscribers = []            # [queue.Queue] of live SSE listeners
        self.cancel_flag = threading.Event()
        self.result = None
        self.error = None
        self.lock = threading.Lock()

    @property
    def job_id(self):
        return self.cfg.job_id

    def emit(self, event: dict):
        with self.lock:
            self.events.append(event)
            if len(self.events) > 2000:
                self.events = self.events[-1500:]
            for q in list(self.subscribers):
                q.put(event)

    def subscribe(self):
        q = queue.Queue()
        with self.lock:
            backlog = list(self.events)
            self.subscribers.append(q)
        return q, backlog

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def public(self):
        d = {"job_id": self.job_id, "status": self.status,
             "config": self.cfg.public_dict(), "error": self.error}
        if self.result:
            d["summary"] = self.result.summary.__dict__
        return d


class JobManager:
    def __init__(self):
        self.jobs = {}                   # job_id -> Job
        self._q = queue.Queue()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def submit(self, cfg: JobConfig) -> Job:
        job = Job(cfg)
        self.jobs[job.job_id] = job
        job.emit({"event": "job_queued", "job_id": job.job_id})
        self._q.put(job)
        return job

    def get(self, job_id: str):
        return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        job.cancel_flag.set()
        if job.status == "queued":
            job.status = "cancelled"
            job.emit({"event": "job_finished", "status": "cancelled"})
        return True

    def _loop(self):
        while True:
            job = self._q.get()
            if job.cancel_flag.is_set():
                continue
            job.status = "running"
            try:
                job.result = run_job(job.cfg, on_event=job.emit,
                                     stop=job.cancel_flag.is_set)
                job.status = job.result.status
            except ArsiError as exc:
                job.status, job.error = "failed", str(exc)
                job.emit({"event": "job_finished", "status": "failed",
                          "error": str(exc)})
            except Exception as exc:     # never kill the worker thread
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                job.emit({"event": "job_finished", "status": "failed",
                          "error": job.error})
            finally:
                job.emit({"event": "stream_end", "status": job.status})


def saved_jobs():
    """Jobs of past sessions, read from data/app/jobs/*/results.json."""
    out = []
    if not JOBS_DIR.exists():
        return out
    for res in sorted(JOBS_DIR.glob("*/results.json"),
                      key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(res, encoding="utf-8") as fh:
                data = json.load(fh)
            out.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return out


def load_saved(job_id: str):
    path = JOBS_DIR / job_id / "results.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
