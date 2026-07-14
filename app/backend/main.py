"""ARSI Studio backend — FastAPI over arsi_core (docs/SPEC.md milestone 2).

Run from the repo root:  venv/bin/python -m uvicorn app.backend.main:app --port 8321
"""
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException, Request, UploadFile   # noqa: E402
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,  # noqa: E402
                               PlainTextResponse, Response, StreamingResponse)
from fastapi.staticfiles import StaticFiles                        # noqa: E402

from arsi_core import APP_DATA                                     # noqa: E402
from arsi_core.adapters import SCRIPTS, get_module                 # noqa: E402
from arsi_core.errors import ArsiError, ModelMissing, OllamaUnreachable  # noqa: E402
from arsi_core.masking import MASKS_DIR, MaskSpec, list_masks      # noqa: E402
from arsi_core.ollama_client import OllamaClient                   # noqa: E402
from arsi_core.runner import JOBS_DIR, JobConfig                   # noqa: E402
from arsi_core.video import extract_frames, probe                  # noqa: E402

from .exports import report_html, report_md, results_xlsx          # noqa: E402
from .jobs import JobManager, load_saved, saved_jobs               # noqa: E402

app = FastAPI(title="ARSI Studio", version="0.1")
manager = JobManager()
VIDEOS_DIR = APP_DATA / "videos"
FRONTEND_DIR = REPO_ROOT / "app" / "frontend"

# Curated wizard catalog: benchmark verdicts baked in (benchmark/README.md).
MODEL_CATALOG = [
    {"name": "GLM-4.6V-Flash 9B", "tag": "haervwe/GLM-4.6V-Flash-9B", "size": "6.1 GB",
     "note": "best frame-level judge (benchmark F1 1.000)", "recommended": True},
    {"name": "Qwen3-VL 8B Instruct", "tag": "qwen3-vl:8b-instruct", "size": "6.5 GB",
     "note": "grounded all-rounder"},
    {"name": "Qwen3.5 9B", "tag": "qwen3.5:9b", "size": "6.3 GB",
     "note": "best instance inventory (object recall 0.978), fastest"},
    {"name": "InternVL3.5 8B", "tag": "blaifa/InternVL3_5:8b", "size": "5.9 GB",
     "note": "precision-extreme, misses small objects"},
    {"name": "Qwen2.5-VL 7B", "tag": "qwen2.5vl:7b", "size": "6.0 GB",
     "note": "original baseline"},
    {"name": "Llama 3.2 Vision 11B", "tag": "llama3.2-vision:11b", "size": "7.8 GB",
     "note": "untested as crop judge"},
]

PIPELINES = [
    {"key": "vlm_01", "name": "vlm_01 single-frame",
     "desc": "Score each frame on its own, no reference.", "ref": False},
    {"key": "vlm_02", "name": "vlm_02 reference-compare",
     "desc": "Show reference + frame to the VLM, report the differences.", "ref": True},
    {"key": "vlm_03", "name": "vlm_03 bounding-box",
     "desc": "The VLM outputs anomaly boxes as JSON on the whole frame.", "ref": False},
    {"key": "vlm_04", "name": "vlm_04 YOLO-hybrid",
     "desc": "YOLO-World localizes candidates, the VLM confirms each crop.", "ref": False},
    {"key": "vlm_05", "name": "vlm_05 reference-diff",
     "desc": "Pixel-diff vs clean reference, VLM judges each changed region.",
     "ref": True, "recommended": True},
]


def client_for(timeout=20.0):
    return OllamaClient(host=_settings().get("ollama_url") or None, timeout=timeout)


_SETTINGS_PATH = APP_DATA / "settings.json"


def _settings():
    if _SETTINGS_PATH.exists():
        with open(_SETTINGS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_settings(d):
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_SETTINGS_PATH, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1)


def _sse(gen):
    def stream():
        for item in gen:
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------- health

@app.get("/api/health")
def health():
    h = client_for(timeout=4).health()
    gpu = shutil.which("nvidia-smi") is not None
    if gpu:
        try:
            gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                 timeout=5).returncode == 0
        except Exception:
            gpu = False
    return {"ollama": h["reachable"], "detail": h.get("detail", ""),
            "models": h["models"], "gpu": gpu,
            "cpu_warning": None if gpu else
            "No NVIDIA GPU detected — VLM calls take 2-4 min each on CPU.",
            "version": app.version}


# ---------------------------------------------------------------- models

@app.get("/api/models")
def models():
    try:
        installed = client_for().model_names()
    except OllamaUnreachable:
        installed = []
    inst_set = set(installed) | {n.removesuffix(":latest") for n in installed}
    catalog = [{**m, "installed": m["tag"] in inst_set} for m in MODEL_CATALOG]
    extra = [n for n in installed
             if n not in {m["tag"] for m in MODEL_CATALOG}
             and n.removesuffix(":latest") not in {m["tag"] for m in MODEL_CATALOG}]
    for n in extra:
        catalog.append({"name": n, "tag": n, "size": "", "note": "installed locally",
                        "installed": True})
    return {"models": catalog}


@app.post("/api/models/pull")
def models_pull(payload: dict):
    tag = payload.get("tag") or ""

    def gen():
        try:
            for p in client_for(timeout=None).pull(tag):
                pct = round(p["completed"] / p["total"] * 100) if p["total"] else 0
                yield {"status": p["status"], "pct": pct}
            yield {"status": "done", "pct": 100}
        except OllamaUnreachable as exc:
            yield {"status": "error", "error": str(exc)}
    return _sse(gen())


@app.delete("/api/models/{tag:path}")
def models_remove(tag: str):
    try:
        client_for()._impl.delete(tag)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(400, f"could not remove '{tag}': {exc}")


# ---------------------------------------------------------------- media (guarded)

MEDIA_ROOTS = [REPO_ROOT / "data", REPO_ROOT / "benchmark" / "annotated"]


@app.get("/api/media/{path:path}")
def media(path: str):
    p = (REPO_ROOT / path).resolve()
    if not any(str(p).startswith(str(root.resolve()) + "/") or p == root.resolve()
               for root in MEDIA_ROOTS):
        raise HTTPException(403, "path outside the served data directories")
    if not p.is_file():
        raise HTTPException(404, path)
    return FileResponse(p)


def media_url(path) -> str:
    p = Path(path).resolve()
    try:
        return "/api/media/" + str(p.relative_to(REPO_ROOT))
    except ValueError:
        return ""


# ---------------------------------------------------------------- demo frames

@app.get("/api/demo-frames")
def demo_frames():
    gt_path = REPO_ROOT / "benchmark" / "ground_truth.json"
    with open(gt_path, encoding="utf-8") as fh:
        gt = json.load(fh)
    refs = gt["references"]
    out = []
    for c in gt["cases"]:
        img = REPO_ROOT / c["image"]
        if not img.exists():
            continue
        out.append({"id": c["id"], "img": media_url(img),
                    "path": c["image"], "source": c.get("source", ""),
                    "anomaly": c["has_anomaly"],
                    "types": c.get("types", []),
                    "label": c.get("note", "")[:60],
                    "reference": refs[c["reference"]]})
    return {"frames": out, "references": refs}


@app.get("/api/references")
def references():
    ref_dir = REPO_ROOT / "data" / "reference"
    out = [{"path": str(p.relative_to(REPO_ROOT)), "img": media_url(p), "name": p.stem}
           for p in sorted(ref_dir.glob("**/*.jpg")) + sorted(ref_dir.glob("**/*.png"))]
    return {"references": out}


# ---------------------------------------------------------------- videos

@app.post("/api/videos")
async def upload_video(file: UploadFile):
    vid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    vdir = VIDEOS_DIR / vid
    vdir.mkdir(parents=True, exist_ok=True)
    dest = vdir / ("source" + Path(file.filename or "video.mp4").suffix)
    with open(dest, "wb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)
    try:
        info = probe(dest)
    except ArsiError as exc:
        shutil.rmtree(vdir, ignore_errors=True)
        raise HTTPException(400, str(exc))
    # filmstrip thumbnails for the trim UI (10 evenly spaced small frames)
    thumbs_meta = extract_frames(dest, vdir / "thumbs",
                                 every_n=max(1, info["frame_count"] // 10),
                                 max_side=320)
    return {"video_id": vid, "info": info,
            "thumbs": [media_url(vdir / "thumbs" / f["file"])
                       for f in thumbs_meta["frames"]][:10]}


@app.get("/api/videos")
def list_videos():
    out = []
    if VIDEOS_DIR.exists():
        for vdir in sorted(VIDEOS_DIR.iterdir(), reverse=True):
            meta = vdir / "frames" / "meta.json"
            if meta.exists():
                with open(meta, encoding="utf-8") as fh:
                    m = json.load(fh)
                out.append({"video_id": vdir.name, "n_frames": len(m["frames"]),
                            "params": m["params"]})
    return {"videos": out}


@app.get("/api/videos/{video_id}/frames")
def video_frames(video_id: str):
    """Frames of an existing extraction (the 'reuse' wizard source)."""
    meta_path = VIDEOS_DIR / video_id / "frames" / "meta.json"
    if not meta_path.exists():
        raise HTTPException(404, "no extraction for this video")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    fdir = VIDEOS_DIR / video_id / "frames"
    return {"video_id": video_id,
            "frames": [{"path": str((fdir / f["file"]).relative_to(REPO_ROOT)),
                        "img": media_url(fdir / f["file"]),
                        "index": f["index"], "time_s": f["time_s"]}
                       for f in meta["frames"]]}


@app.post("/api/videos/{video_id}/extract")
def extract(video_id: str, payload: dict):
    vdir = VIDEOS_DIR / video_id
    src = next(iter(vdir.glob("source.*")), None)
    if not src:
        raise HTTPException(404, "video not found")
    meta = extract_frames(
        src, vdir / "frames",
        every_n=payload.get("every_n"), every_s=payload.get("every_s"),
        start_s=payload.get("start_s", 0.0), end_s=payload.get("end_s"))
    frames = [{"path": str((vdir / "frames" / f["file"]).relative_to(REPO_ROOT)),
               "img": media_url(vdir / "frames" / f["file"]),
               "index": f["index"], "time_s": f["time_s"]}
              for f in meta["frames"]]
    return {"video_id": video_id, "frames": frames}


# ---------------------------------------------------------------- masks

@app.get("/api/masks")
def masks():
    return {"masks": [{**m.to_dict(), "hash": m.hash} for m in list_masks()]}


@app.post("/api/masks")
def save_mask(payload: dict):
    spec = MaskSpec.from_dict(payload)
    if not spec.name or "/" in spec.name:
        raise HTTPException(400, "invalid mask name")
    path = spec.save()
    return {"saved": str(path.relative_to(REPO_ROOT)), "hash": spec.hash}


@app.delete("/api/masks/{name}")
def delete_mask(name: str):
    path = MASKS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, name)
    path.unlink()
    return {"ok": True}


@app.post("/api/masks/preview")
def mask_preview(payload: dict):
    """Render the given zones onto a frame; returns the masked JPEG."""
    spec = MaskSpec.from_dict({"name": "preview", **payload})
    img_path = (REPO_ROOT / payload["image"]).resolve()
    if not any(str(img_path).startswith(str(r.resolve()) + "/") for r in MEDIA_ROOTS):
        raise HTTPException(403, "image outside data directories")
    from io import BytesIO
    from PIL import Image
    with Image.open(img_path) as im:
        out = spec.apply(im.convert("RGB"))
    buf = BytesIO()
    out.save(buf, format="JPEG", quality=88)
    return Response(buf.getvalue(), media_type="image/jpeg")


# ---------------------------------------------------------------- jobs

@app.get("/api/pipelines")
def pipelines():
    out = []
    for p in PIPELINES:
        module = get_module(p["key"])
        prompts = {"default": module.PROMPT}
        if p["key"] == "vlm_05":
            prompts = {"conservative": module.PROMPT,
                       "lenient": module.PROMPT_LENIENT}
        out.append({**p, "prompts": prompts, "default_model": module.MODEL_NAME})
    return {"pipelines": out}


@app.post("/api/jobs")
def create_job(payload: dict):
    script = payload.get("script")
    if script not in SCRIPTS:
        raise HTTPException(400, f"unknown script '{script}'")
    frames = payload.get("frames") or []
    if not frames:
        raise HTTPException(400, "no frames given")

    def _abs(p):
        rp = (REPO_ROOT / p).resolve()
        if not any(str(rp).startswith(str(r.resolve()) + "/") for r in MEDIA_ROOTS):
            raise HTTPException(403, f"frame outside data directories: {p}")
        return str(rp)

    mask_name = payload.get("mask")
    cfg = JobConfig(
        script=script, frames=[_abs(f) for f in frames],
        model=payload.get("model") or None,
        prompt=payload.get("prompt") or None,
        prompt_name=payload.get("prompt_name", "default"),
        reference=_abs(payload["reference"]) if payload.get("reference") else None,
        mask=str(MASKS_DIR / f"{mask_name}.json") if mask_name else None,
        params=payload.get("params") or {})
    # fail fast in the request (docs/SPEC.md: model missing -> 409 + pull hint)
    if cfg.model:
        try:
            client_for().ensure_model(cfg.model)
        except ModelMissing as exc:
            raise HTTPException(409, str(exc))
        except OllamaUnreachable as exc:
            raise HTTPException(503, str(exc))
    job = manager.submit(cfg)
    return {"job_id": job.job_id}


@app.get("/api/jobs")
def jobs_index():
    live = {j.job_id: j.public() for j in manager.jobs.values()}
    hist = []
    for data in saved_jobs():
        jid = data["job_id"]
        if jid in live:
            continue
        hist.append({"job_id": jid, "status": data["status"],
                     "config": data["config"], "summary": data["summary"]})
    return {"jobs": list(live.values()) + hist}


def _job_data(job_id: str):
    job = manager.get(job_id)
    if job and job.result:
        return job.result.to_dict()
    data = load_saved(job_id)
    if data is None and job is not None:
        return job.public()          # queued/running: no frames yet
    if data is None:
        raise HTTPException(404, job_id)
    return data


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str):
    data = _job_data(job_id)
    for f in data.get("frames", []):
        f["img"] = media_url(f["image"])
    if data.get("config", {}).get("reference"):
        data["config"]["reference_img"] = media_url(data["config"]["reference"])
    live = manager.get(job_id)
    if live:
        data["status"] = live.status if live.status != "completed" \
            else data.get("status", live.status)
    return data


@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, job_id)
    q, backlog = job.subscribe()

    def gen():
        try:
            for e in backlog:
                yield e
                if e.get("event") == "stream_end":
                    return
            while True:
                e = q.get()
                yield e
                if e.get("event") == "stream_end":
                    return
        finally:
            job.unsubscribe(q)
    return _sse(gen())


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    if not manager.cancel(job_id):
        raise HTTPException(404, job_id)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/report.md")
def job_report_md(job_id: str):
    return PlainTextResponse(report_md(_job_data(job_id)),
                             media_type="text/markdown")


@app.get("/api/jobs/{job_id}/report.html")
def job_report_html(job_id: str):
    return HTMLResponse(report_html(_job_data(job_id)))


@app.get("/api/jobs/{job_id}/results.json")
def job_results_json(job_id: str):
    return JSONResponse(_job_data(job_id))


@app.get("/api/jobs/{job_id}/export.xlsx")
def job_xlsx(job_id: str):
    blob = results_xlsx(_job_data(job_id))
    return Response(blob, media_type="application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition":
                             f'attachment; filename="{job_id}.xlsx"'})


# ---------------------------------------------------------------- settings

@app.get("/api/settings")
def get_settings():
    s = _settings()
    sizes = {}
    for label, path in (("frames", VIDEOS_DIR), ("results", JOBS_DIR)):
        total = 0
        if path.exists():
            total = sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())
        sizes[label] = total
    try:
        data = client_for().list()
        ms = getattr(data, "models", None) or (data.get("models", [])
                                               if isinstance(data, dict) else [])
        sizes["models"] = sum(getattr(m, "size", 0) or
                              (m.get("size", 0) if isinstance(m, dict) else 0)
                              for m in ms)
    except Exception:
        sizes["models"] = 0
    return {"ollama_url": s.get("ollama_url", "http://localhost:11434"),
            "defaults": s.get("defaults", {"script": "vlm_05",
                                           "prompt": "conservative"}),
            "storage": sizes, "n_jobs": len(jobs_index()["jobs"])}


@app.post("/api/settings")
def set_settings(payload: dict):
    s = _settings()
    s.update({k: v for k, v in payload.items()
              if k in ("ollama_url", "defaults")})
    _save_settings(s)
    return {"ok": True}


# ---------------------------------------------------------------- frontend

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
