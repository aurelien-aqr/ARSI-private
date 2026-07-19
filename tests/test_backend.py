"""API tests for the FastAPI backend (docs/SPEC.md milestone 4): endpoint
contracts with a fake Ollama, the failure-injection pass (model absent, bad
JSON, corrupt frame, server down, garbage uploads) and storage cleanup.

All runtime state goes to the ARSI_APP_DATA temp dir set up in conftest.py.
"""
import json
import os
import time
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.backend.main as backend
from arsi_core import runner
from arsi_core.ollama_client import OllamaClient

from conftest import FakeOllama

APP_DATA = Path(os.environ["ARSI_APP_DATA"])
REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKED_FRAME = "data/raw/tram_1762_v1_f0001.jpg"   # committed benchmark frame


class DownOllama:
    """Every call fails — simulates an unreachable server."""
    def list(self):
        raise ConnectionError("connection refused")

    def chat(self, **kw):
        raise ConnectionError("connection refused")


@pytest.fixture
def api(monkeypatch):
    """TestClient wired to a programmable fake Ollama, for both the request
    path (backend.client_for) and the worker thread (runner.OllamaClient)."""
    def make(replies=None, models=("qwen3-vl:8b-instruct",), down=False):
        impl = DownOllama() if down else FakeOllama(replies, models)
        oc = OllamaClient(impl=impl)
        monkeypatch.setattr(backend, "client_for", lambda timeout=None: oc)
        monkeypatch.setattr(runner, "OllamaClient", lambda *a, **k: oc)
        return TestClient(backend.app), impl
    return make


def app_image(name, size=(120, 90), color=(120, 120, 120)):
    """A real image under APP_DATA (inside data/, so the media guard accepts it)."""
    path = APP_DATA / "testimgs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def wait_job(client, job_id, timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        data = client.get(f"/api/jobs/{job_id}").json()
        if data.get("status") in ("completed", "failed", "cancelled"):
            return data
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


# ---------------------------------------------------------------- health / models

def test_health_up_and_down(api):
    client, _ = api(models=("qwen3-vl:8b-instruct",))
    h = client.get("/api/health").json()
    assert h["ollama"] is True and "qwen3-vl:8b-instruct" in h["models"]

    client, _ = api(down=True)
    h = client.get("/api/health").json()
    assert h["ollama"] is False and h["models"] == []


def test_models_catalog_marks_installed_latest_aware(api):
    client, _ = api(models=("qwen3-vl:8b-instruct",
                            "haervwe/GLM-4.6V-Flash-9B:latest",
                            "weird/custom:7b"))
    rows = {m["tag"]: m for m in client.get("/api/models").json()["models"]}
    assert rows["qwen3-vl:8b-instruct"]["installed"]
    assert rows["haervwe/GLM-4.6V-Flash-9B"]["installed"]     # ":latest" normalized
    assert not rows["qwen3.5:9b"]["installed"]
    assert rows["weird/custom:7b"]["installed"]               # extra local model listed


# ---------------------------------------------------------------- media guard

def test_media_guard(api):
    client, _ = api()
    assert client.get("/api/media/etc/passwd").status_code == 403
    assert client.get("/api/media/arsi_core/cli.py").status_code == 403
    assert client.get("/api/media/data/raw/nope.jpg").status_code == 404
    r = client.get(f"/api/media/{TRACKED_FRAME}")
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/")


# ---------------------------------------------------------------- uploads (failure injection)

def test_reference_upload_rejects_garbage(api):
    client, _ = api()
    r = client.post("/api/references",
                    files={"file": ("evil.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400
    assert not (REPO_ROOT / "data" / "reference" / "uploaded" / "evil.jpg").exists()


def test_reference_upload_accepts_image(api):
    client, _ = api()
    buf = BytesIO()
    Image.new("RGB", (60, 40), (10, 200, 10)).save(buf, format="JPEG")
    dest = REPO_ROOT / "data" / "reference" / "uploaded" / "pytest_ref_tmp.jpg"
    try:
        r = client.post("/api/references",
                        files={"file": ("pytest ref tmp.jpg", buf.getvalue(),
                                        "image/jpeg")})
        assert r.status_code == 200
        body = r.json()
        assert body["path"].endswith("pytest_ref_tmp.jpg")   # space normalized
        assert dest.exists()
    finally:
        dest.unlink(missing_ok=True)


def test_video_upload_rejects_garbage(api):
    client, _ = api()
    vids_dir = APP_DATA / "videos"
    before = set(p.name for p in vids_dir.iterdir()) if vids_dir.exists() else set()
    r = client.post("/api/videos",
                    files={"file": ("junk.mp4", b"\x00\x01 not a video", "video/mp4")})
    assert r.status_code == 400
    # the partial upload dir was cleaned up, nothing lingers
    after = set(p.name for p in vids_dir.iterdir()) if vids_dir.exists() else set()
    assert after == before


def make_test_video(path, n_frames=12, size=(64, 48)):
    import cv2
    import numpy as np
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, size)
    assert vw.isOpened(), "OpenCV cannot create the MJPG test video"
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), i * 20 % 255, dtype="uint8")
        vw.write(frame)
    vw.release()
    return path


def test_video_upload_extract_list_delete(api, tmp_path):
    client, _ = api()
    src = make_test_video(tmp_path / "clip.avi")
    r = client.post("/api/videos",
                    files={"file": ("clip.avi", src.read_bytes(), "video/avi")})
    assert r.status_code == 200
    vid = r.json()["video_id"]
    assert r.json()["info"]["frame_count"] > 0

    r = client.post(f"/api/videos/{vid}/extract", json={"every_n": 4})
    frames = r.json()["frames"]
    assert len(frames) == 3 and frames[0]["img"].startswith("/api/media/")

    assert vid in {v["video_id"] for v in client.get("/api/videos").json()["videos"]}
    listing = client.get(f"/api/videos/{vid}/frames").json()
    assert len(listing["frames"]) == 3

    sto = client.get("/api/storage").json()
    entry = next(v for v in sto["videos"] if v["video_id"] == vid)
    assert entry["n_frames"] == 3 and entry["bytes"] > 0 and not entry["in_use"]

    assert client.delete(f"/api/videos/{vid}").json() == {"ok": True}
    assert client.delete(f"/api/videos/{vid}").status_code == 404
    assert vid not in {v["video_id"]
                       for v in client.get("/api/storage").json()["videos"]}


# ---------------------------------------------------------------- masks

def test_mask_crud_and_preview(api):
    client, _ = api()
    spec = {"name": "pytest-mask", "camera": "cam1", "image_size": [120, 90],
            "zones": [{"id": "1", "label": "Zone 1",
                       "polygon": [[0, 0], [50, 0], [50, 40]]}]}
    r = client.post("/api/masks", json=spec)
    assert r.status_code == 200 and r.json()["hash"]

    assert client.post("/api/masks", json={**spec, "name": "a/b"}).status_code == 400

    names = {m["name"] for m in client.get("/api/masks").json()["masks"]}
    assert "pytest-mask" in names

    r = client.post("/api/masks/preview",
                    json={"image": TRACKED_FRAME, "image_size": [120, 90],
                          "zones": spec["zones"]})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"

    assert client.delete("/api/masks/pytest-mask").json() == {"ok": True}
    assert client.delete("/api/masks/pytest-mask").status_code == 404


# ---------------------------------------------------------------- jobs: request-time failures

def test_job_rejects_bad_requests(api):
    client, _ = api()
    frame = str(app_image("ok.jpg"))
    assert client.post("/api/jobs", json={"script": "vlm_99",
                                          "frames": [frame]}).status_code == 400
    assert client.post("/api/jobs", json={"script": "vlm_01",
                                          "frames": []}).status_code == 400
    assert client.post("/api/jobs", json={"script": "vlm_01",
                                          "frames": ["/etc/passwd"]}).status_code == 403


def test_job_model_missing_409(api):
    client, _ = api(models=("qwen3-vl:8b-instruct",))
    r = client.post("/api/jobs", json={"script": "vlm_01",
                                       "frames": [str(app_image("m.jpg"))],
                                       "model": "not-pulled:9b"})
    assert r.status_code == 409
    assert "not-pulled:9b" in r.json()["detail"]


def test_job_ollama_down_503(api):
    client, _ = api(down=True)
    r = client.post("/api/jobs", json={"script": "vlm_01",
                                       "frames": [str(app_image("d.jpg"))],
                                       "model": "qwen3-vl:8b-instruct"})
    assert r.status_code == 503


# ---------------------------------------------------------------- jobs: full flow + per-frame failures

VALID_VLM01 = ("GRAFFITI: no\nVANDALISM: no\nFORGOTTEN OBJECT: yes\n"
               "- Phone (left seat)\nSEVERITY: 2")


def test_job_flow_bad_json_isolated_per_frame(api):
    """Frame 1 parses; frame 2 answers garbage 3 times (ParseError -> retries
    exhausted) — the job must complete with the failure isolated to frame 2."""
    client, _ = api(replies=[VALID_VLM01, "garbage", "garbage", "garbage"])
    f1, f2 = app_image("flow1.jpg"), app_image("flow2.jpg")
    r = client.post("/api/jobs", json={
        "script": "vlm_01", "frames": [str(f1), str(f2)],
        "model": "qwen3-vl:8b-instruct", "params": {"max_retries": 2}})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    data = wait_job(client, job_id)
    assert data["status"] == "completed"
    ok, bad = data["frames"]
    assert ok["status"] == "ok" and ok["anomaly"] is True
    assert ok["detections"][0]["label"] == "Phone"
    assert bad["status"] == "failed" and bad["attempts"] == 3
    assert "ParseError" in bad["error"]
    assert data["summary"] == {"n_frames": 2, "n_ok": 1, "n_anomalous": 1,
                               "n_failed": 1,
                               "wall_seconds": data["summary"]["wall_seconds"]}

    # exports built from the same saved results
    assert job_id in client.get(f"/api/jobs/{job_id}/report.md").text
    assert client.get(f"/api/jobs/{job_id}/report.html").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/results.json").json()["job_id"] == job_id
    r = client.get(f"/api/jobs/{job_id}/export.xlsx")
    assert r.status_code == 200 and r.content[:2] == b"PK"

    # SSE backlog of the finished job replays through stream_end and closes
    with client.stream("GET", f"/api/jobs/{job_id}/events") as s:
        events = [json.loads(line[6:]) for line in s.iter_lines()
                  if line.startswith("data: ")]
    assert events[-1]["event"] == "stream_end"
    assert any(e.get("event") == "frame_retry" for e in events)

    # storage lists it; delete removes the directory and the listing
    sto = client.get("/api/storage").json()
    assert any(j["job_id"] == job_id for j in sto["jobs"])
    assert client.delete(f"/api/jobs/{job_id}").json() == {"ok": True}
    assert not (APP_DATA / "jobs" / job_id).exists()
    assert client.delete(f"/api/jobs/{job_id}").status_code == 404


def test_job_corrupt_frame_fails_cleanly(api):
    """A file that is not an image must fail that frame with FrameError,
    without retries and without killing the worker."""
    bad = APP_DATA / "testimgs" / "corrupt.jpg"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("this is not a jpeg")
    # vlm_03 opens the frame locally (for bbox scaling) -> FrameError path;
    # vlm_01/02 never read the file themselves, the Ollama server does.
    client, _ = api(replies=["[]"])
    r = client.post("/api/jobs", json={
        "script": "vlm_03", "frames": [str(bad), str(app_image("after.jpg"))],
        "model": "qwen3-vl:8b-instruct"})
    data = wait_job(client, r.json()["job_id"])
    assert data["status"] == "completed"
    corrupt, after = data["frames"]
    assert corrupt["status"] == "failed" and "FrameError" in corrupt["error"]
    assert corrupt["attempts"] == 1          # unreadable file: no pointless retries
    assert after["status"] == "ok"           # the batch went on


def test_job_unknown_id_404(api):
    client, _ = api()
    assert client.get("/api/jobs/nope-000").status_code == 404
    assert client.post("/api/jobs/nope-000/cancel").status_code == 404
    assert client.delete("/api/jobs/nope-000").status_code == 404


# ---------------------------------------------------------------- settings

def test_settings_roundtrip(api):
    client, _ = api()
    r = client.get("/api/settings").json()
    assert "storage" in r and "ollama_url" in r
    assert client.post("/api/settings",
                       json={"ollama_url": "http://gpu-box:11434",
                             "ignored_key": 1}).json() == {"ok": True}
    assert client.get("/api/settings").json()["ollama_url"] == "http://gpu-box:11434"
