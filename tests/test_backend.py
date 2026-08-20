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
    """Every call fails - simulates an unreachable server."""
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

    r = client.get("/api/masks/pytest-mask/labelme")
    assert r.status_code == 200
    doc = r.json()
    assert doc["imageWidth"] == 120 and doc["shapes"][0]["shape_type"] == "polygon"
    assert "pytest-mask.json" in r.headers["content-disposition"]

    assert client.delete("/api/masks/pytest-mask").json() == {"ok": True}
    assert client.delete("/api/masks/pytest-mask").status_code == 404
    assert client.get("/api/masks/pytest-mask/labelme").status_code == 404


def test_labelme_import(api):
    """The team annotates in LabelMe; import converts without saving, so the
    zones can be checked over the frame first."""
    client, _ = api()
    doc = {"version": "5.5.0", "imageWidth": 120, "imageHeight": 90,
           "shapes": [
               {"label": "window", "shape_type": "rectangle",
                "points": [[10, 10], [60, 40]]},
               {"label": "rail", "shape_type": "linestrip",
                "points": [[0, 0], [5, 5]]}]}
    r = client.post("/api/masks/labelme", json={"labelme": doc, "camera": "cam9"})
    assert r.status_code == 200
    m = r.json()
    assert m["image_size"] == [120, 90] and m["camera"] == "cam9"
    assert m["zones"][0]["polygon"] == [[10, 10], [60, 10], [60, 40], [10, 40]]
    assert [s["shape_type"] for s in m["skipped"]] == ["linestrip"]
    # importing does not save: the user names the preset afterwards
    assert not any(x["name"] == "imported"
                   for x in client.get("/api/masks").json()["masks"])

    for bad in ({}, {"shapes": []}, {"imageWidth": 120, "imageHeight": 90,
                                     "shapes": [{"shape_type": "point",
                                                 "points": [[1, 1]]}]}):
        assert client.post("/api/masks/labelme",
                           json={"labelme": bad}).status_code == 400


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


def test_localizer_catalog_and_validation(api):
    """The region proposer is selectable independently of the judge: the UI reads
    the catalogue from the backend (so availability is resolved server-side), and
    an unknown choice must be refused before any frame is processed."""
    from arsi_core import localizers
    client, _ = api()
    body = client.get("/api/localizers").json()
    assert body["default"] == localizers.DEFAULT
    assert {l["key"] for l in body["localizers"]} == set(localizers.names())
    assert all(l["summary"] for l in body["localizers"])

    frame, ref = str(app_image("loc1.jpg")), str(app_image("loc-ref.jpg"))
    bad = client.post("/api/jobs", json={"script": "vlm_05", "frames": [frame],
                                        "reference": ref, "localizer": "magic"})
    assert bad.status_code == 400 and "magic" in bad.json()["detail"]


def test_the_picker_warns_before_the_run_that_a_veto_cannot_fire(api, monkeypatch):
    """'dino+dinomaly' is selectable on any camera but only DOES anything where a
    Dinomaly model was trained. That is exactly the kind of fact that must reach
    the operator at selection time - the picker sits directly above the reference
    chooser - and not as a per-frame info key discovered after paying for a run.
    The rule lives server-side so the UI cannot fork resolve_camera's matching."""
    from arsi_core import localizers
    client, _ = api()
    monkeypatch.setattr(localizers, "checkpoint_cameras", lambda: ["39T-cam53"])

    def row(**q):
        body = client.get("/api/localizers", params=q).json()
        return body, {l["key"]: l for l in body["localizers"]}["dino+dinomaly"]

    body, covered = row(reference="data/benchmark_39T/39T-cam53_08-55-37_t60.jpg")
    assert covered["reference_ok"] is True and "39T-cam53" in covered["reference_note"]
    assert body["dinomaly_cameras"] == ["39T-cam53"]

    _, orphan = row(reference="some_other_camera_ref.png")
    assert orphan["reference_ok"] is False
    assert orphan["available"] is True          # a WARNING, not a block
    assert "39T-cam53" in orphan["reference_note"]     # names what IS trained

    # the reference is matched by NAME, never opened: the wizard asks about a
    # reference the operator has not committed to yet
    _, unknown = row(reference="/nowhere/39T-cam53_whatever.jpg")
    assert unknown["reference_ok"] is True

    # and the arms that need no checkpoint stay silent rather than saying "n/a"
    others, _ = row(reference="anything.jpg")
    assert all(not l["reference_note"]
               for l in others["localizers"] if l["key"] != "dino+dinomaly")


def test_job_records_the_localizer_it_ran(api):
    """A finished job must say which localizer produced its boxes - two runs that
    differ only by the proposal stage would otherwise be indistinguishable in the
    history, the compare view and the xlsx."""
    client, _ = api(["NO, nothing on the empty floor."] * 8)
    frame, ref = str(app_image("locrun.jpg")), str(app_image("locrun-ref.jpg"))
    r = client.post("/api/jobs", json={"script": "vlm_05", "frames": [frame],
                                       "reference": ref, "localizer": "photo",
                                       "params": {"PERSON_FILTER": False}})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    wait_job(client, job_id)
    cfg = client.get(f"/api/jobs/{job_id}").json()["config"]
    assert cfg["localizer"] == "photo"
    saved = client.get(f"/api/jobs/{job_id}/results.json").json()
    assert saved["config"]["localizer"] == "photo"
    md = client.get(f"/api/jobs/{job_id}/report.md").text
    assert "Localizer" in md and "photo" in md


def test_a_job_from_before_the_picker_is_named_photo_not_blank():
    """Jobs that predate the localizer picker carry no field - but they DID use
    one: the shipped pixel diff, which is still the default. Showing them blank
    next to a photo+dino run reads as "unknown localizer" and makes the two look
    incomparable, so they are named photo and flagged inferred. The flag is what
    the reports print; the xlsx column keeps the plain name so it stays sortable
    (two spellings of photo would split the column)."""
    from app.backend.exports import localizer_of
    assert localizer_of({"script": "vlm_05", "localizer": "photo+dino"}) == ("photo+dino", True)
    assert localizer_of({"script": "vlm_05"}) == ("photo", False)
    assert localizer_of({"script": "vlm_01", "model": "m"}) == ("", True)


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
    exhausted) - the job must complete with the failure isolated to frame 2."""
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


def test_masked_job_serves_masked_images_to_the_ui(api):
    """A masked job must expose the masked copies everywhere the UI draws a
    frame: the live event stream, the frame list and the reference."""
    client, _ = api(replies=[VALID_VLM01])
    spec = {"name": "pytest-live-mask", "image_size": [120, 90],
            "zones": [{"id": "1", "label": "win",
                       "polygon": [[0, 0], [60, 0], [60, 45], [0, 45]]}]}
    assert client.post("/api/masks", json=spec).status_code == 200
    frame, ref = app_image("m-frame.jpg"), app_image("m-ref.jpg")
    r = client.post("/api/jobs", json={
        "script": "vlm_02", "frames": [str(frame)], "reference": str(ref),
        "mask": "pytest-live-mask", "model": "qwen3-vl:8b-instruct"})
    job_id = r.json()["job_id"]
    data = wait_job(client, job_id)

    masked = f"/jobs/{job_id}/masked/"          # tolerant of the APP_DATA root
    assert masked in data["frames"][0]["img"]
    assert masked in data["config"]["reference_masked_img"]
    assert masked not in data["config"]["reference_img"]   # the untouched pick

    with client.stream("GET", f"/api/jobs/{job_id}/events") as s:
        events = [json.loads(line[6:]) for line in s.iter_lines()
                  if line.startswith("data: ")]
    by_event = {e["event"]: e for e in events}
    assert masked in by_event["mask_applied"]["reference_img"]
    assert masked in by_event["frame_start"]["img"]
    assert masked in by_event["frame_done"]["img"]
    # the verdict rides along so the run screen can render it live
    assert by_event["frame_done"]["anomaly"] is True
    assert by_event["frame_done"]["detections"][0]["label"] == "Phone"


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


def test_cpu_only_machine_gets_a_workable_vlm_timeout(api, monkeypatch):
    """The runner's default timeout is 120 s but one crop call takes 2-4 min on
    CPU, so every uncached frame used to fail with ReadTimeout and the pipeline
    looked broken when it was only slow. An explicit timeout still wins."""
    client, _ = api(["GRAFFITI: no\nVANDALISM: no\nFORGOTTEN OBJECT: no"] * 4)
    monkeypatch.setattr(backend, "has_gpu", lambda: False)
    frame = str(app_image("cpu-t.jpg"))
    jid = client.post("/api/jobs", json={"script": "vlm_01",
                                        "frames": [frame]}).json()["job_id"]
    wait_job(client, jid)
    cfg = client.get(f"/api/jobs/{jid}").json()["config"]
    assert cfg["params"]["timeout_s"] == backend.CPU_TIMEOUT_S

    jid2 = client.post("/api/jobs", json={"script": "vlm_01", "frames": [frame],
                                          "params": {"timeout_s": 30}}).json()["job_id"]
    wait_job(client, jid2)
    cfg2 = client.get(f"/api/jobs/{jid2}").json()["config"]
    assert cfg2["params"]["timeout_s"] == 30


def test_force_cancel_aborts_the_call_in_flight(api, monkeypatch):
    """A graceful cancel waits for the VLM call in flight - up to a few minutes on
    CPU. Force stop aborts its connection, which is the only way to stop Ollama
    mid-generation (measured: Client.close() leaves the reader blocked, closing the
    fd does not wake it, socket.shutdown() does). Everything already judged is
    still kept."""
    client, impl = api(["GRAFFITI: no\nVANDALISM: no\nFORGOTTEN OBJECT: no"] * 6)
    aborted = {"n": 0}

    def fake_abort():
        aborted["n"] += 1
        return 1
    frames = [str(app_image(f"fc{i}.jpg")) for i in range(2)]
    jid = client.post("/api/jobs", json={"script": "vlm_01",
                                         "frames": frames}).json()["job_id"]
    wait_job(client, jid)
    job = backend.manager.get(jid)
    monkeypatch.setattr(job.client, "abort", fake_abort)

    # a finished job has nothing to abort, but the endpoint must stay well-behaved
    r = client.post(f"/api/jobs/{jid}/cancel", json={"force": True})
    assert r.status_code == 200 and r.json()["forced"] is True
    assert r.json()["aborted_calls"] == 0 and aborted["n"] == 0   # not running

    job.status = "running"                      # now it has a call to abort
    r = client.post(f"/api/jobs/{jid}/cancel", json={"force": True})
    assert r.json()["aborted_calls"] == 1 and aborted["n"] == 1
    assert job.cancel_flag.is_set() and job.forced is True
    assert client.post("/api/jobs/nope/cancel", json={"force": True}).status_code == 404


def test_plain_cancel_does_not_abort_anything(api, monkeypatch):
    client, _ = api(["GRAFFITI: no\nVANDALISM: no\nFORGOTTEN OBJECT: no"] * 4)
    jid = client.post("/api/jobs", json={
        "script": "vlm_01", "frames": [str(app_image("pc.jpg"))]}).json()["job_id"]
    wait_job(client, jid)
    job = backend.manager.get(jid)
    calls = []
    monkeypatch.setattr(job.client, "abort", lambda: calls.append(1) or 1)
    job.status = "running"
    r = client.post(f"/api/jobs/{jid}/cancel")
    assert r.status_code == 200 and r.json()["forced"] is False
    assert calls == [] and job.cancel_flag.is_set()
