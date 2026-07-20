import json

import pytest

from arsi_core.errors import ModelMissing
from arsi_core.masking import MaskSpec
from arsi_core.runner import JobConfig, run_job

CLEAN = "GRAFFITI: no\nVANDALISM: no\nFORGOTTEN OBJECT: no"
ANOM = "GRAFFITI: no\nVANDALISM: no\nFORGOTTEN OBJECT: yes\n  - phone (seat)"


def make_cfg(tmp_path, frames, **kw):
    kw.setdefault("script", "vlm_01")
    kw.setdefault("model", "qwen3-vl:8b-instruct")
    return JobConfig(frames=[str(f) for f in frames],
                     job_dir=tmp_path / "job", **kw)


def test_batch_continues_after_parse_failures(fake_client, img_factory, tmp_path):
    f1, f2 = img_factory("f1.jpg"), img_factory("f2.jpg")
    # f1: garbage on every attempt (1 + 2 retries); f2: clean on first try
    client = fake_client(["garbage", "still garbage", "garbage again", CLEAN])
    events = []
    result = run_job(make_cfg(tmp_path, [f1, f2], params={"max_retries": 2}),
                     on_event=events.append, client=client)

    assert [f.status for f in result.frames] == ["failed", "ok"]
    assert result.frames[0].attempts == 3
    assert result.summary.n_failed == 1 and result.summary.n_ok == 1
    assert result.status == "completed"        # partial failure != job failure
    retries = [e for e in events if e["event"] == "frame_retry"]
    assert len(retries) == 3
    # the retry prompt carries the format reminder
    second_call = client._impl.calls[1]["messages"][0]["content"]
    assert "REMINDER" in second_call

    # results.json + job.log written
    job_dir = tmp_path / "job"
    saved = json.loads((job_dir / "results.json").read_text())
    assert saved["summary"]["n_frames"] == 2
    assert (job_dir / "job.log").exists()


def test_anomaly_counting(fake_client, img_factory, tmp_path):
    frames = [img_factory(f"f{i}.jpg") for i in range(3)]
    client = fake_client([CLEAN, ANOM, CLEAN])
    result = run_job(make_cfg(tmp_path, frames), client=client)
    assert result.summary.n_anomalous == 1
    assert [f.anomaly for f in result.frames] == [False, True, False]


def test_missing_model_aborts_before_any_frame(fake_client, img_factory, tmp_path):
    client = fake_client([], models=["something-else"])
    with pytest.raises(ModelMissing):
        run_job(make_cfg(tmp_path, [img_factory("f1.jpg")]), client=client)
    assert client._impl.calls == []


def test_mask_is_materialized_for_frames_and_reference(fake_client, img_factory, tmp_path):
    ref = img_factory("ref.jpg", color=(200, 200, 200))
    f1 = img_factory("f1.jpg", color=(200, 200, 200))
    mask = MaskSpec(name="win", image_size=[400, 300], zones=[
        {"id": "z", "label": "w", "polygon": [[0, 0], [100, 0], [100, 50], [0, 50]]}])
    mask_path = mask.save(tmp_path / "win.json")

    client = fake_client([CLEAN])
    result = run_job(make_cfg(tmp_path, [f1], script="vlm_02",
                              reference=str(ref), mask=str(mask_path)),
                     client=client)
    assert result.frames[0].status == "ok"
    sent = client._impl.calls[0]["messages"][0]["images"]
    masked_dir = tmp_path / "job" / "masked"
    assert sent == [str(masked_dir / "ref.jpg"), str(masked_dir / "f1.jpg")]
    from PIL import Image
    with Image.open(sent[1]) as img:
        assert img.getpixel((50, 25)) == (0, 0, 0)


def test_masked_run_reports_masked_paths_to_the_ui(fake_client, img_factory, tmp_path):
    """The run screen and the results page render whatever the events and the
    saved config point at, so those must be the masked copies — showing the
    untouched frame would misrepresent what the VLM was given."""
    ref = img_factory("ref.jpg", color=(200, 200, 200))
    f1 = img_factory("f1.jpg", color=(200, 200, 200))
    mask = MaskSpec(name="win", image_size=[400, 300], zones=[
        {"id": "z", "label": "w", "polygon": [[0, 0], [100, 0], [100, 50], [0, 50]]}])
    mask_path = mask.save(tmp_path / "win.json")

    events = []
    result = run_job(make_cfg(tmp_path, [f1], script="vlm_02",
                              reference=str(ref), mask=str(mask_path)),
                     client=fake_client([CLEAN]), on_event=events.append)
    masked_dir = str(tmp_path / "job" / "masked")

    assert result.frames[0].image.startswith(masked_dir)
    assert result.config["reference_masked"].startswith(masked_dir)
    by_event = {e["event"]: e for e in events}
    assert by_event["mask_applied"]["reference"].startswith(masked_dir)
    assert by_event["frame_start"]["frame"].startswith(masked_dir)
    assert by_event["frame_done"]["frame"].startswith(masked_dir)


def test_frame_events_carry_the_verdict_for_the_live_view(fake_client, img_factory,
                                                          tmp_path):
    events = []
    run_job(make_cfg(tmp_path, [img_factory("f1.jpg")], script="vlm_01"),
            client=fake_client([CLEAN]), on_event=events.append)
    starts = [e for e in events if e["event"] == "frame_start"]
    done = [e for e in events if e["event"] == "frame_done"]
    assert [e["index"] for e in starts] == [0]
    assert starts[0]["frame_id"] == "f1"
    # detections travel with the event so the run screen can draw boxes before
    # results.json exists
    assert done[0]["detections"] == []
    assert done[0]["anomaly"] is False


def test_default_model_is_checked_upfront(fake_client, img_factory, tmp_path):
    # model=None resolves to the script default, which must still be verified
    client = fake_client([], models=["something-else"])
    with pytest.raises(ModelMissing):
        run_job(make_cfg(tmp_path, [img_factory("f1.jpg")], model=None), client=client)


def test_transport_retry_keeps_original_prompt(fake_client, img_factory, tmp_path):
    from arsi_core.errors import VLMCallError
    client = fake_client([VLMCallError("timeout"), CLEAN])
    result = run_job(make_cfg(tmp_path, [img_factory("f1.jpg")]), client=client)
    assert result.frames[0].status == "ok" and result.frames[0].attempts == 2
    # no format reminder on a transport retry (it would change the vlm_05
    # cache fingerprint); the reminder is reserved for ParseError retries
    assert "REMINDER" not in client._impl.calls[1]["messages"][0]["content"]


def test_cancel_between_frames_keeps_partials(fake_client, img_factory, tmp_path):
    frames = [img_factory(f"f{i}.jpg") for i in range(3)]
    client = fake_client([CLEAN, CLEAN, CLEAN])
    stop_after = {"n": 1}

    def stop():
        return sum(1 for _ in ()) or stop_after["n"] <= 0

    def on_event(e):
        if e["event"] == "frame_done":
            stop_after["n"] -= 1

    result = run_job(make_cfg(tmp_path, frames), client=client,
                     on_event=on_event, stop=stop)
    assert result.status == "cancelled"
    assert len(result.frames) == 1 and result.frames[0].status == "ok"
