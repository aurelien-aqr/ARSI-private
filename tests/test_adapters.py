import pytest

from arsi_core.adapters import (parse_structured_report, parse_bbox_json,
                                scale_bboxes, run_frame)
from arsi_core.errors import FrameError, ParseError

GOOD_REPORT = """GRAFFITI: no - clean walls
VANDALISM: yes - torn seat near the door
FORGOTTEN OBJECT: yes
  - black phone (right-hand seats)
  - wallet (floor)
DESCRIPTION: two objects and a torn seat.
SEVERITY: 3
"""


def test_parse_structured_report_full():
    anomaly, dets = parse_structured_report(GOOD_REPORT)
    assert anomaly is True
    labels = {(d.label, d.type, d.zone) for d in dets}
    assert ("black phone", "object", "right-hand seats") in labels
    assert ("wallet", "object", "floor") in labels
    assert any(d.type == "damage" for d in dets)
    assert all(d.severity == 3 for d in dets)


def test_parse_structured_report_clean():
    anomaly, dets = parse_structured_report(
        "GRAFFITI: no\nVANDALISM: no\nFORGOTTEN OBJECT: no\nSEVERITY: 1")
    assert anomaly is False and dets == []


def test_parse_structured_report_garbage_raises():
    with pytest.raises(ParseError):
        parse_structured_report("The tram looks fine to me!")


def test_parse_bbox_json_fenced_and_broken():
    assert parse_bbox_json('```json\n[{"label": "graffiti", "bbox": [0, 0, 1, 1], '
                           '"severity": 2}]\n```')[0]["label"] == "graffiti"
    assert parse_bbox_json("[]") == []
    with pytest.raises(ParseError):
        parse_bbox_json('[{"label": "graffiti", "bbox": [0,0,1,]}]')  # bad JSON
    with pytest.raises(ParseError):
        parse_bbox_json("no boxes here")


def test_scale_bboxes_heuristic():
    # 0-1000 scale (Qwen default) on a 2000x1000 image -> x2, x1
    dets = scale_bboxes([{"label": "forgotten_object", "bbox": [100, 100, 200, 200]}],
                        2000, 1000)
    assert dets[0].bbox == [200, 100, 400, 200]
    assert dets[0].type == "object"
    # 0-1 normalized
    dets = scale_bboxes([{"label": "vandalism", "bbox": [0.5, 0.5, 1.0, 1.0]}], 200, 100)
    assert dets[0].bbox == [100, 50, 200, 100]
    assert dets[0].type == "damage"


def test_run_frame_vlm01(fake_client, img_factory):
    img = img_factory("f1.jpg")
    fr = run_frame("vlm_01", img, client=fake_client([GOOD_REPORT]))
    assert fr.status == "ok" and fr.anomaly is True
    assert len(fr.detections) == 3
    assert fr.raw_response.startswith("GRAFFITI")


def test_run_frame_vlm02_sends_both_images(fake_client, img_factory):
    ref, img = img_factory("ref.jpg"), img_factory("f1.jpg")
    client = fake_client(["GRAFFITI: no\nVANDALISM: no\nFORGOTTEN OBJECT: no"])
    fr = run_frame("vlm_02", img, reference=ref, client=client)
    assert fr.anomaly is False
    sent = client._impl.calls[0]["messages"][0]["images"]
    assert sent == [str(ref), str(img)]     # reference first — order matters


def test_run_frame_vlm02_without_reference_fails(fake_client, img_factory):
    with pytest.raises(FrameError):
        run_frame("vlm_02", img_factory("f1.jpg"), client=fake_client([]))


def test_run_frame_vlm03(fake_client, img_factory):
    img = img_factory("f1.jpg", size=(1000, 500))
    reply = '[{"label": "forgotten_object", "bbox": [0.1, 0.2, 0.3, 0.4], "severity": 2}]'
    fr = run_frame("vlm_03", img, client=fake_client([reply]))
    assert fr.anomaly is True
    assert fr.detections[0].bbox == [100, 100, 300, 200]


def test_run_frame_missing_image(fake_client):
    with pytest.raises(FrameError):
        run_frame("vlm_01", "does/not/exist.jpg", client=fake_client([]))


def test_parse_structured_report_inline_object_note():
    anomaly, dets = parse_structured_report(
        "GRAFFITI: no\nVANDALISM: no\nFORGOTTEN OBJECT: yes - a black phone on a seat")
    assert anomaly is True
    assert [d.label for d in dets] == ["a black phone on a seat"]
