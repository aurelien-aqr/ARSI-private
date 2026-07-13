"""End-to-end vlm_05 adapter on synthetic images: the real localizer finds the
synthetic object, a fake judge names it, verdicts land in (and come back
from) the cache — keyed with and without a mask hash."""
import pytest

from arsi_core.adapters import run_frame
from arsi_core.cache import VerdictCache

# a bright 60x60 square on a mid-grey scene: photometric diff >> threshold 40
REF_KW = dict(size=(400, 300), color=(128, 128, 128))
INSP_RECTS = [((200, 150, 260, 210), (250, 250, 250))]

PARAMS = {"PERSON_FILTER": False}   # skip the YOLO person veto in unit tests


@pytest.fixture
def cache(tmp_path):
    return VerdictCache(path=tmp_path / "verdicts.json", seed_paths=())


def test_detects_synthetic_object_and_caches(fake_client, img_factory, cache):
    ref = img_factory("ref.jpg", **REF_KW)
    insp = img_factory("insp.jpg", **REF_KW, rects=INSP_RECTS)
    client = fake_client(["YES, white box on floor."] * 5)

    fr = run_frame("vlm_05", insp, reference=ref, client=client,
                   params=PARAMS, cache=cache)
    assert fr.status == "ok" and fr.anomaly is True
    assert len(fr.detections) >= 1
    bbox = fr.detections[0].bbox
    assert bbox[0] <= 210 and bbox[2] >= 250    # box covers the square
    assert "white box" in fr.detections[0].label
    n_calls = len(client._impl.calls)
    assert n_calls >= 1 and len(cache) == n_calls

    # second run: served entirely from cache, zero VLM calls
    client2 = fake_client([])                    # would raise if called
    fr2 = run_frame("vlm_05", insp, reference=ref, client=client2,
                    params=PARAMS, cache=cache)
    assert fr2.anomaly is True
    assert len(fr2.detections) == len(fr.detections)


def test_person_and_disappear_labels_are_dropped(fake_client, img_factory, cache):
    ref = img_factory("ref.jpg", **REF_KW)
    insp = img_factory("insp.jpg", **REF_KW, rects=INSP_RECTS)
    client = fake_client(["YES, a person's arm."] * 5)
    fr = run_frame("vlm_05", insp, reference=ref, client=client,
                   params=PARAMS, cache=cache)
    assert fr.anomaly is False and fr.detections == []


def test_mask_hash_partitions_the_cache(fake_client, img_factory, cache):
    ref = img_factory("ref.jpg", **REF_KW)
    insp = img_factory("insp.jpg", **REF_KW, rects=INSP_RECTS)
    run_frame("vlm_05", insp, reference=ref, params=PARAMS, cache=cache,
              client=fake_client(["YES, white box on floor."] * 5))
    # same images, but a mask is now active: cached unmasked verdicts must NOT
    # be served — the judge is called again under the mask-suffixed key
    client = fake_client(["NO, same empty floor."] * 5)
    fr = run_frame("vlm_05", insp, reference=ref, params=PARAMS, cache=cache,
                   client=client, mask_hash="abc123def456")
    assert len(client._impl.calls) >= 1
    assert fr.anomaly is False


def test_clean_pair_yields_no_regions(fake_client, img_factory, cache):
    ref = img_factory("ref.jpg", **REF_KW)
    insp = img_factory("insp.jpg", **REF_KW)     # identical scene
    fr = run_frame("vlm_05", insp, reference=ref, client=fake_client([]),
                   params=PARAMS, cache=cache)
    assert fr.anomaly is False and fr.detections == []
