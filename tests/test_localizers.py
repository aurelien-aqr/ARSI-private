"""The localizer is a stage you pick, not a hardcoded half of vlm_05.

The DINOv2 tests skip when torch or the cached backbone is absent, so the suite
stays green on a machine that never ran a feature localizer.
"""
import json

import pytest

from arsi_core import localizers
from arsi_core.adapters import get_module, run_frame
from arsi_core.cache import VerdictCache
from arsi_core.errors import FrameError

REF_KW = dict(size=(400, 300), color=(128, 128, 128))
INSP_RECTS = [((200, 150, 260, 210), (250, 250, 250))]
PARAMS = {"PERSON_FILTER": False}

dino_only = pytest.mark.skipif(
    not localizers.availability("photo+dino")[0] or not localizers._weights_cached(),
    reason="needs torch and the cached DINOv2 backbone")


@pytest.fixture
def cache(tmp_path):
    return VerdictCache(path=tmp_path / "verdicts.json", seed_paths=())


def test_registry_shape():
    assert localizers.DEFAULT in localizers.names()
    rows = {r["key"]: r for r in localizers.catalog()}
    assert set(rows) == set(localizers.names())
    assert rows["photo"]["available"] is True          # never needs torch
    assert rows["photo+dino"]["recommended"] is True
    for r in rows.values():
        assert r["summary"]                            # the card renders this
        assert "measured" not in r                     # kept out of the payload


def test_unknown_localizer_is_rejected_before_the_job():
    with pytest.raises(FrameError):
        localizers.check("magic")
    assert localizers.availability("magic")[0] is False


def test_photo_is_the_shipped_localizer_untouched(img_factory):
    """DEFAULT must stay byte-identical to vlm_05.localize() — every published
    benchmark number and every job in the history was produced with it."""
    ref = img_factory("ref.jpg", **REF_KW)
    insp = img_factory("insp.jpg", **REF_KW, rects=INSP_RECTS)
    m = get_module("vlm_05")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "PERSON_FILTER", False)
        direct, _ = m.localize(str(ref), str(insp))
        viareg, info = localizers.localize("photo", m, str(ref), str(insp))
    assert [r["bbox"] for r in direct] == [r["bbox"] for r in viareg]
    assert info["localizer"] == "photo"


@dino_only
def test_gate_is_a_strict_subset_of_the_pixel_diff(img_factory):
    """The property the whole cheap-A/B story rests on: the gate only DROPS
    boxes, keeping the survivors' coordinates identical, so their cached
    verdicts stay valid. If this ever stops holding, a localizer A/B silently
    becomes a full re-run."""
    ref = img_factory("ref.jpg", **REF_KW)
    insp = img_factory("insp.jpg", **REF_KW, rects=INSP_RECTS)
    m = get_module("vlm_05")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "PERSON_FILTER", False)
        photo, _ = localizers.localize("photo", m, str(ref), str(insp))
        gated, info = localizers.localize("photo+dino", m, str(ref), str(insp))
    photo_boxes = [tuple(r["bbox"]) for r in photo]
    gated_boxes = [tuple(r["bbox"]) for r in gated]
    assert set(gated_boxes) <= set(photo_boxes)
    assert len(gated_boxes) == len(set(gated_boxes))
    assert info["localizer"] == "photo+dino"
    assert info["gated_away"] == len(photo_boxes) - len(gated_boxes)


@dino_only
def test_gate_keeps_a_real_object_and_reuses_the_cache(fake_client, img_factory, cache):
    ref = img_factory("ref.jpg", **REF_KW)
    insp = img_factory("insp.jpg", **REF_KW, rects=INSP_RECTS)

    fr = run_frame("vlm_05", insp, reference=ref, params=PARAMS, cache=cache,
                   client=fake_client(["YES, white box on floor."] * 5),
                   localizer="photo+dino")
    assert fr.anomaly is True and fr.detections
    bbox = fr.detections[0].bbox
    assert bbox[0] <= 210 and bbox[2] >= 250
    assert "localizer" in fr.raw_response and "photo+dino" in fr.raw_response

    # the verdicts were cached under the box coordinates, not the localizer, so
    # re-running with the plain pixel diff must not re-call the judge for them
    before = len(cache)
    run_frame("vlm_05", insp, reference=ref, params=PARAMS, cache=cache,
              client=fake_client(["YES, white box on floor."] * 20),
              localizer="photo")
    assert len(cache) >= before          # may add boxes the gate had dropped


@dino_only
def test_dino_alone_localizes_the_object(img_factory):
    ref = img_factory("ref.jpg", **REF_KW)
    insp = img_factory("insp.jpg", **REF_KW, rects=INSP_RECTS)
    m = get_module("vlm_05")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "PERSON_FILTER", False)
        regions, info = localizers.localize("dino", m, str(ref), str(insp))
    assert info["localizer"] == "dino"
    assert any(r["bbox"][0] <= 215 and r["bbox"][2] >= 245 for r in regions)


# The API-surface tests for this live in test_backend.py, where the `api`
# fixture and the APP_DATA image helper are defined.


def test_job_config_records_the_localizer():
    from arsi_core.runner import JobConfig
    cfg = JobConfig(script="vlm_05", frames=["a.jpg"], localizer="photo+dino")
    assert cfg.public_dict()["localizer"] == "photo+dino"
    assert json.dumps(cfg.public_dict())    # must stay JSON-serialisable


# --- the localization stage must be inspectable afterwards -------------------

def test_candidates_record_which_stage_dropped_each_region(fake_client, img_factory,
                                                           cache):
    """A miss has two causes — never localized, or localized and rejected — and
    the frame result must let you tell them apart without reading raw text."""
    ref = img_factory("ref.jpg", **REF_KW)
    insp = img_factory("insp.jpg", **REF_KW, rects=INSP_RECTS)
    fr = run_frame("vlm_05", insp, reference=ref, params=PARAMS, cache=cache,
                   client=fake_client(["NO, same empty floor."] * 20))
    assert fr.detections == []                       # nothing kept
    assert fr.candidates                             # but regions WERE proposed
    assert all(c["outcome"] == "rejected" for c in fr.candidates)
    assert all(c["bbox"] and c["verdict"] == "no" for c in fr.candidates)
    # a judge NO must not be attributed to one of our post-filters
    assert all(not c["dropped_by"] for c in fr.candidates)
    st = fr.localization
    assert st["name"] == localizers.DEFAULT
    assert st["proposed"] == len(fr.candidates) and st["kept"] == 0
    assert st["rejected"] == len(fr.candidates)


def test_a_post_filtered_yes_is_not_reported_as_a_judge_no(fake_client, img_factory,
                                                           cache):
    """The judge said YES and one of OUR filters overrode it: that is a bug in the
    filter, not in the model, so it must not be filed under 'judge said NO'."""
    ref = img_factory("ref.jpg", **REF_KW)
    insp = img_factory("insp.jpg", **REF_KW, rects=INSP_RECTS)
    # "a person's arm" is a YES that is_non_anomaly() overrides
    fr = run_frame("vlm_05", insp, reference=ref, params=PARAMS, cache=cache,
                   client=fake_client(["YES, a person's arm."] * 20))
    assert fr.detections == []
    assert fr.candidates and all(c["verdict"] == "yes" for c in fr.candidates)
    assert all(c["outcome"] == "filtered" for c in fr.candidates)
    assert all(c["dropped_by"] for c in fr.candidates)
    assert fr.localization["filtered"] == len(fr.candidates)
    assert fr.localization["rejected"] == 0
