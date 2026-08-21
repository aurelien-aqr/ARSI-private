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

dinomaly_only = pytest.mark.skipif(
    not localizers.availability("dino+dinomaly")[0] or not localizers._weights_cached(),
    reason="needs torch, the cached DINOv2 backbone and a Dinomaly checkpoint")


@pytest.fixture
def cache(tmp_path):
    return VerdictCache(path=tmp_path / "verdicts.json", seed_paths=())


def test_registry_shape():
    assert localizers.DEFAULT in localizers.names()
    rows = {r["key"]: r for r in localizers.catalog()}
    assert set(rows) == set(localizers.names())
    assert rows["photo"]["available"] is True          # never needs torch
    # Exactly one badge, on a real key: the INVARIANT the picker relies on.
    # Which key carries it is a measured call that has already moved once
    # (photo+dino -> dino on 2026-08-19, on the end-to-end numbers in the
    # module docstring), so it is asserted separately and deliberately.
    badged = [k for k, r in rows.items() if r["recommended"]]
    assert len(badged) == 1 and badged[0] in localizers.names()
    assert badged == ["dino"]
    # NOT "dino+dinomaly", however good its cost number: the badge is what a
    # user picks without reading, and that arm silently does nothing on a camera
    # with no checkpoint. Recommending it would recommend "dino" under a name
    # that promises more.
    for r in rows.values():
        assert r["summary"]                            # the card renders this
        assert "measured" not in r                     # kept out of the payload


def test_unknown_localizer_is_rejected_before_the_job():
    with pytest.raises(FrameError):
        localizers.check("magic")
    assert localizers.availability("magic")[0] is False


def test_photo_is_the_shipped_localizer_untouched(img_factory):
    """DEFAULT must stay byte-identical to vlm_05.localize() - every published
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
    """A miss has two causes - never localized, or localized and rejected - and
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


# --- the Dinomaly veto -------------------------------------------------------

def test_checkpoint_glob_matches_dinomaly_naming():
    """localizers.py spells the checkpoint filename itself, to stay torch-free
    on the availability path. This is the pin that stops the two spellings from
    drifting: if dinomaly.checkpoint_path ever changes, this fails instead of
    the arm silently reporting "no checkpoint trained yet" forever."""
    pytest.importorskip("torch")
    import tools.dinomaly as dm
    expected = dm.checkpoint_path("some-cam")
    assert expected.parent == localizers.WEIGHTS_DIR
    assert expected.name == localizers.CKPT_GLOB.replace("*", "some-cam")


def test_camera_comes_from_the_reference_path(tmp_path, monkeypatch):
    monkeypatch.setattr(localizers, "checkpoint_cameras",
                        lambda: ["1760-cam04", "tram_1762"])
    assert localizers.resolve_camera("data/benchmark_1760/1760-cam04_t070.jpg") \
        == "1760-cam04"
    assert localizers.resolve_camera(
        "data/reference/tram_1762_v1_f0227_masked_reference.jpg") == "tram_1762"
    # a scene with no model of its own resolves to nothing rather than to some
    # other camera's model - the fallback is 'dino', never a wrong checkpoint
    assert localizers.resolve_camera(
        "data/reference/tram_variant/tram_variant_reference.png") is None
    # and an explicit param wins over the path
    assert localizers.resolve_camera("whatever.jpg",
                                     {"DINOMALY_CAMERA": "3333-cam53"}) == "3333-cam53"


def test_the_arm_is_offered_only_when_some_checkpoint_exists(monkeypatch):
    monkeypatch.setattr(localizers, "checkpoint_cameras", lambda: [])
    ok, why = localizers.availability("dino+dinomaly")
    assert ok is False and "dinomaly_train" in why
    with pytest.raises(FrameError):
        localizers.check("dino+dinomaly")
    # the other three do not depend on a checkpoint
    assert localizers.availability("photo")[0] is True


@dino_only
def test_veto_falls_back_to_dino_on_an_unknown_camera(img_factory):
    """A tmp_path reference matches no checkpoint. The contract is that this
    degrades to exactly 'dino' - same boxes, nothing deleted - and SAYS so."""
    ref = img_factory("ref.jpg", **REF_KW)
    insp = img_factory("insp.jpg", **REF_KW, rects=INSP_RECTS)
    m = get_module("vlm_05")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "PERSON_FILTER", False)
        plain, _ = localizers.localize("dino", m, str(ref), str(insp))
        fell, info = localizers.localize("dino+dinomaly", m, str(ref), str(insp))
    assert [r["bbox"] for r in fell] == [r["bbox"] for r in plain]
    assert info["localizer"] == "dino+dinomaly"
    assert info["dinomaly_camera"] is None
    assert info["vetoed"] == 0 and "dino" in info["dinomaly"]


@dinomaly_only
def test_veto_is_a_strict_subset_of_dino(img_factory):
    """Same invariant as the DINOv2 gate, for the same reason: the veto only
    DROPS boxes, so the survivors keep byte-identical coordinates and their
    cached verdicts stay valid. This is what makes a dino / dino+dinomaly A/B
    cost only the frames the first arm already paid for."""
    cam = localizers.checkpoint_cameras()[0]
    ref = img_factory("ref.jpg", **REF_KW)
    insp = img_factory("insp.jpg", **REF_KW, rects=INSP_RECTS)
    m = get_module("vlm_05")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "PERSON_FILTER", False)
        plain, _ = localizers.localize("dino", m, str(ref), str(insp))
        vetoed, info = localizers.localize("dino+dinomaly", m, str(ref), str(insp),
                                           {"DINOMALY_CAMERA": cam})
    plain_boxes = [tuple(r["bbox"]) for r in plain]
    kept_boxes = [tuple(r["bbox"]) for r in vetoed]
    assert set(kept_boxes) <= set(plain_boxes)
    assert info["dinomaly_camera"] == cam
    assert info["vetoed"] == len(plain_boxes) - len(kept_boxes)


@dinomaly_only
def test_warmup_reports_whether_the_veto_will_fire(img_factory):
    """Whether this job gets the veto at all is a fact about the job, and the
    operator must learn it when it starts - not by reading a per-frame info key
    after paying for the run."""
    cam = localizers.checkpoint_cameras()[0]
    covered = img_factory(f"{cam}_ref.jpg", **REF_KW)
    unknown = img_factory("nothing_matches.jpg", **REF_KW)
    assert cam in localizers.warmup("dino+dinomaly", [str(covered)])
    assert "will not fire" in localizers.warmup("dino+dinomaly", [str(unknown)])
    mixed = localizers.warmup("dino+dinomaly", [str(covered), str(unknown)])
    assert cam in mixed and "no checkpoint" in mixed
    assert localizers.warmup("photo") == ""       # nothing to say, and no crash
