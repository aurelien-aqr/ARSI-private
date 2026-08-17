"""arsi_core.benchmarks - dataset I/O, validation, and the scoring rules.

The point of the last test class: the Studio scores a benchmark run through
`score_full`, while `benchmark/run_benchmark.py` scores the CLI run through its
own `metrics()`. Two implementations of one definition drift silently, so the
matching rules are locked to vlm_05's, and the aggregation is locked to the
numbers of the published report in benchmark/archive/.
"""
import json

import pytest

import vlm_05_reference_diff as vlm05
from arsi_core import benchmarks as B

ARCHIVE = B.REPO_ROOT / "benchmark" / "archive"


# --------------------------------------------------------------- matching rules

BOXES = [
    ([0, 0, 100, 100], [0, 0, 100, 100]),        # identical
    ([0, 0, 100, 100], [50, 50, 150, 150]),      # partial
    ([0, 0, 100, 100], [90, 90, 200, 200]),      # slight, IoU < 0.1
    ([0, 0, 100, 100], [200, 200, 300, 300]),    # disjoint
    ([0, 0, 100, 100], [10, 10, 20, 20]),        # contained (centre inside)
    ([0, 0, 400, 400], [180, 180, 220, 220]),    # mega-blob over a small box
    ([0, 0, 10, 10], [10, 10, 20, 20]),          # touching corners, zero area
]


@pytest.mark.parametrize("a,b", BOXES)
def test_iou_matches_vlm05(a, b):
    assert B.iou(a, b) == vlm05._iou(a, b)
    assert B.iou(b, a) == vlm05._iou(b, a)


@pytest.mark.parametrize("a,b", BOXES)
def test_overlaps_matches_vlm05(a, b):
    assert B.overlaps(a, b) == vlm05._boxes_overlap(a, b)
    assert B.overlaps(b, a) == vlm05._boxes_overlap(b, a)


def test_strict_iou_threshold_is_the_published_one():
    # 0.3 is what every "strict IoU>=0.3" column in benchmark/README.md means
    assert B.STRICT_IOU == 0.3


# --------------------------------------------------------------- load / save

def test_ships_one_benchmark():
    """One protocol over every labelled frame: the two trams are cameras of the
    same benchmark, not two datasets."""
    assert B.list_ids() == [B.DEFAULT]


def test_load_accepts_id_and_path():
    by_id, doc = B.load(B.DEFAULT)
    by_path, doc2 = B.load(f"benchmark/datasets/{B.DEFAULT}.json")
    assert by_id == by_path == B.DEFAULT
    assert doc == doc2


def test_load_unknown_dataset_raises():
    with pytest.raises(B.DatasetError):
        B.load("nope")


def test_shipped_dataset_is_valid():
    _, doc = B.load()
    errors, _ = B.validate(doc)
    assert errors == []


def test_shipped_dataset_round_trips_byte_identical():
    """The writer must reproduce the committed file exactly, or the first save
    from the Studio would rewrite all 50 cases and bury the real edit."""
    _, doc = B.load()
    assert B.dumps(doc) + "\n" == B.resolve(B.DEFAULT).read_text(encoding="utf-8")


def test_dumps_is_reloadable_and_stable():
    _, doc = B.load()
    reparsed = json.loads(B.dumps(doc))
    assert reparsed == doc
    assert B.dumps(reparsed) == B.dumps(doc)


def test_digest_tracks_labels_not_formatting():
    _, doc = B.load()
    same = json.loads(json.dumps(doc, indent=4))
    assert B.digest(same) == B.digest(doc)
    moved = json.loads(json.dumps(doc))
    moved["cases"][0]["instances"][0]["bbox"][0] += 1
    assert B.digest(moved) != B.digest(doc)


def test_save_writes_backup_and_rejects_invalid(tmp_path, monkeypatch):
    _, doc = B.load()                       # before redirecting the dir
    monkeypatch.setattr(B, "DATASETS_DIR", tmp_path)
    monkeypatch.setattr(B, "BACKUPS_DIR", tmp_path / ".backups")
    B.save("copy", doc)
    assert (tmp_path / "copy.json").is_file()
    doc["cases"][0]["instances"][0]["bbox"] = [10, 10, 5, 5]      # inverted
    with pytest.raises(B.DatasetError):
        B.save("copy", doc)
    B.save("copy", B.load(tmp_path / "copy.json")[1])             # unchanged
    assert list((tmp_path / ".backups").glob("copy-*.json"))


def test_a_stray_non_dict_case_is_a_validation_error_not_a_crash():
    """A hand-edited file with a null or a string in `cases` must produce the
    error the API can report, not an AttributeError that 500s the whole screen."""
    errors, _ = B.validate({"references": REFS, "cases": ["oops", None]})
    assert len(errors) == 2 and all("must be an object" in e for e in errors)


def test_dumps_keeps_header_keys_it_does_not_know_about():
    _, doc = B.load()
    doc["provenance"] = {"built_by": "someone else"}
    assert json.loads(B.dumps(doc))["provenance"] == {"built_by": "someone else"}


def test_backups_of_two_saves_in_the_same_second_both_survive(tmp_path, monkeypatch):
    _, doc = B.load()
    monkeypatch.setattr(B, "DATASETS_DIR", tmp_path)
    monkeypatch.setattr(B, "BACKUPS_DIR", tmp_path / ".backups")
    B.save("copy", doc)
    B.save("copy", doc)
    B.save("copy", doc)
    assert len(list((tmp_path / ".backups").glob("copy-*.json"))) == 2


def test_summary_counts_the_whole_benchmark():
    """Both trams, one protocol: 29 cases of 1762 (45 instances) + 21 of 39T (24)."""
    ds_id, doc = B.load()
    s = B.summary(ds_id, doc)
    assert (s["n_cases"], s["n_anomalous"], s["n_clean"], s["n_instances"]) \
        == (50, 31, 19, 69)
    assert s["references"] == ["39T-cam52", "39T-cam53", "39T-cam54", "39T-cam55",
                              "real", "variant"]


def test_the_benchmark_mixes_reference_sizes():
    """Why every overlay resolves its coordinate space per case: `real` is
    1920x1080, `variant` 1672x941 and the 39T cameras 1280x720."""
    from PIL import Image
    _, doc = B.load()
    sizes = {k: Image.open(B.REPO_ROOT / v).size
             for k, v in doc["references"].items()}
    assert sizes["real"] == (1920, 1080)
    assert sizes["variant"] == (1672, 941)
    assert sizes["39T-cam52"] == (1280, 720)


# --------------------------------------------------------------- case payloads

REFS = {"real": "data/reference/tram_1762_v1_f0227_masked_reference.jpg"}
IMG = "data/masked/tram_1762_v1_f0151_masked.jpg"


def _payload(**kw):
    base = {"id": "case_x", "image": IMG, "reference": "real", "source": "real",
            "has_anomaly": True, "note": "n",
            "instances": [{"type": "object", "bbox": [10, 10, 40.6, 40],
                           "label": "phone"}]}
    base.update(kw)
    return base


def test_normalize_case_rounds_boxes_and_derives_types():
    case = B.normalize_case(_payload(), REFS)
    assert case["instances"][0]["bbox"] == [10, 10, 41, 40]
    assert case["types"] == ["object"]
    assert list(case) == ["id", "image", "reference", "source", "has_anomaly",
                          "types", "note", "instances"]


def test_normalize_case_drops_unknown_keys_and_bad_types():
    case = B.normalize_case(_payload(sneaky="x", instances=[
        {"type": "nonsense", "bbox": [1, 2, 3, 4]}]), REFS)
    assert "sneaky" not in case
    assert case["instances"][0]["type"] == "unknown"


@pytest.mark.parametrize("payload", [
    {"id": "bad id"},
    {"id": "ok", "reference": "missing"},
    {"id": "ok", "instances": [{"type": "object", "bbox": [5, 5, 5, 9]}]},
    {"id": "ok", "instances": [{"type": "object", "bbox": [1, 2, 3]}]},
])
def test_normalize_case_rejects(payload):
    with pytest.raises(B.DatasetError):
        B.normalize_case(_payload(**payload), REFS)


def test_clean_case_with_boxes_is_an_error_but_anomaly_without_boxes_is_a_warning():
    doc = {"references": REFS, "cases": [
        B.normalize_case(_payload(id="a", has_anomaly=False, instances=[]), REFS)]}
    doc["cases"][0]["instances"] = [{"type": "object", "bbox": [1, 1, 9, 9]}]
    errors, _ = B.validate(doc)
    assert any("labelled clean" in e for e in errors)

    doc["cases"][0].update(has_anomaly=True, instances=[])
    errors, warnings = B.validate(doc)
    assert errors == []
    assert any("no instance box" in w for w in warnings)


# --------------------------------------------------------------- scoring

def _frame(dets, **kw):
    d = {"frame_id": "f", "image": "i.jpg", "status": "ok",
         "anomaly": bool(dets), "seconds": 1.0,
         "detections": [{"bbox": b, "label": "x", "type": "object"} for b in dets],
         "candidates": [{"bbox": b, "outcome": "kept"} for b in dets],
         "localization": {"name": "photo", "proposed": len(dets)}}
    d.update(kw)
    return d


def test_score_case_frame_outcomes():
    anom = {"id": "a", "has_anomaly": True, "instances": [
        {"type": "object", "bbox": [0, 0, 100, 100]}]}
    clean = {"id": "c", "has_anomaly": False, "instances": []}
    assert B.score_case(anom, _frame([[10, 10, 90, 90]]))["outcome"] == "TP"
    assert B.score_case(anom, _frame([]))["outcome"] == "FN"
    assert B.score_case(clean, _frame([[0, 0, 50, 50]]))["outcome"] == "FP"
    assert B.score_case(clean, _frame([]))["outcome"] == "TN"


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_incomplete_frame_is_not_counted_as_a_prediction(status):
    """A crashed frame must not be scored as 'predicted clean' - that would turn
    infrastructure failures into recall. `cancelled` counts as incomplete too:
    the runner builds that status for a frame stopped part-way through its
    regions, so its instances were never judged either."""
    case = {"id": "a", "has_anomaly": True,
            "instances": [{"type": "object", "bbox": [0, 0, 10, 10]}]}
    score = B.score_full([case], [_frame([], status=status, anomaly=None,
                                         error="boom")])
    assert score["cases"][0]["outcome"] == "ERR"
    assert score["frame"]["n"] == 0 and score["frame"]["n_failed"] == 1
    # and out of the object level too: an unjudged instance is not a miss
    assert score["objects"]["inst_total"] == 0
    assert score["n_scored"] == 0


def test_lenient_and_strict_instance_matching_differ():
    case = {"id": "a", "has_anomaly": True, "instances": [
        {"type": "object", "bbox": [180, 180, 220, 220]}]}
    blob = B.score_full([case], [_frame([[0, 0, 400, 400]])])
    assert blob["objects"]["inst_detected"] == 1          # lenient: hit
    assert blob["objects"]["inst_detected_strict"] == 0   # strict: boxes nothing


def test_region_precision_counts_unmatched_boxes():
    case = {"id": "a", "has_anomaly": True, "instances": [
        {"type": "object", "bbox": [0, 0, 100, 100]}]}
    score = B.score_full([case], [_frame([[10, 10, 90, 90], [500, 500, 600, 600]])])
    ob = score["objects"]
    assert (ob["kept_total"], ob["fp_regions"], ob["region_precision"]) == (2, 1, 0.5)


def test_score_full_is_partial_run_safe():
    cases = [{"id": str(i), "has_anomaly": True,
              "instances": [{"type": "object", "bbox": [0, 0, 10, 10]}]}
             for i in range(5)]
    score = B.score_full(cases, [_frame([[0, 0, 10, 10]])] * 2)
    assert (score["n_cases"], score["n_scored"]) == (5, 2)
    assert score["objects"]["inst_total"] == 2


def test_score_localize_reports_the_blob_canary():
    case = {"id": "a", "has_anomaly": True, "instances": [
        {"type": "object", "bbox": [180, 180, 220, 220]}]}
    row = {"regions": [{"bbox": [0, 0, 400, 400], "area": 160000}], "seconds": 0.5}
    score = B.score_localize([case], [row])
    loc = score["localization"]
    assert (loc["inst_localized"], loc["inst_localized_strict"]) == (1, 0)
    assert loc["max_box"] == 160000
    assert loc["regions_anomaly"] == 1 and loc["regions_clean"] == 0


# --------------- the published numbers, reproduced by the new aggregation ------

@pytest.mark.skipif(not (ARCHIVE / "results.json").exists(),
                    reason="published CLI results not in the checkout")
def test_aggregation_matches_the_published_report():
    """`benchmark/archive/results.json` holds the per-case rows of the GLM x
    conservative run whose totals are in archive/report.md (merge OFF: 40/45,
    strict 33/45, 29 FP of 86). Feed those same per-case counts through this
    module's aggregation and the totals must come out identical."""
    rows = json.loads((ARCHIVE / "results.json").read_text())["rows"]
    _, doc = B.load()
    by_id = {c["id"]: c for c in doc["cases"]}

    cases, frames = [], []
    for row in rows:
        case = by_id[row["id"]]
        cases.append(case)
        # Rebuild a FrameResult whose kept boxes reproduce the row's counts: one
        # box on each detected instance, duplicates until the row's n_kept is
        # reached (the real run often keeps 2-3 boxes on one object), then
        # `fp_regions` boxes far away. Aggregation is what is under test here,
        # not box matching.
        kept = [inst["bbox"] for inst, hit in
                zip(case["instances"], _hits(row, case)) if hit]
        for i in range(row["n_kept"] - row["fp_regions"] - len(kept)):
            kept.append(kept[i % len(kept)])
        kept += [[9000 + 100 * i, 9000, 9050 + 100 * i, 9050]
                 for i in range(row["fp_regions"])]
        frames.append(_frame(kept, anomaly=row["flagged"]))

    score = B.score_full(cases, frames)
    published_frame = {"TP": 17, "FP": 0, "TN": 12, "FN": 0}
    assert score["frame"]["counts"] == published_frame
    assert score["frame"]["f1"] == 1.0
    ob = score["objects"]
    assert ob["inst_total"] == 45
    assert ob["inst_detected"] == sum(r["instances_detected"] for r in rows) == 40
    assert ob["fp_regions"] == sum(r["fp_regions"] for r in rows) == 29
    assert ob["kept_total"] == 86
    assert round(ob["region_precision"], 3) == 0.663
    assert ob["per_type"]["graffiti"] == {"detected": 6, "total": 6, "recall": 1.0}
    assert ob["per_type"]["object"]["detected"] == 28


def _hits(row, case):
    """Which instances of `case` the published row counted as detected. The row
    only stores the COUNT, so take the first n - per-type totals still line up
    because a row's instances are almost always one type, and the two
    multi-type rows are fully detected."""
    n = row["instances_detected"]
    return [True] * n + [False] * (len(case["instances"]) - n)
