"""The benchmark ground truth, and the score computed against it.

There is ONE benchmark: `benchmark/datasets/ground_truth.json`, every labelled
frame, whatever it shows. A case pairs an inspection image with the clean reference
it is diffed against, a frame-level `has_anomaly` label and zero or more typed
instance boxes in the pixel space of THAT case's reference. It is read by the
Studio, by `benchmark/run_benchmark.py` and by `benchmark/eval_localization.py`.

Nothing here describes what the frames contain or how many there are: a viewpoint
is a `references` key and a provenance is a `source`, both of which the filters,
the run subsets and the per-source tables group on, and every count is derived
from the file. Adding a camera, a tram or a rendered scene is adding cases. The
directory is per-dataset only because a run records which one it scored and an
imported public protocol would be a second file (docs/PUBLIC_DATASETS.md).

Scoring rules are the ones the published reports were produced with, and they are
NOT reimplemented from a description: `iou` / `overlaps` are copied from
`vlm_05_reference_diff` and `tests/test_benchmarks.py` locks them to that module,
so the two can never drift. Two levels, as in `benchmark/README.md`:

  frame level   a case is flagged if >= 1 region survives the judge -> TP/FP/TN/FN
  object level  an instance counts as detected if any kept box overlaps it
                (lenient: IoU > 0.1 OR either centre inside the other); a stricter
                IoU >= 0.3 recall is reported alongside because the lenient rule
                credits a frame-sized blob with hitting everything

Editing is a first-class operation here (the 3333 labels were drafted by a model
and need human correction), so `save()` writes through a stable formatter: one
line per case, one line per instance, keys in a fixed order. A corrected box then
shows up as a one-line git diff instead of reshuffling the whole file.
"""
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from . import REPO_ROOT
from .schema import ANOMALY_TYPES

BENCH_DIR = REPO_ROOT / "benchmark"
DATASETS_DIR = BENCH_DIR / "datasets"
BACKUPS_DIR = DATASETS_DIR / ".backups"
RUNS_DIR = BENCH_DIR / "runs"

#: The one benchmark. Everything defaults to it; a caller may still name another
#: dataset id or hand over a path (an archived copy, an imported protocol).
DEFAULT = "ground_truth"

#: Instance types that get their own recall row. `unknown` is accepted on a case
#: (schema.ANOMALY_TYPES) but is not a benchmark category. Same set as
#: run_benchmark.TYPES, so the per-type tables match the published reports.
TYPES = ("object", "graffiti", "damage", "litter")
STRICT_IOU = 0.3

CASE_KEYS = ("id", "image", "reference", "source", "has_anomaly", "types",
             "note", "instances")
INSTANCE_KEYS = ("type", "label", "bbox", "note")
SOURCES = ("real", "gpt", "variant", "self")


class DatasetError(ValueError):
    """Invalid dataset document or case payload (the API turns this into a 400)."""


# --------------------------------------------------------------- matching rules

def iou(a, b) -> float:
    """Box IoU. Copied from vlm_05_reference_diff._iou (locked by a test)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def overlaps(box_a, box_b) -> bool:
    """Lenient match: IoU > 0.1 OR either centre falls inside the other box.
    Copied from vlm_05_reference_diff._boxes_overlap (locked by a test)."""
    if iou(box_a, box_b) > 0.1:
        return True
    for inner, outer in ((box_a, box_b), (box_b, box_a)):
        cx = (inner[0] + inner[2]) / 2
        cy = (inner[1] + inner[3]) / 2
        if outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]:
            return True
    return False


# --------------------------------------------------------------- load / save

def dataset_path(ds_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", ds_id or ""):
        raise DatasetError(f"invalid dataset id '{ds_id}'")
    return DATASETS_DIR / f"{ds_id}.json"


def list_ids():
    return sorted(p.stem for p in DATASETS_DIR.glob("*.json")) \
        if DATASETS_DIR.exists() else []


def resolve(ref) -> Path:
    """Accept a dataset id, a repo-relative path or an absolute path."""
    ref = str(ref or DEFAULT)
    direct = Path(ref) if Path(ref).is_absolute() else REPO_ROOT / ref
    if ref.endswith(".json") and direct.is_file():
        return direct
    try:
        path = dataset_path(ref)
    except DatasetError:
        raise DatasetError(f"no benchmark dataset at '{ref}'")
    if path.is_file():
        return path
    raise DatasetError(f"no benchmark dataset '{ref}' (have: "
                       f"{', '.join(list_ids()) or 'none'})")


def load(ref=DEFAULT):
    """(ds_id, doc) for a dataset id or path. Raises DatasetError if unreadable."""
    path = resolve(ref)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("cases"), list):
        raise DatasetError(f"{path.name} is not a dataset document "
                           f"(needs 'references' + 'cases')")
    doc.setdefault("references", {})
    return path.stem, doc


def _one_line(obj, keys) -> str:
    parts = [f"{json.dumps(k, ensure_ascii=False)}: "
             f"{json.dumps(obj[k], ensure_ascii=False)}"
             for k in keys if k in obj]
    extra = [k for k in obj if k not in keys]
    parts += [f"{json.dumps(k, ensure_ascii=False)}: "
              f"{json.dumps(obj[k], ensure_ascii=False)}" for k in sorted(extra)]
    return "{" + ", ".join(parts) + "}"


def dumps(doc: dict) -> str:
    """Serialize a dataset the way these files are written by hand: the header
    keys one per line, then one line per case with its instances listed one per
    line underneath. Stable, so editing one box is a one-line diff.

    Unknown header keys are preserved after the known ones - dropping them would
    make the first save from the Studio silently delete provenance a dataset
    carries that this module happens not to know about."""
    head = [k for k in ("_about", "name") if k in doc]
    head += [k for k in sorted(doc) if k not in ("_about", "name", "references", "cases")]
    lines = ["{"]
    for k in head:
        lines.append(f"  {json.dumps(k, ensure_ascii=False)}: "
                     f"{json.dumps(doc[k], ensure_ascii=False)},")
    lines.append('  "references": {')
    refs = list(doc.get("references", {}).items())
    for i, (k, v) in enumerate(refs):
        comma = "," if i < len(refs) - 1 else ""
        lines.append(f"    {json.dumps(k, ensure_ascii=False)}: "
                     f"{json.dumps(v, ensure_ascii=False)}{comma}")
    lines.append("  },")
    lines.append('  "cases": [')
    cases = doc.get("cases", [])
    for ci, case in enumerate(cases):
        tail = "," if ci < len(cases) - 1 else ""
        keys = [k for k in CASE_KEYS if k in case and k != "instances"]
        keys += [k for k in sorted(case) if k not in CASE_KEYS]
        head_txt = ", ".join(f"{json.dumps(k, ensure_ascii=False)}: "
                             f"{json.dumps(case[k], ensure_ascii=False)}"
                             for k in keys)
        instances = case.get("instances") or []
        if not instances:
            lines.append(f"    {{{head_txt}, \"instances\": []}}{tail}")
            continue
        lines.append(f"    {{{head_txt}, \"instances\": [")
        for ii, inst in enumerate(instances):
            end = "]}" + tail if ii == len(instances) - 1 else ","
            lines.append(f"      {_one_line(inst, INSTANCE_KEYS)}{end}")
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines)


def digest(doc: dict) -> str:
    """Fingerprint of the LABELS (references + cases), so a run can tell that the
    ground truth it was scored against has been edited since."""
    payload = json.dumps({"references": doc.get("references", {}),
                          "cases": doc.get("cases", [])},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def save(ds_id: str, doc: dict) -> Path:
    """Validate, back up the current file, then write atomically. Raises
    DatasetError before touching anything if the document is unusable - a
    half-valid ground truth is worse than a rejected edit."""
    errors, _ = validate(doc)
    if errors:
        raise DatasetError("; ".join(errors[:5]))
    path = dataset_path(ds_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
        # This backup is what tools/build_3333_benchmark.py tells the reader to
        # rely on when a rebuild replaces human corrections, so losing one is not
        # acceptable. A timestamp alone does not guarantee that: two saves land in
        # the same millisecond easily enough (measured). Resolve the collision
        # instead of narrowing the window.
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        dest = BACKUPS_DIR / f"{ds_id}-{stamp}.json"
        n = 0
        while dest.exists():
            n += 1
            dest = BACKUPS_DIR / f"{ds_id}-{stamp}-{n}.json"
        dest.write_bytes(path.read_bytes())
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(dumps(doc) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


# --------------------------------------------------------------- validation

def _check_bbox(bbox, where: str, errors: list):
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in bbox)):
        errors.append(f"{where}: bbox must be [x0, y0, x1, y1] numbers")
        return False
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        errors.append(f"{where}: bbox is empty or inverted {list(bbox)}")
        return False
    return True


def validate(doc: dict):
    """(errors, warnings). Errors block a save; warnings are shown in the UI.

    A `has_anomaly` case with no instance box is a WARNING, not an error: an
    anomaly nobody could box yet is a legitimate frame-level label, and refusing
    it would push the labeller into inventing coordinates."""
    errors, warnings = [], []
    refs = doc.get("references") or {}
    if not isinstance(refs, dict) or not refs:
        errors.append("'references' must be a non-empty {key: path} map")
        refs = refs if isinstance(refs, dict) else {}
    for key, rel in refs.items():
        if not (REPO_ROOT / str(rel)).is_file():
            errors.append(f"reference '{key}': image not found ({rel})")
    seen = set()
    for i, case in enumerate(doc.get("cases") or []):
        # the isinstance check comes FIRST: a stray string or null in `cases`
        # (hand-edited file) used to raise AttributeError here, which the API
        # cannot turn into a validation error - it 500s the whole screen
        if not isinstance(case, dict):
            errors.append(f"case #{i}: must be an object, got {type(case).__name__}")
            continue
        cid = case.get("id") or f"#{i}"
        if not case.get("id"):
            errors.append(f"case #{i}: missing 'id'")
            continue
        if case["id"] in seen:
            errors.append(f"case '{cid}': duplicate id")
        seen.add(case["id"])
        if not case.get("image"):
            errors.append(f"case '{cid}': missing 'image'")
        elif not (REPO_ROOT / str(case["image"])).is_file():
            errors.append(f"case '{cid}': image not found ({case['image']})")
        if case.get("reference") not in refs:
            errors.append(f"case '{cid}': reference '{case.get('reference')}' "
                          f"is not a key of 'references'")
        if not isinstance(case.get("has_anomaly"), bool):
            errors.append(f"case '{cid}': 'has_anomaly' must be true or false")
        instances = case.get("instances")
        if instances is not None and not isinstance(instances, list):
            errors.append(f"case '{cid}': 'instances' must be a list")
            instances = []
        for j, inst in enumerate(instances or []):
            where = f"case '{cid}' instance #{j}"
            if not isinstance(inst, dict):
                errors.append(f"{where}: must be an object")
                continue
            if inst.get("type") not in ANOMALY_TYPES:
                errors.append(f"{where}: type must be one of {ANOMALY_TYPES}")
            _check_bbox(inst.get("bbox"), where, errors)
        if case.get("has_anomaly") and not (instances or []):
            warnings.append(f"case '{cid}': labelled anomalous but has no "
                            f"instance box, so it scores at frame level only")
        if not case.get("has_anomaly") and (instances or []):
            errors.append(f"case '{cid}': labelled clean but carries "
                          f"{len(instances)} instance box(es)")
        if case.get("source") and case["source"] not in SOURCES:
            warnings.append(f"case '{cid}': unusual source '{case['source']}'")
    return errors, warnings


def normalize_case(payload: dict, refs: dict) -> dict:
    """Turn an API payload into a stored case. Unknown keys are dropped rather
    than persisted, so the UI cannot grow the schema by accident."""
    if not isinstance(payload, dict):
        raise DatasetError("case payload must be an object")
    case = {
        "id": str(payload.get("id") or "").strip(),
        "image": str(payload.get("image") or "").strip(),
        "reference": str(payload.get("reference") or "").strip(),
        "source": str(payload.get("source") or "real").strip(),
        "has_anomaly": bool(payload.get("has_anomaly")),
        "types": [],
        "note": str(payload.get("note") or "").strip(),
        "instances": [],
    }
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", case["id"]):
        raise DatasetError("case id must be 1-80 chars of [A-Za-z0-9_.-]")
    for inst in payload.get("instances") or []:
        if not isinstance(inst, dict):
            raise DatasetError("each instance must be an object")
        bbox = inst.get("bbox")
        errors = []
        _check_bbox(bbox, f"case '{case['id']}'", errors)
        if errors:
            raise DatasetError(errors[0])
        typ = inst.get("type") if inst.get("type") in ANOMALY_TYPES else "unknown"
        out = {"type": typ, "bbox": [round(v) for v in bbox]}
        if str(inst.get("label") or "").strip():
            out["label"] = str(inst["label"]).strip()
        if str(inst.get("note") or "").strip():
            out["note"] = str(inst["note"]).strip()
        case["instances"].append(out)
    # `types` is a derived summary of the instance types - it was maintained by
    # hand and drifted; deriving it removes a whole class of silent mismatch.
    case["types"] = sorted({i["type"] for i in case["instances"]})
    if case["reference"] not in refs:
        raise DatasetError(f"reference '{case['reference']}' is not a key of "
                           f"'references' ({', '.join(refs) or 'none'})")
    return case


# --------------------------------------------------------------- summaries

def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)          # a dataset dir outside the repo (tests, exports)


def summary(ds_id: str, doc: dict) -> dict:
    cases = doc.get("cases") or []
    errors, warnings = validate(doc)
    cams = sorted({c.get("reference", "") for c in cases if c.get("reference")})
    return {
        "id": ds_id,
        "name": doc.get("name") or ds_id,
        "path": _display_path(resolve(ds_id)),
        "about": doc.get("_about", ""),
        "n_cases": len(cases),
        "n_anomalous": sum(1 for c in cases if c.get("has_anomaly")),
        "n_clean": sum(1 for c in cases if not c.get("has_anomaly")),
        "n_instances": sum(len(c.get("instances") or []) for c in cases),
        "references": cams,
        "digest": digest(doc),
        "errors": errors,
        "warnings": warnings,
    }


def _d(a, b):
    return a / b if b else 0.0


# --------------------------------------------------------------- full scoring

def _match(instances, kept):
    """(instance hits, strict hits, which kept boxes matched something)."""
    hit = [False] * len(instances)
    strict = [False] * len(instances)
    matched = [False] * len(kept)
    for gi, inst in enumerate(instances):
        for ki, box in enumerate(kept):
            if overlaps(box, inst["bbox"]):
                hit[gi] = True
                matched[ki] = True
            if iou(box, inst["bbox"]) >= STRICT_IOU:
                strict[gi] = True
    return hit, strict, matched


def score_case(case: dict, frame: dict) -> dict:
    """One case scored against one FrameResult dict (docs/SPEC.md schema).

    A frame that did not complete gets outcome "ERR" and is left out of every
    aggregate: counting a crashed frame as a clean prediction would quietly turn
    infrastructure failures into recall. That includes `cancelled` - the runner
    builds that status for a frame stopped mid-way through its regions,
    explicitly so it is not read as "clean frame, fully looked at"."""
    instances = case.get("instances") or []
    status = frame.get("status", "ok")
    dets = [d for d in (frame.get("detections") or []) if d.get("bbox")]
    kept_boxes = [d["bbox"] for d in dets]
    hit, strict, matched = _match(instances, kept_boxes)
    flagged = bool(frame.get("anomaly")) if frame.get("anomaly") is not None \
        else bool(frame.get("detections"))
    if status != "ok":
        outcome = "ERR"
    elif case.get("has_anomaly"):
        outcome = "TP" if flagged else "FN"
    else:
        outcome = "FP" if flagged else "TN"
    cands = frame.get("candidates") or []
    loc = frame.get("localization") or {}
    type_detect = {}
    for t in TYPES:
        idx = [i for i, ins in enumerate(instances) if ins.get("type") == t]
        if idx:
            type_detect[t] = [sum(hit[i] for i in idx), len(idx)]
    return {
        "id": case["id"], "image": case.get("image"),
        "reference": case.get("reference"), "source": case.get("source", "?"),
        "has_anomaly": bool(case.get("has_anomaly")),
        "types": case.get("types", []), "note": case.get("note", ""),
        "status": status, "outcome": outcome, "flagged": flagged,
        "n_regions": loc.get("proposed", len(cands)),
        "n_classified": len(cands),
        "n_kept": len(dets),
        "localization": loc,
        "kept": [{"bbox": d["bbox"], "label": d.get("label", ""),
                  "type": d.get("type", "unknown"), "channel": d.get("channel"),
                  "matched": matched[i]} for i, d in enumerate(dets)],
        "kept_labels": [d.get("label", "") for d in dets],
        "candidates": [c for c in cands if c.get("outcome") != "kept"],
        "instances": [{**inst, "hit": hit[i], "strict": strict[i]}
                      for i, inst in enumerate(instances)],
        "instances_total": len(instances),
        "instances_detected": sum(hit),
        "instances_detected_strict": sum(strict),
        "fp_regions": sum(1 for m in matched if not m),
        "type_detect": type_detect,
        "fresh_calls": sum(1 for c in cands if c.get("cached") is False),
        "cached_calls": sum(1 for c in cands if c.get("cached") is True),
        "seconds": frame.get("seconds", 0.0),
        "error": frame.get("error"),
    }


def score_full(cases: list, frames: list) -> dict:
    """Score a (possibly partial) run: `frames[i]` is the FrameResult of
    `cases[i]`. Metric definitions are run_benchmark.metrics'."""
    all_rows = [score_case(c, f) for c, f in zip(cases, frames) if f]
    counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    # An ERR row measured nothing complete, so it is excluded from the object
    # level too: its instances are not misses, they were never judged. The count
    # is reported instead of the row being dropped silently.
    rows = [r for r in all_rows if r["outcome"] != "ERR"]
    n_err = len(all_rows) - len(rows)
    for r in rows:
        counts[r["outcome"]] += 1
    tp, fp, tn, fn = counts["TP"], counts["FP"], counts["TN"], counts["FN"]
    n = tp + fp + tn + fn
    frame = {"counts": counts, "n": n, "n_failed": n_err,
             "accuracy": _d(tp + tn, n), "precision": _d(tp, tp + fp),
             "recall": _d(tp, tp + fn), "specificity": _d(tn, tn + fp),
             "f1": _d(2 * tp, 2 * tp + fp + fn)}

    inst_total = sum(r["instances_total"] for r in rows)
    inst_det = sum(r["instances_detected"] for r in rows)
    inst_strict = sum(r["instances_detected_strict"] for r in rows)
    fp_regions = sum(r["fp_regions"] for r in rows)
    kept_total = sum(r["n_kept"] for r in rows)
    per_type = {}
    for t in TYPES:
        det = tot = 0
        for r in rows:
            if t in r["type_detect"]:
                det += r["type_detect"][t][0]
                tot += r["type_detect"][t][1]
        if tot:
            per_type[t] = {"detected": det, "total": tot, "recall": _d(det, tot)}
    per_source = {}
    for r in rows:
        s = per_source.setdefault(r["source"], {"cases": 0, "inst_total": 0,
                                                "inst_detected": 0, "fp_regions": 0})
        s["cases"] += 1
        s["inst_total"] += r["instances_total"]
        s["inst_detected"] += r["instances_detected"]
        s["fp_regions"] += r["fp_regions"]
    objects = {"inst_total": inst_total, "inst_detected": inst_det,
               "recall": _d(inst_det, inst_total),
               "inst_detected_strict": inst_strict,
               "recall_strict": _d(inst_strict, inst_total),
               "fp_regions": fp_regions, "kept_total": kept_total,
               "region_precision": _d(kept_total - fp_regions, kept_total),
               "per_type": per_type, "per_source": per_source}
    return {"mode": "full", "n_cases": len(cases), "n_scored": len(rows),
            "frame": frame, "objects": objects,
            "regions_judged": sum(r["n_classified"] for r in all_rows),
            "fresh_calls": sum(r["fresh_calls"] for r in all_rows),
            "cached_calls": sum(r["cached_calls"] for r in all_rows),
            # every row, ERR included: the per-case list must still show what
            # happened to a frame that failed
            "cases": all_rows}


# --------------------------------------------------------------- localizer-only

def score_localize(cases: list, rows: list) -> dict:
    """Localization-only scoring (no judge): an instance is localized if any
    PROPOSED region overlaps it. Upper bound on end-to-end recall, and the
    number that threshold tuning moves. Same shape as eval_localization.py's
    printout, per case and in total."""
    out = []
    for case, row in zip(cases, rows):
        if not row:
            continue
        instances = case.get("instances") or []
        boxes = [r["bbox"] for r in (row.get("regions") or [])]
        hit, strict, _ = _match(instances, boxes)
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        out.append({
            "id": case["id"], "image": case.get("image"),
            "reference": case.get("reference"), "source": case.get("source", "?"),
            "has_anomaly": bool(case.get("has_anomaly")),
            "note": case.get("note", ""),
            "n_regions": len(boxes),
            "regions": row.get("regions") or [],
            "localization": row.get("localization") or {},
            "instances": [{**inst, "hit": hit[i], "strict": strict[i]}
                          for i, inst in enumerate(instances)],
            "instances_total": len(instances),
            "instances_detected": sum(hit),
            "instances_detected_strict": sum(strict),
            "max_box": max(areas) if areas else 0,
            "seconds": row.get("seconds", 0.0),
            "error": row.get("error"),
        })
    inst_total = sum(r["instances_total"] for r in out)
    per_type = {}
    for r in out:                     # rows carry their own instances; zipping
        for inst in r["instances"]:   # them back onto `cases` misaligns on a
                                      # partial run (out skips unmeasured cases)
            t = inst.get("type", "unknown")
            d = per_type.setdefault(t, {"detected": 0, "total": 0})
            d["detected"] += int(inst["hit"])
            d["total"] += 1
    for d in per_type.values():
        d["recall"] = _d(d["detected"], d["total"])
    det = sum(r["instances_detected"] for r in out)
    strict = sum(r["instances_detected_strict"] for r in out)
    return {
        "mode": "localize", "n_cases": len(cases), "n_scored": len(out),
        "localization": {
            "inst_total": inst_total, "inst_localized": det,
            "recall": _d(det, inst_total),
            "inst_localized_strict": strict, "recall_strict": _d(strict, inst_total),
            "regions_anomaly": sum(r["n_regions"] for r in out if r["has_anomaly"]),
            "regions_clean": sum(r["n_regions"] for r in out if not r["has_anomaly"]),
            "frames_clean": sum(1 for r in out if not r["has_anomaly"]),
            "regions_total": sum(r["n_regions"] for r in out),
            "max_box": max((r["max_box"] for r in out), default=0),
            "per_type": per_type,
        },
        "cases": out,
    }


# --------------------------------------------------------------- report

def headline(score: dict) -> dict:
    """The few numbers a run is compared on, for the runs table."""
    if score.get("mode") == "localize":
        loc = score["localization"]
        return {"mode": "localize", "recall": loc["recall"],
                "recall_strict": loc["recall_strict"],
                "localized": f"{loc['inst_localized']}/{loc['inst_total']}",
                "strict": f"{loc['inst_localized_strict']}/{loc['inst_total']}",
                "regions": loc["regions_total"]}
    fr, ob = score["frame"], score["objects"]
    return {"mode": "full", "f1": fr["f1"], "accuracy": fr["accuracy"],
            "recall": fr["recall"], "specificity": fr["specificity"],
            "counts": fr["counts"],
            "object_recall": ob["recall"], "object_recall_strict": ob["recall_strict"],
            "detected": f"{ob['inst_detected']}/{ob['inst_total']}",
            "strict": f"{ob['inst_detected_strict']}/{ob['inst_total']}",
            "region_precision": ob["region_precision"],
            "fp_regions": ob["fp_regions"], "kept_total": ob["kept_total"]}


def report_md(run: dict, score: dict) -> str:
    """Markdown report - same sections and wording as run_benchmark.write_report,
    with the run's configuration in the header (that script scores whatever
    vlm_05 is configured with; here the configuration is a choice, so it has to
    be written down next to the numbers)."""
    cfg = run.get("config") or {}
    L = [f"# Benchmark run - {run.get('dataset', '?')} · {cfg.get('mode', 'full')}\n"]
    L.append(f"**Run:** `{run.get('run_id', '')}`  ")
    L.append(f"**Status:** {run.get('status', '?')} "
             f"({score.get('n_scored', 0)}/{score.get('n_cases', 0)} cases)  ")
    L.append(f"**Dataset:** `{run.get('dataset', '')}` "
             f"(digest `{run.get('digest', '')}`)  ")
    if cfg.get("mode") == "localize":
        L.append(f"**Localizer:** `{cfg.get('localizer', '')}` - no VLM call.  ")
    else:
        L.append(f"**Pipeline:** `{cfg.get('script', '')}` · localizer "
                 f"`{cfg.get('localizer') or '-'}` · judge `{cfg.get('model', '')}` "
                 f"· prompt `{cfg.get('prompt_name', '')}`  ")
    if run.get("params"):
        L.append(f"**Params:** `{json.dumps(run['params'], sort_keys=True)}`  ")
    L.append(f"**Wall-clock:** {run.get('wall_seconds', 0) / 60:.1f} min.\n")

    if score.get("mode") == "localize":
        loc = score["localization"]
        L.append("## Localization only (upper bound on end-to-end recall)\n")
        L.append(f"- Instances localized: **{loc['inst_localized']} / "
                 f"{loc['inst_total']}** → recall {loc['recall']:.3f} "
                 f"(strict IoU≥{STRICT_IOU}: {loc['inst_localized_strict']} / "
                 f"{loc['inst_total']} = {loc['recall_strict']:.3f})")
        L.append(f"- Regions proposed: **{loc['regions_total']}** "
                 f"({loc['regions_anomaly']} on anomaly frames, "
                 f"{loc['regions_clean']} on {loc['frames_clean']} clean frames)")
        L.append(f"- Biggest box: {loc['max_box']:,} px "
                 f"(the blob canary - a frame-sized box hits every instance "
                 f"leniently while boxing nothing)\n")
        if loc["per_type"]:
            L.append("| type | localized | recall |")
            L.append("|---|---|---|")
            for t, d in sorted(loc["per_type"].items()):
                L.append(f"| {t} | {d['detected']} / {d['total']} | {d['recall']:.2f} |")
            L.append("")
        L.append("## Per-case\n")
        L.append("| id | truth | regions | instances localized | biggest box |")
        L.append("|---|---|---|---|---|")
        for r in score["cases"]:
            truth = "anomaly" if r["has_anomaly"] else "clean"
            inst = (f"{r['instances_detected']}/{r['instances_total']}"
                    if r["instances_total"] else "-")
            L.append(f"| {r['id']} | {truth} | {r['n_regions']} | {inst} | "
                     f"{r['max_box']:,} |")
        L.append("")
        return "\n".join(L)

    if cfg.get("prompt"):
        L.append("## Prompt\n")
        L.append("```\n" + cfg["prompt"] + "\n```\n")
    fr, ob = score["frame"], score["objects"]
    c = fr["counts"]
    L.append("## 1) Frame-level (binary: is the frame anomalous?)\n")
    L.append(f"- Cases: **{fr['n']}**  (TP={c['TP']}, FP={c['FP']}, "
             f"TN={c['TN']}, FN={c['FN']})"
             + (f"  · {fr['n_failed']} incomplete frame(s) (failed or cancelled) "
                f"excluded from every aggregate" if fr["n_failed"] else ""))
    L.append(f"- **Accuracy** {fr['accuracy']:.3f} · **Precision** "
             f"{fr['precision']:.3f} · **Recall** {fr['recall']:.3f} · "
             f"**Specificity** {fr['specificity']:.3f} · **F1** {fr['f1']:.3f}\n")
    L.append("| | predicted anomaly | predicted clean |")
    L.append("|---|---|---|")
    L.append(f"| **actual anomaly** | TP = {c['TP']} | FN = {c['FN']} |")
    L.append(f"| **actual clean**   | FP = {c['FP']} | TN = {c['TN']} |\n")

    L.append("## 2) Object-level (did we box each real anomaly?)\n")
    L.append(f"- Instances detected: **{ob['inst_detected']} / {ob['inst_total']}** "
             f"→ **object recall {ob['recall']:.3f}** "
             f"(strict IoU≥{STRICT_IOU}: {ob['inst_detected_strict']} / "
             f"{ob['inst_total']} = {ob['recall_strict']:.3f})")
    L.append(f"- False-positive regions (kept boxes matching no real anomaly): "
             f"**{ob['fp_regions']}** of {ob['kept_total']} kept "
             f"→ region precision {ob['region_precision']:.3f}")
    fresh, cached = score.get("fresh_calls", 0), score.get("cached_calls", 0)
    L.append(f"- Regions sent to the judge: {score.get('regions_judged', 0)} - "
             + (f"**{fresh} fresh VLM call(s)**, {cached} served from cache.\n"
                if fresh else f"all {cached} verdicts served from cache "
                              f"(0 new calls).\n"))
    if ob["per_type"]:
        L.append("| type | instances detected | recall |")
        L.append("|---|---|---|")
        for t, d in ob["per_type"].items():
            L.append(f"| {t} | {d['detected']} / {d['total']} | {d['recall']:.2f} |")
        L.append("")
    if ob["per_source"]:
        L.append("| source | cases | instances detected | FP regions |")
        L.append("|---|---|---|---|")
        for s, d in sorted(ob["per_source"].items()):
            L.append(f"| {s} | {d['cases']} | {d['inst_detected']} / "
                     f"{d['inst_total']} | {d['fp_regions']} |")
        L.append("")

    L.append("## Per-case results\n")
    L.append("| id | truth | frame | instances hit | FP boxes | kept labels |")
    L.append("|---|---|---|---|---|---|")
    order = {"TP": 0, "FN": 1, "FP": 2, "TN": 3, "ERR": 4}
    for r in sorted(score["cases"], key=lambda x: (order[x["outcome"]], x["id"])):
        truth = "anomaly" if r["has_anomaly"] else "clean"
        inst = (f"{r['instances_detected']}/{r['instances_total']}"
                if r["instances_total"] else "-")
        labels = ", ".join(x for x in r["kept_labels"] if x) or "-"
        L.append(f"| {r['id']} | {truth} | **{r['outcome']}** | {inst} | "
                 f"{r['fp_regions']} | {labels} |")
    L.append("")
    return "\n".join(L)
