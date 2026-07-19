"""Human review layer over a finished job: per-detection TP/FP verdicts,
missed-object boxes (FN) drawn by the reviewer, and per-frame confirmation.

Stored as review.json next to the job's results.json. Metrics are recomputed
from (results, review) on every read/write — the file stores only judgements,
never derived numbers, so the two can never disagree.

Frame-level correctness follows the supervisor's spreadsheet rule: a frame
with ANY missed ground-truth object scores FN, even when other objects were
detected (partial detection = FN); rating nuance lives at object level.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from arsi_core.schema import ANOMALY_TYPES

VERDICTS = ("tp", "fp")


def review_path(job_dir) -> Path:
    return Path(job_dir) / "review.json"


def load_review(job_dir, job_id: str) -> dict:
    path = review_path(job_dir)
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {"job_id": job_id, "updated": None, "frames": {}}


class ReviewError(ValueError):
    """Invalid review payload (unknown frame, bad verdict, malformed box)."""


def _norm_missed(entry, frame_id):
    bbox = entry.get("bbox")
    if not (isinstance(bbox, list) and len(bbox) == 4
            and all(isinstance(v, (int, float)) for v in bbox)):
        raise ReviewError(f"{frame_id}: missed box needs a numeric [x0,y0,x1,y1]")
    x0, y0, x1, y1 = (round(v) for v in bbox)
    if x1 <= x0 or y1 <= y0:
        raise ReviewError(f"{frame_id}: missed box is empty or inverted")
    label = str(entry.get("label") or "").strip()
    if not label:
        raise ReviewError(f"{frame_id}: missed box needs a label")
    typ = entry.get("type") if entry.get("type") in ANOMALY_TYPES else "unknown"
    return {"bbox": [x0, y0, x1, y1], "label": label, "type": typ}


def validate_review(results: dict, frames_payload: dict) -> dict:
    """Return a normalized {frame_id: {verdicts, missed, done}} dict or raise
    ReviewError. A frame may only be marked done once every detection has a
    verdict — 'reviewed' must mean reviewed."""
    by_id = {f["frame_id"]: f for f in results.get("frames", [])}
    if not isinstance(frames_payload, dict):
        raise ReviewError("'frames' must be an object keyed by frame_id")
    out = {}
    for frame_id, entry in frames_payload.items():
        if frame_id not in by_id:
            raise ReviewError(f"unknown frame_id '{frame_id}'")
        n_dets = len(by_id[frame_id].get("detections") or [])
        verdicts = {}
        for k, v in (entry.get("verdicts") or {}).items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                raise ReviewError(f"{frame_id}: verdict key '{k}' is not an index")
            if not 0 <= idx < n_dets:
                raise ReviewError(f"{frame_id}: detection index {idx} out of range")
            if v not in VERDICTS:
                raise ReviewError(f"{frame_id}: verdict must be one of {VERDICTS}")
            verdicts[str(idx)] = v
        missed = [_norm_missed(m, frame_id) for m in (entry.get("missed") or [])[:100]]
        done = bool(entry.get("done"))
        if done and len(verdicts) < n_dets:
            raise ReviewError(f"{frame_id}: cannot confirm — "
                              f"{n_dets - len(verdicts)} detection(s) unreviewed")
        if verdicts or missed or done:
            out[frame_id] = {"verdicts": verdicts, "missed": missed, "done": done}
    return out


def save_review(job_dir, job_id: str, frames_payload: dict, results: dict) -> dict:
    doc = {"job_id": job_id,
           "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "frames": validate_review(results, frames_payload)}
    path = review_path(job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    return doc


def export_stats(results: dict, review: dict) -> dict:
    """What tools/export_lora_dataset.py would harvest from this review:
    judged detections WITH a bbox (vlm_01/02 verdicts have none) + missed
    boxes, on confirmed frames of a job that has a reference image."""
    has_ref = bool((results.get("config") or {}).get("reference"))
    by_id = {f["frame_id"]: f for f in results.get("frames", [])}
    yes = no = skipped_no_bbox = 0
    for fid, e in review.get("frames", {}).items():
        if not e.get("done"):
            continue
        dets = (by_id.get(fid) or {}).get("detections") or []
        for idx, v in e.get("verdicts", {}).items():
            if int(idx) < len(dets) and dets[int(idx)].get("bbox"):
                if v == "tp":
                    yes += 1
                else:
                    no += 1
            else:
                skipped_no_bbox += 1
        yes += len(e.get("missed", []))
    return {"yes": yes, "no": no, "skipped_no_bbox": skipped_no_bbox,
            "exportable": has_ref, "samples": (yes + no) if has_ref else 0}


def _rate(num, den):
    return round(num / den, 3) if den else None


def frame_correctness(frame: dict, entry: dict):
    """TP/FP/TN/FN for one reviewed frame; None for failed frames."""
    if frame.get("status") != "ok":
        return None
    tp = sum(1 for v in entry["verdicts"].values() if v == "tp")
    fn = len(entry["missed"])
    predicted = bool(frame.get("anomaly"))
    if fn:
        return "FN"                     # supervisor rule: any miss -> FN
    if tp:
        return "TP"
    return "FP" if predicted else "TN"


def compute_metrics(results: dict, review: dict) -> dict:
    frames = results.get("frames", [])
    entries = review.get("frames", {})
    done = {fid: e for fid, e in entries.items() if e.get("done")}

    obj_tp = obj_fp = obj_fn = 0
    per_type = {}
    cm = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    correctness = {}
    for f in frames:
        e = done.get(f["frame_id"])
        if not e:
            continue
        dets = f.get("detections") or []
        for idx, v in e["verdicts"].items():
            typ = dets[int(idx)]["type"] if int(idx) < len(dets) else "unknown"
            t = per_type.setdefault(typ, {"tp": 0, "fp": 0, "fn": 0})
            if v == "tp":
                obj_tp += 1
                t["tp"] += 1
            else:
                obj_fp += 1
                t["fp"] += 1
        for m in e["missed"]:
            obj_fn += 1
            per_type.setdefault(m["type"], {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
        c = frame_correctness(f, e)
        if c:
            cm[c] += 1
            correctness[f["frame_id"]] = c

    for t in per_type.values():
        t["recall"] = _rate(t["tp"], t["tp"] + t["fn"])
    n_scored = sum(cm.values())
    fr = {**cm, "n_scored": n_scored,
          "accuracy": _rate(cm["TP"] + cm["TN"], n_scored),
          "precision": _rate(cm["TP"], cm["TP"] + cm["FP"]),
          "recall": _rate(cm["TP"], cm["TP"] + cm["FN"]),
          "specificity": _rate(cm["TN"], cm["TN"] + cm["FP"]),
          "f1": _rate(2 * cm["TP"], 2 * cm["TP"] + cm["FP"] + cm["FN"])}
    return {
        "progress": {"n_frames": len(frames), "n_done": len(done),
                     "n_failed": sum(1 for f in frames if f.get("status") != "ok")},
        "objects": {"tp": obj_tp, "fp": obj_fp, "fn": obj_fn,
                    "precision": _rate(obj_tp, obj_tp + obj_fp),
                    "recall": _rate(obj_tp, obj_tp + obj_fn),
                    "f1": _rate(2 * obj_tp, 2 * obj_tp + obj_fp + obj_fn)},
        "frames": fr,
        "per_type": per_type,
        "correctness": correctness,
    }
