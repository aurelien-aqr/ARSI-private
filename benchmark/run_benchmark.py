#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - benchmark/run_benchmark.py
#  Reproducible benchmark for the vlm_05 reference-diff anomaly detector.
#
#  For every labelled case in ground_truth.json it runs the SAME pipeline as
#  vlm_05_reference_diff.py (diff -> connected-component regions -> per-region
#  VLM YES/NO in "filter" mode -> drop person/"disappeared" labels -> de-dupe
#  overlapping boxes) and scores it against the ground truth at TWO levels:
#
#   1) FRAME level (binary): a frame is "flagged" if >=1 region survives.
#        TP/FP/TN/FN -> accuracy, precision, recall, specificity, F1.
#
#   2) OBJECT level: each ground-truth instance (a typed box) counts as detected
#        if any kept region overlaps it; kept regions overlapping NO instance are
#        false-positive regions. Gives per-instance recall, per-type recall and a
#        region-level precision - this is what exposes misses/FPs that the coarse
#        frame-level metric hides.
#
#  The VLM is slow on CPU (~a few min per region), so every VLM verdict is cached
#  in benchmark/cache.json keyed by (image, reference, box, model, prompt). The
#  run is RESUMABLE and results.json / report.md are rewritten after every case.
#
#  Run from the repository root:   python benchmark/run_benchmark.py
# =============================================================================

import sys, json, time, hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import vlm_05_reference_diff as m           # the script under test
from PIL import Image, ImageDraw, ImageFont

BENCH_DIR  = Path(__file__).resolve().parent
GT_PATH    = BENCH_DIR / "ground_truth.json"
CACHE_PATH = BENCH_DIR / "cache.json"
RESULTS    = BENCH_DIR / "results.json"
REPORT     = BENCH_DIR / "report.md"
ANNOT_DIR  = BENCH_DIR / "annotated"
ANNOT_DIR.mkdir(exist_ok=True)

TYPES = ["object", "graffiti", "damage", "litter"]


def prompt_fingerprint():
    h = hashlib.sha1((m.MODEL_NAME + "||" + m.PROMPT).encode("utf-8")).hexdigest()
    return h[:12]


def load_json(path, default):
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return default


def resolve(path):
    p = Path(path)
    return str(p if p.is_absolute() else REPO_ROOT / p)


def overlaps(box_a, box_b):
    """Lenient match: IoU>0.1 OR either centre falls inside the other box."""
    if m._iou(box_a, box_b) > 0.1:
        return True
    for inner, outer in ((box_a, box_b), (box_b, box_a)):
        cx = (inner[0] + inner[2]) / 2
        cy = (inner[1] + inner[3]) / 2
        if outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]:
            return True
    return False


def classify_cached(image, reference, region, cache, fp, img_key, ref_key):
    key = f"{ref_key}|{img_key}|{region['bbox']}|{m.MODEL_NAME}|{fp}"
    if key in cache:
        c = cache[key]
        return c["yes"], c["label"]
    is_obj, label = None, ""
    for attempt in range(3):
        try:
            is_obj, label = m.classify_with_vlm(image, reference, region,
                                                m.CROP_MARGIN, m.CROP_CONTEXT)
            break
        except Exception as exc:
            print(f"      retry {attempt+1}/3 ({type(exc).__name__})", flush=True)
            time.sleep(5)
    if is_obj is None:
        return False, "(vlm error)"
    cache[key] = {"yes": bool(is_obj), "label": label}
    with open(CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=1, ensure_ascii=False)
    return is_obj, label


def run_case(case, refmap, cache, fp):
    ref_path = resolve(refmap[case["reference"]])
    img_path = resolve(case["image"])
    reference = Image.open(ref_path).convert("RGB")
    image = Image.open(img_path).convert("RGB")
    if image.size != reference.size:
        image = image.resize(reference.size)

    mask = m.change_mask(ref_path, img_path)
    regions = m.find_regions(mask, m.DOWNSCALE, m.DILATE, m.MIN_AREA, m.MAX_AREA)
    n_regions = len(regions)
    regions = regions[:m.MAX_REGIONS]

    ref_key, img_key = Path(ref_path).name, Path(img_path).name
    kept = []
    for r in regions:
        is_obj, label = classify_cached(image, reference, r, cache, fp, img_key, ref_key)
        # person / "disappeared" / hallucination (unnamed or small-object-on-huge) -> not an anomaly
        if m.is_non_anomaly(label) or m.is_implausible(label, r["area"]):
            is_obj = False
        r["vlm_label"], r["vlm_is_object"] = label, is_obj
        if is_obj:
            kept.append(r)
    kept = m.dedupe_regions(kept)             # collapse multiple boxes on one object

    # ---- object-level scoring against instance ground truth -----------------
    instances = case.get("instances", [])
    inst_hit = [False] * len(instances)
    kept_matched = [False] * len(kept)
    for gi, inst in enumerate(instances):
        for ki, r in enumerate(kept):
            if overlaps(r["bbox"], inst["bbox"]):
                inst_hit[gi] = True
                kept_matched[ki] = True
    detected = sum(inst_hit)
    fp_regions = sum(1 for mtd in kept_matched if not mtd)
    type_detect = {}
    for t in TYPES:
        idx = [i for i, ins in enumerate(instances) if ins["type"] == t]
        if idx:
            type_detect[t] = (sum(inst_hit[i] for i in idx), len(idx))

    # ---- annotate -----------------------------------------------------------
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for inst in instances:                    # ground truth in blue
        x0, y0, x1, y1 = (int(v) for v in inst["bbox"])
        draw.rectangle([x0, y0, x1, y1], outline=(60, 120, 255), width=2)
    for ki, r in enumerate(kept):             # kept: green if matches GT, red if FP
        x0, y0, x1, y1 = (int(v) for v in r["bbox"])
        col = (0, 200, 0) if kept_matched[ki] else (235, 0, 0)
        draw.rectangle([x0, y0, x1, y1], outline=col, width=3)
        draw.text((x0 + 3, max(0, y0 - 20)), r["vlm_label"] or "?", fill=col, font=font)
    image.save(ANNOT_DIR / f"{case['id']}.jpg")

    flagged = len(kept) > 0
    return {
        "id": case["id"], "reference": case["reference"], "image": case["image"],
        "has_anomaly": case["has_anomaly"], "types": case["types"],
        "n_regions": n_regions, "n_classified": len(regions), "n_kept": len(kept),
        "flagged": flagged,
        "outcome": ("TP" if case["has_anomaly"] and flagged else
                    "FN" if case["has_anomaly"] and not flagged else
                    "FP" if (not case["has_anomaly"]) and flagged else "TN"),
        "kept_labels": [r["vlm_label"] for r in kept],
        "instances_total": len(instances), "instances_detected": detected,
        "fp_regions": fp_regions, "type_detect": type_detect,
    }


def metrics(rows):
    c = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for r in rows:
        c[r["outcome"]] += 1
    tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
    n = tp + fp + tn + fn
    def d(a, b):
        return a / b if b else 0.0
    frame = {"counts": c, "n": n, "accuracy": d(tp + tn, n),
             "precision": d(tp, tp + fp), "recall": d(tp, tp + fn),
             "specificity": d(tn, tn + fp),
             "f1": d(2 * tp, 2 * tp + fp + fn)}

    inst_total = sum(r["instances_total"] for r in rows)
    inst_det = sum(r["instances_detected"] for r in rows)
    fp_regions = sum(r["fp_regions"] for r in rows)
    kept_total = sum(r["n_kept"] for r in rows)
    per_type = {}
    for t in TYPES:
        det = tot = 0
        for r in rows:
            if t in r.get("type_detect", {}):
                dd, tt = r["type_detect"][t]
                det += dd
                tot += tt
        if tot:
            per_type[t] = {"detected": det, "total": tot, "recall": d(det, tot)}
    obj = {"inst_total": inst_total, "inst_detected": inst_det,
           "recall": d(inst_det, inst_total),
           "fp_regions": fp_regions, "kept_total": kept_total,
           "region_precision": d(kept_total - fp_regions, kept_total),
           "per_type": per_type}
    return frame, obj


def write_report(rows, frame, obj, elapsed, done, total):
    L = []
    L.append("# vlm_05 reference-diff — anomaly detection benchmark\n")
    status = "COMPLETE" if done == total else f"PARTIAL ({done}/{total} cases)"
    L.append(f"**Status:** {status}  ")
    L.append(f"**Model:** `{m.MODEL_NAME}` (Ollama)  ")
    L.append(f"**Decision rule:** frame flagged if the VLM keeps ≥1 region "
             f"(`filter` mode) after dropping person/\"disappeared\" labels and "
             f"de-duplicating overlapping boxes.  ")
    L.append(f"**Diff / region params:** DIFF_THRESHOLD={m.DIFF_THRESHOLD}, "
             f"BLUR_RADIUS={m.BLUR_RADIUS}, MIN_AREA={m.MIN_AREA}, "
             f"MAX_AREA={m.MAX_AREA}, MAX_REGIONS={m.MAX_REGIONS}.  ")
    L.append(f"**Wall-clock:** {elapsed/60:.1f} min (CPU-only Ollama).\n")

    L.append("## Prompt\n")
    L.append("```\n" + m.PROMPT + "\n```\n")

    c = frame["counts"]
    L.append("## 1) Frame-level (binary: is the frame anomalous?)\n")
    L.append(f"- Cases: **{frame['n']}**  (TP={c['TP']}, FP={c['FP']}, "
             f"TN={c['TN']}, FN={c['FN']})")
    L.append(f"- **Accuracy** {frame['accuracy']:.3f} · **Precision** "
             f"{frame['precision']:.3f} · **Recall** {frame['recall']:.3f} · "
             f"**Specificity** {frame['specificity']:.3f} · **F1** {frame['f1']:.3f}\n")
    L.append("| | predicted anomaly | predicted clean |")
    L.append("|---|---|---|")
    L.append(f"| **actual anomaly** | TP = {c['TP']} | FN = {c['FN']} |")
    L.append(f"| **actual clean**   | FP = {c['FP']} | TN = {c['TN']} |\n")

    L.append("## 2) Object-level (did we box each real anomaly?)\n")
    L.append(f"- Instances detected: **{obj['inst_detected']} / {obj['inst_total']}** "
             f"→ **object recall {obj['recall']:.3f}**")
    L.append(f"- False-positive regions (kept boxes matching no real anomaly): "
             f"**{obj['fp_regions']}** of {obj['kept_total']} kept "
             f"→ region precision {obj['region_precision']:.3f}\n")
    if obj["per_type"]:
        L.append("| type | instances detected | recall |")
        L.append("|---|---|---|")
        for t, dd in obj["per_type"].items():
            L.append(f"| {t} | {dd['detected']} / {dd['total']} | {dd['recall']:.2f} |")
        L.append("")

    L.append("## Per-case results\n")
    L.append("| id | truth | frame | instances hit | FP boxes | VLM kept-labels |")
    L.append("|---|---|---|---|---|---|")
    order = {"TP": 0, "FN": 1, "FP": 2, "TN": 3}
    for r in sorted(rows, key=lambda x: (order[x["outcome"]], x["id"])):
        truth = "anomaly" if r["has_anomaly"] else "clean"
        inst = (f"{r['instances_detected']}/{r['instances_total']}"
                if r["instances_total"] else "—")
        labels = ", ".join(x for x in r["kept_labels"] if x) or "—"
        L.append(f"| {r['id']} | {truth} | **{r['outcome']}** | {inst} | "
                 f"{r['fp_regions']} | {labels} |")
    L.append("")
    L.append("Annotated images: `benchmark/annotated/<id>.jpg` "
             "(blue = ground-truth boxes, green = correct detections, red = "
             "false-positive boxes). Raw results: `benchmark/results.json`.\n")
    REPORT.write_text("\n".join(L), encoding="utf-8")


def main():
    gt = load_json(GT_PATH, None)
    refmap, cases = gt["references"], gt["cases"]
    cache = load_json(CACHE_PATH, {})
    fp = prompt_fingerprint()
    if m.USE_VLM:
        m.check_model(m.MODEL_NAME)

    print("Ordering cases by region count...", flush=True)
    costed = []
    for case in cases:
        ref_path, img_path = resolve(refmap[case["reference"]]), resolve(case["image"])
        n = len(m.find_regions(m.change_mask(ref_path, img_path),
                               m.DOWNSCALE, m.DILATE, m.MIN_AREA, m.MAX_AREA))
        costed.append((min(n, m.MAX_REGIONS), case))
    costed.sort(key=lambda x: x[0])

    rows, t0, total = [], time.time(), len(costed)
    for i, (ncost, case) in enumerate(costed, 1):
        print(f"[{i}/{total}] {case['id']}  (~{ncost} regions)", flush=True)
        row = run_case(case, refmap, cache, fp)
        rows.append(row)
        print(f"      -> {row['outcome']}  inst {row['instances_detected']}/"
              f"{row['instances_total']}  FP {row['fp_regions']}  "
              f"labels={[x for x in row['kept_labels'] if x]}", flush=True)
        with open(RESULTS, "w", encoding="utf-8") as fh:
            json.dump({"generated": time.time(), "prompt": m.PROMPT,
                       "model": m.MODEL_NAME, "rows": rows}, fh, indent=2, ensure_ascii=False)
        fr, ob = metrics(rows)
        write_report(rows, fr, ob, time.time() - t0, i, total)

    print(f"\nDONE {total} cases in {(time.time()-t0)/60:.1f} min. Report: {REPORT}",
          flush=True)


if __name__ == "__main__":
    main()
