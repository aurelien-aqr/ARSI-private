#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - benchmark/eval_localization.py
#  Localization-ONLY evaluation of the vlm_05 change detector - no VLM calls.
#
#  The full benchmark (run_benchmark.py) costs ~15 s of CPU VLM time per region.
#  But the change-detection stage can be scored on its own against the instance
#  boxes in ground_truth.json in SECONDS: an anomaly instance is "localized" if
#  ANY candidate region overlaps its GT box (same lenient rule as the benchmark;
#  the VLM can only keep what the localizer produced, so this is an upper bound
#  on end-to-end recall). This makes threshold / diff-variant tuning a measured
#  choice instead of guesswork.
#
#  Variants:
#    shipped    EXACTLY what vlm_05_reference_diff.localize() ships (multi-
#               channel union + person veto + salience cap) - the regression
#               check to run after any localizer change
#    photo      base photometric channel alone at DIFF_THRESHOLD
#    photo25/30/35   photometric at other global thresholds (kept because they
#               document WHY the multi-channel design exists: thr 25 merges
#               busy frames into MAX_AREA-killed mega-blobs, see 2026-07-12)
#    hp         per-pixel high-pass diff (NEGATIVE result kept for the record:
#               sensor/JPEG noise decorrelates between frames and floods it)
#    dino[z][@floor]  DINOv2 feature localizer INSTEAD of the diff
#               (tools/dino_localizer.py), e.g. dino4@0.10. Better on
#               cross-session noise, worse on strict IoU - see benchmark/README
#    gate[thr][mean]  shipped boxes FILTERED by DINOv2 feature support, e.g.
#               gate0.12. Same recall, ~70% fewer regions - the shipped-quality
#               boxes with the feature signal used only to veto
#    dinomaly[z][@floor]  REFERENCE-FREE: a Dinomaly model (CVPR 2025) trained on
#               this camera's nominal frames scores every patch by how badly it
#               reconstructs. No reference frame is read, so the reference's
#               session cannot leak into the comparison
#    dgate[thr][mean]  shipped boxes filtered by that model's reconstruction
#               error - the reference-free counterpart of gate
#
#  Run from the repository root:
#      python benchmark/eval_localization.py
#      python benchmark/eval_localization.py --variants shipped
#      python benchmark/eval_localization.py --variants photo,shipped --cases gpt
# =============================================================================

import sys, json, argparse
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import vlm_05_reference_diff as m

GT_PATH = Path(__file__).resolve().parent / "ground_truth.json"


# --- experimental variants (the shipped channels live in vlm_05 itself) -------

def hp_diff(a, b, black, sigma=8, thr=14):
    """Per-pixel high-pass diff. Kept as a documented dead end: JPEG/sensor
    noise is high-frequency and DIFFERS between the two frames, so |hp(a)-hp(b)|
    floods even same-session pairs (~60 regions on empty frames). The shipped
    edge channel avoids this because relu(|grad b|-|grad a|) cancels symmetric
    noise instead of summing it."""
    ha = a - m._blur_f(a, sigma)
    hb = b - m._blur_f(b, sigma)
    d = np.abs(ha - hb)
    d[black] = 0
    d = m._blur_f(d, 2, scale=4.0)
    return d > thr, d


def _photo(thr):
    def fn(a, b, black):
        return m.photo_data(a, b, black, thr=thr)
    return fn


VARIANTS = {
    "photo":   lambda a, b, k: m.photo_data(a, b, k),
    "photo25": _photo(25),
    "photo30": _photo(30),
    "photo35": _photo(35),
    "hp":      lambda a, b, k: hp_diff(a, b, k),
}


def eval_case(case, refmap, variant):
    ref_path = str(REPO_ROOT / refmap[case["reference"]])
    img_path = str(REPO_ROOT / case["image"])
    if variant == "shipped":
        regions, loc = m.localize(ref_path, img_path)
        extra = (f" base={loc['base']} +lo={loc['second']} +edge={loc['edge']}"
                 f" -person={loc['person_veto']}")
    elif variant.startswith("gate"):
        # shipped boxes, filtered by DINOv2 feature support: "gate0.10" or
        # "gate0.10mean" (default statistic is the max patch distance in the box)
        import tools.dino_localizer as dl
        spec = variant[4:]
        stat = "mean" if spec.endswith("mean") else "max"
        thr = float(spec.replace("mean", "") or dl.GATE_THRESHOLD)
        regions, loc = dl.localize_gated(ref_path, img_path, gate=thr, stat=stat)
        extra = (f" -gate={loc['gated_away']} -person={loc['person_veto']}"
                 f" {loc['seconds']}s")
    elif variant.startswith("dinomaly"):
        # REFERENCE-FREE: a Dinomaly model trained on this camera's nominal
        # frames (tools/dinomaly_train.py). "dinomaly", "dinomaly4@0.35".
        # Must be tested before the "dino" prefix below - it starts with it.
        import tools.dinomaly_localizer as dml
        spec = variant[len("dinomaly"):]
        zs, _, fs = spec.partition("@")
        regions, loc = dml.localize(ref_path, img_path,
                                    z_thr=float(zs) if zs else None,
                                    abs_floor=float(fs) if fs else None)
        extra = (f" raw={loc['raw']} patches={loc['patches_over']}"
                 f" -person={loc['person_veto']} {loc['seconds']}s")
    elif variant.startswith("bgate"):
        # BOTH gates over the shipped boxes: "bgate0.08+0.05" keeps a region only
        # if the reference features AND the nominal model both find something
        # there. The two disagree by construction - one compares to a frame, the
        # other to a model of the scene - so the question is whether their vetoes
        # are independent enough to cut more at the same recall.
        import tools.dino_localizer as dl
        import tools.dinomaly_localizer as dml
        a, _, b = variant[5:].partition("+")
        regions, loc = dl.localize_gated(ref_path, img_path, gate=float(a))
        n_dino = len(regions)
        dml.support(ref_path, img_path, regions)
        regions = [r for r in regions if r["dinomaly_max"] > float(b)]
        extra = (f" -gate={loc['gated_away']} -dgate={n_dino - len(regions)}"
                 f" -person={loc['person_veto']}")
    elif variant.startswith("dgate"):
        # shipped boxes filtered by the Dinomaly model's reconstruction error,
        # the reference-free counterpart of "gate": "dgate0.30", "dgate0.30mean"
        import tools.dinomaly_localizer as dml
        spec = variant[5:]
        stat = "mean" if spec.endswith("mean") else "max"
        thr = float(spec.replace("mean", "") or dml.GATE_THRESHOLD)
        regions, loc = dml.localize_gated(ref_path, img_path, gate=thr, stat=stat)
        extra = (f" -dgate={loc['gated_away']} -person={loc['person_veto']}"
                 f" {loc['seconds']}s")
    elif variant.startswith("dino"):
        # dino / dino3.5 / dino5 ... = DINOv2 feature localizer at that z-threshold
        # (tools/dino_localizer.py). Same post-processing as shipped, so the only
        # thing that differs is the change signal: features instead of pixels.
        import tools.dino_localizer as dl
        spec = variant[4:]                      # "", "6", "4@0.10" -> z / z@floor
        zs, _, fs = spec.partition("@")
        z = float(zs) if zs else dl.Z_THRESHOLD
        floor = float(fs) if fs else dl.ABS_FLOOR
        regions, loc = dl.localize(ref_path, img_path, z_thr=z, abs_floor=floor)
        extra = (f" raw={loc['raw']} patches={loc['patches_over']}"
                 f" -person={loc['person_veto']} {loc['seconds']}s")
    else:
        a, b, black = m._gray_pair(ref_path, img_path)
        mask, _ = VARIANTS[variant](a, b, black)
        regions = m.find_regions(mask, m.DOWNSCALE, m.DILATE,
                                 m.MIN_AREA, m.MAX_AREA)
        extra = ""
    hits = [any(m._boxes_overlap(r["bbox"], inst["bbox"]) for r in regions)
            for inst in case.get("instances", [])]
    # Strict hits are NOT decoration: the lenient rule counts a frame-sized box
    # as hitting every instance, so a merge that chains regions into a mega-blob
    # scores a perfect 45/45 while boxing nothing. Only this column catches it
    # (the 2026-07-21 merge sweep: 45/45 lenient but 22/45 strict). Same for the
    # biggest box, printed below as the blob canary.
    strict = [any(m._iou(r["bbox"], inst["bbox"]) >= 0.3 for r in regions)
              for inst in case.get("instances", [])]
    boxes = [(r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1])
             for r in regions]
    return {
        "id": case["id"], "anomaly": case["has_anomaly"],
        "n_regions": len(regions), "extra": extra,
        "types": [i["type"] for i in case.get("instances", [])],
        "hits": hits, "strict": strict, "max_box": max(boxes) if boxes else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="photo,shipped",
                    help="comma list from: shipped, dino<z>[@floor], "
                         "gate<thr>[mean], dinomaly<z>[@floor], dgate<thr>[mean], "
                         + ", ".join(VARIANTS))
    ap.add_argument("--cases", default="", help="only case ids containing this")
    ap.add_argument("--quiet", action="store_true", help="summary lines only")
    args = ap.parse_args()

    gt = json.loads(GT_PATH.read_text())
    cases = [c for c in gt["cases"] if args.cases in c["id"]]

    for name in args.variants.split(","):
        name = name.strip()
        rows = [eval_case(c, gt["references"], name) for c in cases]
        inst = sum(len(r["hits"]) for r in rows)
        det = sum(sum(r["hits"]) for r in rows)
        reg_a = sum(r["n_regions"] for r in rows if r["anomaly"])
        reg_c = sum(r["n_regions"] for r in rows if not r["anomaly"])
        per_type = {}
        for r in rows:
            for t, h in zip(r["types"], r["hits"]):
                d0, n0 = per_type.get(t, (0, 0))
                per_type[t] = (d0 + int(h), n0 + 1)
        det_s = sum(sum(r["strict"]) for r in rows)
        max_box = max((r["max_box"] for r in rows), default=0)
        print(f"\n=== {name} ===")
        print(f"instance recall: {det}/{inst}   strict IoU>=0.3: {det_s}/{inst}"
              f"   biggest box: {max_box:,} px")
        print(f"regions: {reg_a} on anomaly frames, {reg_c} on clean frames")
        print("per-type: " + "  ".join(
            f"{t}={d}/{n}" for t, (d, n) in sorted(per_type.items())))
        if args.quiet:
            continue
        for r in rows:
            missed = [t for t, h in zip(r["types"], r["hits"]) if not h]
            note = f"  MISSED:{missed}" if missed else ""
            flag = "A" if r["anomaly"] else "-"
            print(f"  [{flag}] {r['id']:<22} regions={r['n_regions']:<3}"
                  f" hit={sum(r['hits'])}/{len(r['hits'])}{note}{r['extra']}")


if __name__ == "__main__":
    main()
