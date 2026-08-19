#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - benchmark/eval_localization.py
#  Localization-ONLY evaluation of the vlm_05 change detector - no VLM calls.
#
#  The full benchmark (run_benchmark.py) costs ~15 s of CPU VLM time per region.
#  But the change-detection stage can be scored on its own against the instance
#  boxes of a dataset in benchmark/datasets/ in SECONDS: an instance is "localized" if
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
#      python benchmark/eval_localization.py --dataset 39T --variants shipped
# =============================================================================

import sys, argparse, json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import vlm_05_reference_diff as m
from arsi_core import benchmarks
import tools.localizer_specs as specs




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
    if variant in VARIANTS:
        a, b, black = m._gray_pair(ref_path, img_path)
        mask, _ = VARIANTS[variant](a, b, black)
        regions = m.find_regions(mask, m.DOWNSCALE, m.DILATE,
                                 m.MIN_AREA, m.MAX_AREA)
        extra = ""
    else:
        # every non-threshold-sweep variant is defined once, in
        # tools/localizer_specs, so this script and tools/rescore_localizer.py
        # cannot disagree about what a spec means
        regions, loc = specs.propose(variant, ref_path, img_path,
                                     specs.camera_of(case))
        extra = "  " + " ".join(f"{k}={loc[k]}" for k in
                                ("raw", "base", "second", "edge", "gated_away",
                                 "vetoed", "person_veto", "seconds")
                                if k in loc)
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
    ap.add_argument("--gt", "--dataset", dest="gt", default=benchmarks.DEFAULT,
                    help="dataset id or path (default: the benchmark ground "
                         "truth, every case)")
    ap.add_argument("--ref", default="", help="only cases whose reference key "
                    "contains this (e.g. --ref 39T for the 39T cameras)")
    ap.add_argument("--json", default="", help="also write the summary here, "
                    "with a per-camera breakdown - what the reports read")
    args = ap.parse_args()

    ds_id, gt = benchmarks.load(args.gt)
    cases = [c for c in gt["cases"] if args.cases in c["id"]
             and args.ref in c.get("reference", "")]
    print(f"dataset: {ds_id}  ({len(cases)} of {len(gt['cases'])} cases)")

    out = {"dataset": ds_id, "n_cases": len(cases), "variants": {}}
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
        out["variants"][name] = {
            "instances": inst, "detected": det, "strict": det_s,
            "regions_anomaly": reg_a, "regions_clean": reg_c,
            "max_box": max_box,
            "per_type": {t: list(v) for t, v in sorted(per_type.items())},
            # per family, because that is the comparison the reports make: one
            # camera with 559 regions and three with 40 must not average away
            "by_family": _by_family(cases, rows),
        }
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

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


def _by_family(cases, rows):
    """Totals per camera family (tram_1762 / 1760 / 39T), keyed as the reports
    name them. A case's family is read off its reference key, the same way
    camera_of() picks the checkpoint."""
    fam = {}
    for c, r in zip(cases, rows):
        ref = c.get("reference", "")
        key = ref.split("-")[0] if "-cam" in ref else "tram_1762"
        d = fam.setdefault(key, {"cases": 0, "instances": 0, "detected": 0,
                                 "strict": 0, "regions_anomaly": 0,
                                 "regions_clean": 0})
        d["cases"] += 1
        d["instances"] += len(r["hits"])
        d["detected"] += sum(r["hits"])
        d["strict"] += sum(r["strict"])
        d["regions_anomaly" if r["anomaly"] else "regions_clean"] += r["n_regions"]
    return fam


if __name__ == "__main__":
    main()
