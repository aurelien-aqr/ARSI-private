#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - tools/cdnet_eval.py
#  The change-proposal stage, evaluated on CDnet 2014 - i.e. on data we did not
#  build, against ground truth we did not draw.
#
#  WHY. docs/PUBLIC_DATASETS.md sets this out already: "I don't think having our
#  own dataset is what gets a paper rejected. Having our own dataset and nothing
#  else is." CDnet is the benchmark that lets us evaluate vlm_05's proposal stage
#  WITHOUT the VLM in the loop, so that when the full pipeline is wrong we can
#  say whether the region was never proposed or was proposed and misjudged.
#
#  WHAT THIS IS AND IS NOT. CDnet ground truth marks MOTION, not abnormality.
#  Nothing in it knows that a bottle was forgotten. The claim this licenses is
#  "our change-detection stage reaches F1 = x on intermittentObjectMotion",
#  never "our anomaly detector reaches F1 = x". The category is the relevant one:
#  its authors define it as "videos containing background objects moving away,
#  abandoned objects and objects stopping for a short while and then moving away".
#
#  PROTOCOL, and every deviation from the official one is named here.
#    reference   CDnet gives no clean reference frame, so we build one the way
#                the benchmark intends its initialisation period to be used: the
#                per-pixel MEDIAN of up to --ref-frames frames sampled uniformly
#                from before temporalROI starts. Where temporalROI starts at 1
#                there is no initialisation period and we fall back to the first
#                frames, which is noted per sequence in the JSON.
#    frames      every --stride-th frame of the temporal ROI. Uniform, so it
#                changes the error bars and not the estimate. Full-ROI runs are
#                possible with --stride 1 and cost ~10x.
#    labels      official CDnet semantics: 255 motion, 50 hard shadow (counts as
#                BACKGROUND), 0 static, 85 outside ROI and 170 unknown are both
#                EXCLUDED, as is anything outside ROI.bmp.
#    metrics     the seven CDnet metrics: Re, Sp, FPR, FNR, PWC, Pr, F1.
#    person veto OFF. On CDnet the foreground is mostly people; our person filter
#                exists to suppress passengers inside a tram and would measure
#                the exact opposite of what this benchmark asks. Leaving it on
#                would be cheating in reverse.
#
#  TWO OUTPUTS PER SEQUENCE, because our stage produces both and they are not
#  the same object:
#    mask   the raw photometric change mask (vlm_05.change_mask) - a genuine
#           per-pixel change detector, directly comparable to the CDnet leaderboard
#    boxes  the rasterised region proposals, i.e. what actually reaches the VLM.
#           Boxes over-cover pixel-accurate ground truth by construction, so
#           their precision is structurally lower. The gap between the two rows
#           IS the cost of boxing, and it is worth reporting rather than hiding.
#
#  Usage:
#    python tools/cdnet_eval.py --category intermittentObjectMotion
#    python tools/cdnet_eval.py --category shadow --localizers photo,dino
# =============================================================================

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Override with CDNET_ROOT when the dataset does not sit next to the repo (it is
# 4.4 GB and lives outside it on purpose).
DATASET = Path(os.environ.get(
    "CDNET_ROOT", "/home/aurelien/Documents/vsb/dataset/CDnet_2014"))
SCRATCH = Path("/tmp/arsi_cdnet")

# CDnet ground-truth palette
GT_STATIC, GT_SHADOW, GT_OUTSIDE, GT_UNKNOWN, GT_MOTION = 0, 50, 85, 170, 255


def sequences(category: str):
    root = DATASET / category
    return sorted(p for p in root.iterdir() if p.is_dir())


def temporal_roi(seq: Path):
    a, b = (seq / "temporalROI.txt").read_text().split()
    return int(a), int(b)


def spatial_roi(seq: Path, shape=None):
    """CDnet ships ROI.bmp at the frame size for most sequences but NOT for all
    (cameraJitter is 352 wide against 320-wide frames), so it is resampled to the
    frame shape with nearest neighbour rather than trusted."""
    for name in ("ROI.bmp", "ROI.jpg"):
        p = seq / name
        if not p.exists():
            continue
        im = Image.open(p).convert("L")
        if shape is not None and (im.height, im.width) != tuple(shape):
            im = im.resize((shape[1], shape[0]), Image.NEAREST)
        return np.asarray(im) > 127
    return None


def build_reference(seq: Path, start: int, n: int = 50):
    """Per-pixel median of the initialisation period. Returns (path, note)."""
    frames = seq / "input"
    hi = start - 1
    note = "median of the pre-ROI initialisation period"
    if hi < 5:                       # no initialisation period in this sequence
        hi = min(50, len(list(frames.glob("in*.jpg"))))
        note = ("temporalROI starts at 1: no initialisation period, "
                "median of the first frames of the ROI itself (optimistic)")
    idx = np.unique(np.linspace(1, hi, min(n, hi)).astype(int))
    stack = np.stack([np.asarray(Image.open(frames / f"in{i:06d}.jpg").convert("RGB"),
                                 dtype=np.uint8) for i in idx])
    med = np.median(stack, axis=0).astype(np.uint8)
    out = SCRATCH / seq.parent.name / seq.name
    out.mkdir(parents=True, exist_ok=True)
    ref = out / "reference.png"
    Image.fromarray(med).save(ref)
    return ref, note, len(idx)


def apply_roi(img_path: Path, roi, out_path: Path):
    """Black out everything outside the spatial ROI, the same way our pipeline
    blacks out tram windows - so the localizer cannot propose there at all."""
    im = np.asarray(Image.open(img_path).convert("RGB")).copy()
    if roi is not None:
        im[~roi] = 0
    Image.fromarray(im).save(out_path)
    return out_path


def rasterize(regions, shape):
    m = np.zeros(shape, bool)
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    return m


def confusion(pred: np.ndarray, gt: np.ndarray, roi):
    """CDnet counting rules. Returns (tp, fp, fn, tn)."""
    valid = (gt == GT_MOTION) | (gt == GT_STATIC) | (gt == GT_SHADOW)
    if roi is not None:
        valid &= roi
    fg = (gt == GT_MOTION) & valid
    bg = valid & ~fg
    tp = int(np.count_nonzero(pred & fg))
    fn = int(np.count_nonzero(~pred & fg))
    fp = int(np.count_nonzero(pred & bg))
    tn = int(np.count_nonzero(~pred & bg))
    return tp, fp, fn, tn


def metrics(tp, fp, fn, tn):
    d = lambda a, b: (a / b) if b else float("nan")            # noqa: E731
    re = d(tp, tp + fn)
    pr = d(tp, tp + fp)
    return {
        "Recall": re,
        "Specificity": d(tn, tn + fp),
        "FPR": d(fp, fp + tn),
        "FNR": d(fn, tp + fn),
        "PWC": 100 * d(fn + fp, tp + fn + fp + tn),
        "Precision": pr,
        "F1": d(2 * pr * re, pr + re) if (pr == pr and re == re and pr + re) else float("nan"),
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def run_sequence(seq: Path, localizers, stride: int, ref_frames: int, limit: int,
                 params: dict = None):
    import vlm_05_reference_diff as v5
    from arsi_core import localizers as loc

    v5.PERSON_FILTER = False        # see the header: mandatory on CDnet

    start, end = temporal_roi(seq)
    probe = np.asarray(Image.open(seq / "input" / f"in{start:06d}.jpg"))
    roi = spatial_roi(seq, probe.shape[:2])
    ref, ref_note, n_ref = build_reference(seq, start, ref_frames)
    ref_masked = apply_roi(ref, roi, ref.with_name("reference_roi.png"))

    idx = list(range(start, end + 1, stride))
    if limit:
        idx = idx[:limit]
    acc = {f"{name}:{mode}": [0, 0, 0, 0]
           for name in localizers for mode in ("mask", "boxes")
           if not (mode == "mask" and name != "photo")}
    work = SCRATCH / seq.parent.name / seq.name
    t0 = time.time()
    for k, i in enumerate(idx, 1):
        gt = np.asarray(Image.open(seq / "groundtruth" / f"gt{i:06d}.png").convert("L"))
        insp = apply_roi(seq / "input" / f"in{i:06d}.jpg", roi, work / "insp.png")
        for name in localizers:
            if name == "photo":                       # raw per-pixel change mask
                m = v5.change_mask(str(ref_masked), str(insp))
                for j, v in enumerate(confusion(m, gt, roi)):
                    acc["photo:mask"][j] += v
            regions, _ = loc.localize(name, v5, str(ref_masked), str(insp), params)
            m = rasterize(regions, gt.shape)
            for j, v in enumerate(confusion(m, gt, roi)):
                acc[f"{name}:boxes"][j] += v
        if k % 25 == 0 or k == len(idx):
            print(f"    {k}/{len(idx)} frames ({time.time() - t0:.0f}s)", flush=True)

    return {
        "frames_evaluated": len(idx), "temporal_roi": [start, end], "stride": stride,
        "reference": {"note": ref_note, "n_frames": n_ref},
        "results": {k: metrics(*v) for k, v in acc.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="intermittentObjectMotion")
    ap.add_argument("--sequences", default="", help="comma-separated subset")
    ap.add_argument("--localizers", default="photo")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--ref-frames", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap frames/sequence")
    ap.add_argument("--params", default="",
                    help="localizer overrides, e.g. DINO_Z=2.5,DINO_FLOOR=0.04")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    localizers = [s.strip() for s in args.localizers.split(",") if s.strip()]
    params = dict(kv.split("=", 1) for kv in args.params.split(",")) if args.params else {}
    seqs = sequences(args.category)
    if args.sequences:
        keep = set(args.sequences.split(","))
        seqs = [s for s in seqs if s.name in keep]

    print(f"CDnet 2014 / {args.category}: {len(seqs)} sequences, "
          f"localizers {localizers}, stride {args.stride}", flush=True)

    out = {"category": args.category, "stride": args.stride,
           "localizers": localizers, "params": params, "sequences": {}}
    for seq in seqs:
        print(f"  {seq.name}", flush=True)
        out["sequences"][seq.name] = run_sequence(
            seq, localizers, args.stride, args.ref_frames, args.limit, params)

    # category totals: sum the confusion matrices, then compute once. CDnet's own
    # leaderboard averages per-sequence metrics instead; both are reported.
    for key in {k for s in out["sequences"].values() for k in s["results"]}:
        tot = [0, 0, 0, 0]
        per = []
        for s in out["sequences"].values():
            c = s["results"][key]["counts"]
            tot[0] += c["tp"]; tot[1] += c["fp"]; tot[2] += c["fn"]; tot[3] += c["tn"]
            per.append(s["results"][key])
        out.setdefault("category_pooled", {})[key] = metrics(*tot)
        out.setdefault("category_mean_of_sequences", {})[key] = {
            m: float(np.nanmean([p[m] for p in per]))
            for m in ("Recall", "Specificity", "FPR", "FNR", "PWC", "Precision", "F1")}

    path = Path(args.out) if args.out else (
        REPO_ROOT / f"docs/public_benchmarks/cdnet_{args.category}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))

    print(f"\n{args.category}: pooled over all evaluated frames")
    print(f"{'':16s}{'Recall':>9s}{'Prec':>9s}{'F1':>9s}{'Sp':>9s}{'FPR':>9s}{'PWC':>9s}")
    for key, m in sorted(out["category_pooled"].items()):
        print(f"{key:16s}{m['Recall']:>9.3f}{m['Precision']:>9.3f}{m['F1']:>9.3f}"
              f"{m['Specificity']:>9.3f}{m['FPR']:>9.4f}{m['PWC']:>9.2f}")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
