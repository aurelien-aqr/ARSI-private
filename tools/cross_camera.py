#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - tools/cross_camera.py
#  Does a Dinomaly model trained on ONE camera work on ANOTHER camera?
#
#  WHY. Every Dinomaly verdict in docs/DECISIONS.md was measured with each
#  camera's own checkpoint. The cost objection recorded there - "it needs a
#  checkpoint per camera" - is only a real objection if a foreign checkpoint
#  actually fails. Nobody had measured that. Dinomaly's own selling point
#  (CVPR 2025) is a single multi-class model, so this is the number that decides
#  whether their argument applies to us or not.
#
#  WHAT IT MEASURES. Image-level separability, the standard I-AUROC of the
#  anomaly-detection literature, so the result lands in the same table as any
#  PatchCore / EfficientAD baseline:
#
#      score(image) = mean of the TOP_Q quantile of the patch anomaly map
#      I-AUROC      = P(score(anomalous frame) > score(clean frame))
#
#  For every (train camera, test camera) pair. The diagonal is the shipped
#  configuration; the off-diagonal is transfer. Also reported, because AUROC on
#  6 frames is fragile:
#    - clean-frame score level, which says whether a foreign model is simply
#      offset (recalibratable) or actually blind (not);
#    - the same numbers on the 18 clean 1760 frames, a pure-negative set: a model
#      that fires there is producing false alarms on an empty tram.
#
#  COST. One DINOv2 encode per benchmark image, shared across all checkpoints
#  (the encoder is frozen and identical - only the decoders differ), then one
#  cheap decoder pass per model. CPU is enough.
#
#  Usage:  python tools/cross_camera.py [--out docs/dino_models/cross_camera.json]
# =============================================================================

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import tools.dinomaly as dm                                    # noqa: E402
import tools.dinomaly2 as d2                                   # noqa: E402

GT = REPO_ROOT / "benchmark" / "datasets" / "ground_truth.json"
TOP_Q = 0.99          # image score = mean of the worst 1% of patches. Max alone
#                       is one patch of noise; the mean of the map is dominated
#                       by the 99% of the frame where nothing happens.


def image_score(smap: np.ndarray, valid: np.ndarray) -> float:
    v = smap[valid.astype(bool)] if valid is not None else smap.ravel()
    if v.size == 0:
        return float("nan")
    thr = np.quantile(v, TOP_Q)
    top = v[v >= thr]
    return float(top.mean()) if top.size else float(v.max())


def auroc(pos, neg) -> float:
    """Rank-based AUROC, ties counted as half. Undefined without both classes."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (pos.size * neg.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--models", default="", help="comma-separated subset of checkpoints")
    ap.add_argument("--variant", default="v1", choices=["v1", "v2"],
                    help="v1 = tools/dinomaly.py (CVPR 2025), "
                         "v2 = tools/dinomaly2.py (arXiv 2510.17611)")
    args = ap.parse_args()
    v2 = args.variant == "v2"
    # With DINOMALY2_CAR=0 the v2 module renames itself to dinomaly2nocar, which
    # switches both the checkpoints read here and the recentering applied when
    # scoring. Following d2.TAG rather than hard-coding "dinomaly2_" is what
    # keeps the ablation from silently scoring CAR checkpoints without CAR.
    out_default = (f"docs/dino_models/cross_camera_{d2.TAG.replace('dinomaly', 'v')}.json"
                   if v2 else "docs/dino_models/cross_camera.json")

    gt = json.loads(GT.read_text())
    cases = gt["cases"]

    prefix = f"{d2.TAG}_" if v2 else "dinomaly_"
    models = args.models.split(",") if args.models else sorted(
        p.stem[len(prefix):] for p in dm.WEIGHTS_DIR.glob(f"{prefix}*.pt")
        if not p.stem.endswith("_baseline")
    )
    print(f"[{d2.TAG if v2 else 'dinomaly'}] {len(models)} checkpoints: {', '.join(models)}", flush=True)

    # A case belongs to the camera of the reference it is diffed against; 'real'
    # and 'variant' are both the tram_1762 camera, and 'variant' is a synthetic
    # scene, so it is kept out of the per-camera tables.
    def camera_of(case):
        ref = case["reference"]
        return "tram_1762" if ref == "real" else ref

    work = [c for c in cases if c["reference"] != "variant"]
    by_cam = defaultdict(lambda: {"pos": [], "neg": []})
    for c in work:
        by_cam[camera_of(c)]["pos" if c["has_anomaly"] else "neg"].append(c)
    for cam, d in sorted(by_cam.items()):
        print(f"  {cam:14s} {len(d['pos']):2d} anomalous / {len(d['neg']):2d} clean", flush=True)

    loaded = {m: (d2.load(m) if v2 else dm.load(m))[0] for m in models}

    # ---- encode once per image, score with every checkpoint -----------------
    # The encoder is frozen and shared by every checkpoint, so it runs once per
    # image and only the (cheap) decoders are repeated.
    scores = defaultdict(dict)            # scores[model][case_id] = float
    t0 = time.time()
    for i, c in enumerate(work, 1):
        path = REPO_ROOT / c["image"]
        if v2:
            layers, cls, valid, grid = d2.encode(str(path))
        else:
            layers, valid, grid = dm.encode(str(path))
        vgrid = valid.numpy().reshape(grid)
        for m in models:
            model = loaded[m]
            smap = (d2.anomaly_map(model, layers, cls, valid, grid) if v2
                    else dm.anomaly_map(model, layers, valid, grid))
            scores[m][c["id"]] = image_score(smap, vgrid)
        if i % 5 == 0 or i == len(work):
            print(f"  {i}/{len(work)} images  ({time.time() - t0:.0f}s)", flush=True)

    # ---- matrix -------------------------------------------------------------
    test_cams = sorted(by_cam)
    matrix, detail = {}, {}
    for m in models:
        matrix[m], detail[m] = {}, {}
        for cam in test_cams:
            pos = [scores[m][c["id"]] for c in by_cam[cam]["pos"]]
            neg = [scores[m][c["id"]] for c in by_cam[cam]["neg"]]
            matrix[m][cam] = auroc(pos, neg)
            detail[m][cam] = {
                "n_pos": len(pos), "n_neg": len(neg),
                "mean_pos": float(np.mean(pos)) if pos else None,
                "mean_neg": float(np.mean(neg)) if neg else None,
            }
        # pooled over the four 3333 cameras: 14 anomalous / 7 clean, the only
        # sample big enough for an AUROC that means something
        p3 = [scores[m][c["id"]] for cam in test_cams if cam.startswith("3333")
              for c in by_cam[cam]["pos"]]
        n3 = [scores[m][c["id"]] for cam in test_cams if cam.startswith("3333")
              for c in by_cam[cam]["neg"]]
        matrix[m]["3333-pooled"] = auroc(p3, n3)

    out = {
        "variant": d2.TAG if v2 else "dinomaly",
        "_about": ("Cross-camera transfer of the per-camera Dinomaly checkpoints. "
                   f"Image score = mean of the top {100*(1-TOP_Q):.0f}% of patch "
                   "anomaly scores, baseline subtraction OFF (the baseline is "
                   "per-camera and would confound the transfer being measured). "
                   "Rows = training camera, columns = test camera; the diagonal "
                   "is the shipped configuration."),
        "top_quantile": TOP_Q,
        "models": models,
        "test_cameras": test_cams,
        "auroc": matrix,
        "detail": detail,
        "scores": {m: scores[m] for m in models},
    }
    out_path = REPO_ROOT / (args.out or out_default)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))

    # ---- print --------------------------------------------------------------
    cols = test_cams + ["3333-pooled"]
    print("\nI-AUROC  (row = trained on, col = tested on)")
    print(f"{'':16s}" + "".join(f"{c[-9:]:>11s}" for c in cols))
    for m in models:
        cells = []
        for c in cols:
            v = matrix[m][c]
            txt = "-" if np.isnan(v) else f"{v:.3f}" + ("*" if m == c else "")
            cells.append(f"{txt:>11s}")
        print(f"{m:16s}" + "".join(cells))
    print("* = the camera this checkpoint was trained on (shipped configuration)")
    print(f"\n-> {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
