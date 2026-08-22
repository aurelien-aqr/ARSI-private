#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - tools/anomalib_baseline.py
#  The standard anomaly-detection baselines, on OUR cameras, with OUR split.
#
#  WHY. docs/DECISIONS.md records anomalib as "not adopted - worth it only to
#  produce a standard baseline table for the paper". That is exactly what this
#  is. Every claim we make about localizers has so far been measured against our
#  own pixel diff; a reviewer will ask what PatchCore does on the same frames,
#  and until now we could not answer.
#
#  WHAT IT RUNS. anomalib 2.6 ships reference implementations of most of the
#  methods in docs/paper/related_work.md, including Dinomaly and AnomalyDINO
#  themselves. So this table does two jobs at once:
#    - it gives the paper its baseline row (PatchCore, PaDiM, ...);
#    - it puts OUR simplified Dinomaly re-implementation next to the reference
#      one on identical data, which is the only honest way to report a number
#      from a re-implementation.
#
#  PROTOCOL, identical to tools/cross_camera.py so the numbers can share a table:
#    train  the camera's nominal frames (the same list dinomaly_train.py uses,
#           benchmark holdout included - see tools/dinomaly_train.nominal_frames)
#    test   that camera's benchmark cases, anomalous vs clean
#    metric image-level AUROC, computed here from the raw scores rather than
#           read out of a framework metric, so the definition cannot drift
#
#  Cameras with no labelled anomalies (the three 1760 views are clean-only) are
#  reported for their score level, not for an AUROC that does not exist.
#
#  Usage (needs the separate anomalib venv - it pins its own lightning):
#      ~/venv-anomalib/bin/python tools/anomalib_baseline.py --models patchcore,padim
# =============================================================================

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

GT = REPO_ROOT / "benchmark" / "datasets" / "ground_truth.json"
STAGE = Path("/tmp/arsi_anomalib")

MODELS = {                       # name -> (class path, kwargs)
    "padim":       ("Padim", {}),
    "patchcore":   ("Patchcore", {}),
    "dinomaly":    ("Dinomaly", {}),
    "anomalydino": ("AnomalyDINO", {}),
    "efficientad": ("EfficientAd", {}),
    "winclip":     ("WinClip", {}),
}
# max_epochs per model: the memory-bank methods (PatchCore, PaDiM, AnomalyDINO)
# fit in one pass over the training set and must not be run longer.
ONE_EPOCH = {"padim", "patchcore", "anomalydino", "winclip"}


def camera_of(case):
    return "tram_1762" if case["reference"] == "real" else case["reference"]


def build_stage(camera, train_files, pos, neg):
    """train/good, test/good, test/bad as symlinks - anomalib wants directories
    and our frames live in three different trees."""
    root = STAGE / camera
    if root.exists():
        shutil.rmtree(root)
    for sub in ("train/good", "test/good", "test/bad"):
        (root / sub).mkdir(parents=True)
    for f in train_files:
        (root / "train/good" / Path(f).name).symlink_to(Path(f).resolve())
    for c in neg:
        p = REPO_ROOT / c["image"]
        (root / "test/good" / f"{c['id']}{p.suffix}").symlink_to(p.resolve())
    for c in pos:
        p = REPO_ROOT / c["image"]
        (root / "test/bad" / f"{c['id']}{p.suffix}").symlink_to(p.resolve())
    return root


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (pos.size * neg.size))


def run_one(model_name, root, seed=0):
    """Fit on train/good, score test/*. Returns {stem: score, ...} and 'bad'
    membership taken from the staged folder, not from the framework's labels."""
    import torch
    from anomalib.data import Folder
    from anomalib.data.utils.split import TestSplitMode, ValSplitMode
    from anomalib.engine import Engine
    import anomalib.models as M

    cls_name, kwargs = MODELS[model_name]
    model = getattr(M, cls_name)(**kwargs)
    dm = Folder(
        name=root.name,
        root=root,
        normal_dir="train/good",
        abnormal_dir="test/bad",
        normal_test_dir="test/good",
        train_batch_size=1 if model_name == "efficientad" else 8,
        eval_batch_size=4,
        num_workers=4,
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.SAME_AS_TEST,   # never carve up a 6-frame
        #                                             test set into a val split
        seed=seed,
    )
    # anomalib installs its own ModelCheckpoint callback, so checkpointing
    # cannot be disabled - point it at a scratch dir instead of fighting it.
    engine = Engine(default_root_dir=str(STAGE / "_runs"), logger=False,
                    max_epochs=1 if model_name in ONE_EPOCH else 10,
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices=1,
                    enable_progress_bar=False, enable_model_summary=False)
    engine.fit(model=model, datamodule=dm)
    preds = engine.predict(model=model, datamodule=dm)

    scores = {}
    for batch in preds or []:
        paths = batch.image_path if hasattr(batch, "image_path") else batch["image_path"]
        sc = batch.pred_score if hasattr(batch, "pred_score") else batch["pred_score"]
        sc = sc.detach().cpu().numpy().reshape(-1)
        for p, s in zip(paths, sc):
            scores[Path(p).stem] = float(s)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="padim,patchcore",
                    help=f"subset of {','.join(MODELS)}")
    ap.add_argument("--cameras", default="")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default="docs/dino_models/anomalib_baseline.json")
    args = ap.parse_args()

    from tools.dinomaly_train import nominal_frames

    gt = json.loads(GT.read_text())
    by_cam = defaultdict(lambda: {"pos": [], "neg": []})
    for c in gt["cases"]:
        if c["reference"] == "variant":
            continue
        by_cam[camera_of(c)]["pos" if c["has_anomaly"] else "neg"].append(c)

    cams = args.cameras.split(",") if args.cameras else sorted(by_cam)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    results, failures = defaultdict(dict), {}
    for cam in cams:
        pos, neg = by_cam[cam]["pos"], by_cam[cam]["neg"]
        train_files, _, _ = nominal_frames(cam, args.stride)
        root = build_stage(cam, train_files, pos, neg)
        print(f"\n=== {cam}: {len(train_files)} train / {len(pos)} anomalous "
              f"/ {len(neg)} clean", flush=True)
        for m in models:
            t0 = time.time()
            try:
                sc = run_one(m, root)
            except Exception as exc:                      # noqa: BLE001
                failures[f"{cam}/{m}"] = f"{type(exc).__name__}: {exc}"
                print(f"  {m:12s} FAILED: {type(exc).__name__}: {exc}", flush=True)
                continue
            p = [sc[c["id"]] for c in pos if c["id"] in sc]
            n = [sc[c["id"]] for c in neg if c["id"] in sc]
            a = auroc(p, n)
            results[m][cam] = {"auroc": a, "n_pos": len(p), "n_neg": len(n),
                               "mean_pos": float(np.mean(p)) if p else None,
                               "mean_neg": float(np.mean(n)) if n else None,
                               "seconds": round(time.time() - t0, 1),
                               "scores": sc}
            print(f"  {m:12s} I-AUROC {a:.3f}   ({time.time() - t0:.0f} s)", flush=True)

    # pooled 3333
    for m in models:
        p3 = [s for cam in cams if cam.startswith("3333") and cam in results[m]
              for c in by_cam[cam]["pos"] for s in [results[m][cam]["scores"].get(c["id"])]
              if s is not None]
        n3 = [s for cam in cams if cam.startswith("3333") and cam in results[m]
              for c in by_cam[cam]["neg"] for s in [results[m][cam]["scores"].get(c["id"])]
              if s is not None]
        if p3 and n3:
            results[m]["3333-pooled"] = {"auroc": auroc(p3, n3),
                                         "n_pos": len(p3), "n_neg": len(n3)}

    out = {"_about": ("anomalib 2.6 reference implementations on the ARSI cameras. "
                      "Train = that camera's nominal frames, test = its benchmark "
                      "cases. I-AUROC computed from raw predicted scores, same "
                      "definition as tools/cross_camera.py. Pooled 3333 scores are "
                      "NOT comparable across cameras for the memory-bank methods - "
                      "each camera has its own fitted model, so read the per-camera "
                      "rows first."),
           "models": models, "stride": args.stride,
           "results": {m: dict(v) for m, v in results.items()},
           "failures": failures}
    (REPO_ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / args.out).write_text(json.dumps(out, indent=1))

    cols = [c for c in cams] + ["3333-pooled"]
    print("\nI-AUROC")
    print(f"{'':14s}" + "".join(f"{c[-9:]:>11s}" for c in cols))
    for m in models:
        cells = []
        for c in cols:
            r = results[m].get(c)
            cells.append(f"{r['auroc']:>11.3f}" if r and not np.isnan(r["auroc"])
                         else f"{'-':>11s}")
        print(f"{m:14s}" + "".join(cells))
    print(f"\n-> {args.out}")
    if failures:
        print(f"{len(failures)} failures recorded in the JSON")


if __name__ == "__main__":
    main()
