#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - tools/dinomaly_localizer.py
#  Dinomaly as a vlm_05 localizer: same (regions, info) contract as
#  vlm_05_reference_diff.localize() and tools/dino_localizer.localize().
#
#  WHAT IS DIFFERENT FROM THE OTHER TWO. photo and photo+dino both compare the
#  inspection frame to ONE reference frame. This one compares it to a MODEL of
#  the nominal scene trained on many frames (tools/dinomaly_train.py), so the
#  reference argument is accepted only to keep the contract - it is used for the
#  image size, never read as an image. That is the whole point of trying it: the
#  cross-session noise that survives the DINOv2 gate is the reference's session
#  leaking into the comparison, and there is no reference here to leak.
#
#  Everything AFTER the score map is the SHIPPED machinery again
#  (dino_localizer.regions_from_patches -> find_regions, person veto, salience
#  cap, merge), so a diff against photo or photo+dino isolates the proposal
#  signal alone.
#
#  Usage:
#      import tools.dinomaly_localizer as dml
#      regions, info = dml.localize(ref, img)            # reference-free
#      regions, info = dml.localize_gated(ref, img)      # shipped boxes, vetoed
#      python tools/dinomaly_localizer.py REF INSP [--out out.jpg]
#      python tools/dinomaly_localizer.py --stats        # score distribution
# =============================================================================

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import tools.dinomaly as dm                                    # noqa: E402
import tools.dino_localizer as dl                              # noqa: E402

CAMERA = "tram_1762"
Z_THRESHOLD = 4.0     # robust z of the anomaly map, same rationale as dino_localizer
ABS_FLOOR = 0.07      # measured 2026-08-16 on the 29 benchmark cases (--stats):
#                       clean frames sit at p99 0.07-0.08, real instances peak at
#                       0.12-0.28, so this is the highest floor that keeps them
GATE_THRESHOLD = 0.05 # localize_gated: the recall-first pick - 0.05 is the
#                       largest veto threshold that still keeps 45/45 instances
#                       (0.08 drops 6 of them). See benchmark/README.

_models = {}
_score_cache = {}


def _model(camera: str):
    if camera not in _models:
        _models[camera] = dm.load(camera)
    return _models[camera]


#  OFF by default, measured: subtracting the per-position baseline buys a much
#  cheaper gate (106-139 regions vs 243) but never reaches full recall - even at
#  a 0.01 veto it drops one instance, because a patch the decoder is weak at also
#  has its real objects partially cancelled. Recall-first says off; set
#  DINOMALY_BASELINE=1 to trade an instance for half the VLM calls.
BASELINE = os.environ.get("DINOMALY_BASELINE", "0") != "0"
BASELINE_FRAMES = 40


def _baseline(camera: str, meta: dict):
    """Mean reconstruction error per patch POSITION, over the training frames.

    Measured need (2026-08-16 heatmap): the error is high on thin structures -
    handrails, poster frames, the mask borders - on EVERY frame, nominal or not.
    Those are not anomalies, they are places this decoder is simply weak, and
    they dominated the standalone localizer's proposals. Subtracting the model's
    own typical error at each position asks the right question: is this patch
    worse than usual HERE, rather than worse than average.

    Cached next to the checkpoint; costs one encode per frame the first time."""
    path = dm.WEIGHTS_DIR / f"dinomaly_{camera}_baseline.npy"
    if path.exists():
        return np.load(path)
    files = meta.get("frames", [])[::max(1, len(meta.get("frames", [1])) // BASELINE_FRAMES)]
    model, _ = _model(camera)
    acc = None
    for i, f in enumerate(files, 1):
        layers, valid, grid = dm.encode(str(REPO_ROOT / f))
        s = dm.anomaly_map(model, layers, valid, grid)
        acc = s if acc is None else acc + s
        if i % 10 == 0:
            print(f"  baseline {i}/{len(files)}", flush=True)
    acc = acc / max(1, len(files))
    np.save(path, acc)
    return acc


def score_map(inspection_path: str, camera: str = CAMERA):
    """(score[gh,gw], valid[gh,gw], meta). Reference-free: the score is how badly
    the nominal model failed to reconstruct each patch, minus how badly it fails
    there on nominal frames (see _baseline; DINOMALY_BASELINE=0 to compare)."""
    key = (str(inspection_path), camera, dm.INPUT_W, BASELINE)
    if key in _score_cache:
        return _score_cache[key]
    model, meta = _model(camera)
    layers, valid, grid = dm.encode(str(inspection_path))
    smap = dm.anomaly_map(model, layers, valid, grid)
    if BASELINE:
        smap = np.clip(smap - _baseline(camera, meta), 0.0, None)
    _score_cache[key] = (smap, valid.numpy().reshape(grid), meta)
    return _score_cache[key]


def localize(reference_path: str, inspection_path: str, z_thr: float = None,
             abs_floor: float = None, camera: str = CAMERA):
    """vlm_05 contract. `reference_path` is used for the output pixel space only.

    A patch is a candidate when it is BOTH a robust-z outlier of this frame's own
    score distribution AND past an absolute floor - the same pair of conditions
    dino_localizer needs, for the same measured reason: on a frame the model
    reconstructs well the MAD collapses and the z alone turns jitter into
    regions."""
    z_thr = Z_THRESHOLD if z_thr is None else z_thr
    abs_floor = ABS_FLOOR if abs_floor is None else abs_floor
    t0 = time.time()
    size = Image.open(reference_path).size
    smap, valid, meta = score_map(inspection_path, camera)

    d = smap[valid]
    med = float(np.median(d)) if d.size else 0.0
    mad = float(np.median(np.abs(d - med))) if d.size else 0.0
    scale = 1.4826 * mad if mad > 1e-6 else (float(d.std()) or 1e-6)
    z = (smap - med) / scale
    z[~valid] = 0.0

    over = (z > z_thr) & (smap > abs_floor)
    return dl.regions_from_patches(over, z, inspection_path, size, "dinomaly",
                                   {"z_thr": z_thr, "floor": abs_floor,
                                    "trained_on": meta.get("n_frames", 0)}, t0)


def support(reference_path: str, inspection_path: str, regions,
            camera: str = CAMERA, stat: str = "max"):
    """Annotate shipped regions with dinomaly_max / dinomaly_mean, the model's
    reconstruction error inside the box."""
    size = Image.open(reference_path).size
    smap, valid, _ = score_map(inspection_path, camera)
    gh, gw = smap.shape
    W, H = size
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        j0, j1 = int(x0 * gw / W), max(int(x0 * gw / W) + 1, int(np.ceil(x1 * gw / W)))
        i0, i1 = int(y0 * gh / H), max(int(y0 * gh / H) + 1, int(np.ceil(y1 * gh / H)))
        p, v = smap[i0:i1, j0:j1], valid[i0:i1, j0:j1]
        p = p[v] if v.any() else p.ravel()
        r["dinomaly_max"] = float(p.max()) if p.size else 0.0
        r["dinomaly_mean"] = float(p.mean()) if p.size else 0.0
    return regions


def localize_gated(reference_path: str, inspection_path: str, gate: float = None,
                   stat: str = "max", camera: str = CAMERA):
    """The shipped pixel-diff boxes, minus the ones the nominal model has no
    trouble reconstructing. Same shape as dino_localizer.localize_gated, so the
    two gates are directly comparable - and the boxes stay byte-identical to
    photo's, which is what keeps the verdict cache usable."""
    gate = GATE_THRESHOLD if gate is None else gate
    t0 = time.time()
    regions, info = dl.SHIPPED_LOCALIZE(reference_path, inspection_path)
    support(reference_path, inspection_path, regions, camera)
    kept = [r for r in regions if r["dinomaly_" + stat] > gate]
    info["gated_away"] = len(regions) - len(kept)
    info["gate"] = gate
    info["seconds"] = round(time.time() - t0, 2)
    return kept, info


# --- CLI probe ---------------------------------------------------------------

def _stats(camera):
    """Score distribution over the benchmark cases - how the floor is chosen.
    Prints p99 and max per case, clean cases first: a usable floor sits above
    every clean p99 and below the max of every anomalous frame."""
    from arsi_core import benchmarks
    _, gt = benchmarks.load()
    rows = []
    for c in sorted(gt["cases"], key=lambda c: c["has_anomaly"]):
        if "1762" not in c["image"] and c["source"] != "gpt":
            continue                      # variant scene: another camera, no model
        smap, valid, _ = score_map(str(REPO_ROOT / c["image"]), camera)
        d = smap[valid]
        rows.append((c["id"], c["has_anomaly"], float(np.percentile(d, 99)),
                     float(d.max()), float(np.median(d))))
    print(f"{'case':24s} {'anom':5s} {'median':>8s} {'p99':>8s} {'max':>8s}")
    for cid, anom, p99, mx, med in rows:
        print(f"{cid:24s} {int(anom):<5d} {med:8.3f} {p99:8.3f} {mx:8.3f}")
    clean_p99 = max(r[2] for r in rows if not r[1])
    anom_max = min(r[3] for r in rows if r[1])
    print(f"\nhighest clean p99: {clean_p99:.3f} | lowest anomalous max: {anom_max:.3f}")


def heatmap(inspection_path: str, out: str, camera: str = CAMERA):
    """The frame with its reconstruction error painted over it. The first thing
    to look at when a threshold behaves oddly: it separates "the model does not
    know this scene" from "the threshold is wrong"."""
    smap, valid, _ = score_map(inspection_path, camera)
    img = Image.open(inspection_path).convert("RGB")
    hi = max(1e-6, float(np.percentile(smap[valid], 99.5)) if valid.any() else 1.0)
    heat = np.clip(smap / hi, 0, 1)
    heat = np.asarray(Image.fromarray((heat * 255).astype(np.uint8))
                      .resize(img.size, Image.BILINEAR), dtype=np.float32) / 255.0
    base = np.asarray(img, dtype=np.float32)
    red = np.stack([np.full_like(heat, 255), heat * 60, heat * 60], -1)
    blend = base * (1 - heat[..., None] * 0.75) + red * (heat[..., None] * 0.75)
    Image.fromarray(blend.astype(np.uint8)).save(out)
    print(f"wrote {out}  (p99.5 = {hi:.3f}, max = {smap.max():.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", nargs="?")
    ap.add_argument("inspection", nargs="?")
    ap.add_argument("--camera", default=CAMERA)
    ap.add_argument("--z", type=float, default=Z_THRESHOLD)
    ap.add_argument("--floor", type=float, default=ABS_FLOOR)
    ap.add_argument("--out", default="")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--heat", default="", help="write a score heatmap of INSPECTION")
    args = ap.parse_args()

    if args.stats:
        _stats(args.camera)
        return
    if args.heat:
        heatmap(args.inspection or args.reference, args.heat, args.camera)
        return
    regions, info = localize(args.reference, args.inspection, args.z, args.floor,
                             args.camera)
    print(info)
    for r in regions:
        print(" ", r["bbox"], round(r["salience"], 2))
    if args.out:
        img = Image.open(args.inspection).convert("RGB")
        d = ImageDraw.Draw(img)
        for r in regions:
            d.rectangle(r["bbox"], outline=(255, 60, 60), width=4)
        img.save(args.out)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
