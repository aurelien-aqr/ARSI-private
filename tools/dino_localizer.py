#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - tools/dino_localizer.py
#  AnomalyDINO-style change localizer: a DROP-IN alternative to the photometric
#  diff of vlm_05_reference_diff.localize(), same signature, same return shape.
#
#  WHY (measured, see benchmark/README + project notes):
#    the shipped photometric diff has localization recall 45/45 but its noise is
#    entirely photometric - a cross-session empty frame yields 15-37 candidate
#    regions purely from lighting / white balance / seat state, and the faint
#    graffiti FN had an in-box diff amplitude of ~9 against DIFF_THRESHOLD 40.
#    Both are failures of comparing PIXELS. This module compares DINOv2 patch
#    FEATURES instead, which are semantic and largely invariant to exposure.
#
#  HOW it differs from the AnomalyDINO paper (arXiv 2405.14529, WACV 2025):
#    the paper builds ONE memory bank of all nominal patches (MVTec objects are
#    centred but not aligned). Our camera is FIXED, so we can do better: a patch
#    is compared only to nominal patches in a small neighbourhood window at the
#    SAME grid position. That is a much tighter null hypothesis - a seat cushion
#    is compared to that seat, not to any seat in the image.
#
#  Scoring is a robust z-score of the cosine distance map (median / MAD over the
#  valid patches), which makes the threshold scene-independent - the constraint
#  the 2026-07-12 probe identified ("the fix must be local-adaptive change
#  detection, not prompt work").
#
#  Everything AFTER the score map is deliberately the SHIPPED machinery, reused
#  by import: find_regions, the YOLOv8n person veto, the MAX_REGIONS salience
#  cap, merge_regions. So an A/B against "shipped" isolates the change-proposal
#  signal and nothing else.
#
#  Usage:
#      import tools.dino_localizer as dl
#      regions, info = dl.localize(ref_path, insp_path)          # z >= Z_THRESHOLD
#      regions, info = dl.localize(ref_path, insp_path, z_thr=5)  # sweep
#      python tools/dino_localizer.py REF INSP [--z 4] [--out out.jpg]
# =============================================================================

import os, sys, time, argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import vlm_05_reference_diff as m


# --- configuration -----------------------------------------------------------

MODEL_NAME = "dinov2_vits14_reg"  # 22 M params, 1.7 s/frame CPU at the size below
PATCH      = 14
INPUT_W    = int(os.environ.get("DINO_INPUT_W", 1120))
#                                 -> 80x45 grid on a 16:9 frame = 24 px/patch
                                  # (a real phone region is ~3600 px = 2-3 patches)
Z_THRESHOLD = 4.0                 # robust z of the cosine distance; swept below
ABS_FLOOR   = 0.08                # AND an absolute cosine-distance floor.
#  Why both (measured 2026-08-12): the per-frame MAD scale ranges over 5x across
#  cases (0.006 on a same-session pair, 0.030 on a frame with a person in it), so
#  one z threshold means a 0.043 absolute cut on one frame and 0.225 on another.
#  On a genuinely near-identical same-session pair the MAD collapses and the z
#  amplifies pure feature jitter into 80+ regions. The floor kills that (those
#  frames sit at p99 = 0.07) while the z keeps handling the cross-session case,
#  where the whole map shifts up. Real anomaly patches reach 0.37-0.69, and the
#  faintest true instance measured (the phone in real_f0112) peaks at 0.126.
NEIGHBOURHOOD = 1                 # +-1 patch of alignment slack (sub-patch jitter)
BLACK_LEVEL = 12                  # same masked-window cutoff as vlm_05._gray_pair

_model = None
_feat_cache = {}


# --- feature extraction ------------------------------------------------------

def _load_model():
    global _model
    if _model is None:
        import torch
        torch.set_num_threads(max(1, (Path("/proc/cpuinfo").read_text()
                                      .count("processor\t")) or 4))
        _model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME,
                                verbose=False).eval()
    return _model


def _grid_size(size):
    """Input size snapped to a multiple of PATCH, keeping the aspect ratio."""
    w, h = size
    gw = max(1, INPUT_W // PATCH)
    gh = max(1, int(round(h * (gw * PATCH) / w / PATCH)))
    return gw, gh


def features(path: str, size):
    """L2-normalised DINOv2 patch features of `path` resampled onto the grid of
    a frame of `size` (the REFERENCE size - both frames must share one grid).
    Returns (feats[gh,gw,dim], valid[gh,gw]) where valid is False on the blacked
    -out window areas. Cached per (path, grid)."""
    import torch
    gw, gh = _grid_size(size)
    key = (str(path), gw, gh)
    if key in _feat_cache:
        return _feat_cache[key]

    img = Image.open(path).convert("RGB").resize((gw * PATCH, gh * PATCH),
                                                 Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    t = torch.from_numpy(x.transpose(2, 0, 1))[None]

    with torch.no_grad():
        out = _load_model().forward_features(t)
    f = out["x_norm_patchtokens"][0].numpy().reshape(gh, gw, -1)
    f /= (np.linalg.norm(f, axis=2, keepdims=True) + 1e-8)

    # patch-level luminance, to drop the masked windows exactly like vlm_05 does
    lum = np.asarray(Image.open(path).convert("L")
                     .resize((gw, gh), Image.BILINEAR), dtype=np.float32)
    valid = lum >= BLACK_LEVEL

    _feat_cache[key] = (f, valid)
    return _feat_cache[key]


# --- score map ---------------------------------------------------------------

def score_map(reference_path: str, inspection_path: str):
    """Robust z-score map (patch grid) of the cosine distance from each
    inspection patch to its nearest nominal patch within NEIGHBOURHOOD at the
    same grid position. Returns (z[gh,gw], valid[gh,gw], raw_distance)."""
    size = Image.open(reference_path).size
    fa, va = features(reference_path, size)          # reference = nominal
    fb, vb = features(inspection_path, size)
    valid = va & vb

    n = NEIGHBOURHOOD
    gh, gw, _ = fb.shape
    best = np.full((gh, gw), -1.0, dtype=np.float32)
    for dy in range(-n, n + 1):
        for dx in range(-n, n + 1):
            shifted = np.roll(np.roll(fa, dy, axis=0), dx, axis=1)
            best = np.maximum(best, np.einsum("ijk,ijk->ij", fb, shifted))
    dist = 1.0 - best                                # cosine distance, >= 0

    d = dist[valid]
    med = float(np.median(d)) if d.size else 0.0
    mad = float(np.median(np.abs(d - med))) if d.size else 0.0
    scale = 1.4826 * mad if mad > 1e-6 else (float(d.std()) or 1e-6)
    z = (dist - med) / scale
    z[~valid] = 0.0
    return z, valid, dist


# --- localization ------------------------------------------------------------

def _upscale(arr, size):
    """Patch-grid array -> full reference resolution, nearest neighbour."""
    return np.asarray(Image.fromarray(arr.astype(np.float32), mode="F")
                      .resize(size, Image.NEAREST), dtype=np.float32)


def localize(reference_path: str, inspection_path: str, z_thr: float = None,
             abs_floor: float = None):
    """Same contract as vlm_05_reference_diff.localize(): returns (regions, info)
    with regions = [{"bbox": [x0,y0,x1,y1], "area": px, ...}] in REFERENCE pixel
    space. A patch is a candidate when it is BOTH a robust-z outlier and past the
    absolute distance floor (see ABS_FLOOR). Post-processing (person veto,
    salience cap, merge) is vlm_05's, so a diff against the shipped localizer
    isolates the change signal alone."""
    z_thr = Z_THRESHOLD if z_thr is None else z_thr
    abs_floor = ABS_FLOOR if abs_floor is None else abs_floor
    t0 = time.time()
    size = Image.open(reference_path).size
    z, valid, dist = score_map(reference_path, inspection_path)

    over = (z > z_thr) & (dist > abs_floor)
    zmap = _upscale(z, size)
    mask = _upscale(over.astype(np.float32), size) > 0.5
    regions = m.find_regions(mask, m.DOWNSCALE, m.DILATE, m.MIN_AREA, m.MAX_AREA)
    info = {"raw": len(regions), "z_thr": z_thr, "floor": abs_floor,
            "patches_over": int(over.sum()), "person_veto": 0}
    for r in regions:
        r["channel"] = "dino"
        r["salience"] = m.salience(r, zmap)

    persons = m.person_boxes(inspection_path, size)
    info["persons"] = len(persons)
    if persons:
        kept = [r for r in regions
                if not any(m._ioa(r["bbox"], p) >= m.PERSON_IOA for p in persons)]
        info["person_veto"] = len(regions) - len(kept)
        regions = kept

    regions.sort(key=lambda r: -r["salience"])
    info["capped"] = len(regions) > m.MAX_REGIONS
    regions = regions[:m.MAX_REGIONS]

    info["before_merge"] = len(regions)
    if m.MERGE_REGIONS:
        regions = m.merge_regions(regions)
    info["merged_away"] = info["before_merge"] - len(regions)
    info["total"] = len(regions)
    info["seconds"] = round(time.time() - t0, 2)
    return regions, info


# --- feature gate over the SHIPPED localizer ---------------------------------
#  The better trade than replacing the diff. Measured 2026-08-12: the DINO map
#  alone halves the cross-session noise but its boxes are quantised to the 24 px
#  patch grid, which costs strict-IoU (37/45 -> 32-34/45). The photometric diff
#  already has 45/45 recall and tight boxes; what it lacks is a way to tell a
#  lighting shift from a new object. So keep ITS boxes and ask the features for a
#  yes/no on each one.

GATE_THRESHOLD = 0.08
#  Chosen recall-first (measured 2026-08-12, 29 cases, GLM x conservative). 0.08
#  is the largest threshold at which NO ground-truth instance loses its region:
#  region precision 0.730 -> 0.815 and 559 -> 243 VLM calls, object recall and
#  strict IoU both untouched. 0.12 reaches precision 0.855 but drops the far
#  phone of real_f0219 (~1 patch wide) - end-to-end recall does not move only
#  because GLM was already rejecting that instance, so it is a loss waiting to
#  surface under a different judge. Raising the grid resolution does NOT fix it:
#  at DINO_INPUT_W=1400 (19 px/patch) gate 0.12 loses 3 instances instead of 1
#  AND keeps more regions - DINOv2 drifts from its training scale.

#  Bound at import time on purpose: tools/rescore_gate.py monkeypatches
#  m.localize with the gated version to drive run_benchmark, and calling it
#  through the module attribute would recurse.
SHIPPED_LOCALIZE = m.localize


def feature_support(reference_path: str, inspection_path: str, regions,
                    stat: str = "max"):
    """Annotate each region with dino_max / dino_mean = the patch-level cosine
    distance to the nominal reference inside that box. In place; also returned."""
    size = Image.open(reference_path).size
    _, valid, dist = score_map(reference_path, inspection_path)
    gh, gw = dist.shape
    W, H = size
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        j0, j1 = int(x0 * gw / W), max(int(x0 * gw / W) + 1, int(np.ceil(x1 * gw / W)))
        i0, i1 = int(y0 * gh / H), max(int(y0 * gh / H) + 1, int(np.ceil(y1 * gh / H)))
        p = dist[i0:i1, j0:j1]
        v = valid[i0:i1, j0:j1]
        p = p[v] if v.any() else p.ravel()
        r["dino_max"] = float(p.max()) if p.size else 0.0
        r["dino_mean"] = float(p.mean()) if p.size else 0.0
    return regions


def localize_gated(reference_path: str, inspection_path: str,
                   gate: float = None, stat: str = "max"):
    """vlm_05's shipped localizer, with every region that has no DINOv2 feature
    support dropped. Returns (regions, info) with info["gated_away"]."""
    gate = GATE_THRESHOLD if gate is None else gate
    t0 = time.time()
    regions, info = SHIPPED_LOCALIZE(reference_path, inspection_path)
    feature_support(reference_path, inspection_path, regions)
    kept = [r for r in regions if r["dino_" + stat] > gate]
    info["gated_away"] = len(regions) - len(kept)
    info["gate"] = gate
    info["seconds"] = round(time.time() - t0, 2)
    return kept, info


# --- CLI probe ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="DINOv2 change localizer probe")
    ap.add_argument("reference")
    ap.add_argument("inspection")
    ap.add_argument("--z", type=float, default=Z_THRESHOLD)
    ap.add_argument("--out", default="", help="write an annotated jpg here")
    args = ap.parse_args()

    regions, info = localize(args.reference, args.inspection, z_thr=args.z)
    print(f"{len(regions)} regions   {info}")
    for r in regions:
        print(f"  {r['bbox']}  area={r['area']:>7,}  salience={r['salience']:.1f}")

    if args.out:
        img = Image.open(args.inspection).convert("RGB")
        img = img.resize(Image.open(args.reference).size)
        d = ImageDraw.Draw(img)
        for r in regions:
            d.rectangle(r["bbox"], outline=(255, 60, 0), width=4)
        img.save(args.out, quality=90)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
