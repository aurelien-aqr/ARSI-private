#!/usr/bin/env python3
"""One dispatcher from a variant string to a set of proposed regions.

Two very different consumers need the SAME proposal step and must not drift
apart: `benchmark/eval_localization.py` scores boxes against the ground truth
with no VLM, and `tools/rescore_localizer.py` drives the full judge. If each
kept its own copy of "what does `ddgate0.05` mean", an end-to-end A/B could
silently compare two different things - so the meaning of a spec lives here,
once.

Spec grammar (the prefix decides who PROPOSES, which is what the 2026-08-19
comparison found to be the thing that matters):

    PIXEL DIFF PROPOSES
      shipped                 vlm_05's multi-channel localizer, untouched
      gate<thr>[mean]         ... minus regions with no DINOv2 feature support
      dgate<thr>[mean]        ... minus regions the nominal Dinomaly model rebuilds
      bgate<a>+<b>            ... minus both

    FEATURES PROPOSE
      dino<z>[@floor]         AnomalyDINO scores the patches and finds the regions
      dinomaly<z>[@floor]     the reference-free model does
      dpgate<thr>             dino proposes, the photometric diff vetoes
      ddgate<thr>             dino proposes, the Dinomaly model vetoes

Every spec returns vlm_05's own `(regions, info)` contract, with bboxes in
REFERENCE pixel space, and `info["total"]` set - run_benchmark reads that key.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import vlm_05_reference_diff as m                                # noqa: E402

#: BOUND AT IMPORT TIME, and the submodules imported eagerly, both for the same
#: reason: tools/rescore_localizer.py drives run_benchmark by monkeypatching
#: `m.localize` with a call back into propose(). Reading m.localize later would
#: then find the patch and recurse forever, and a lazily imported
#: dino_localizer would capture the patched function in its own
#: SHIPPED_LOCALIZE. Importing here costs a torch import that every caller of
#: this module pays anyway. (dino_localizer.py:252 guards the same hazard.)
SHIPPED_LOCALIZE = m.localize

import tools.dino_localizer as dl                                # noqa: E402
import tools.dinomaly_localizer as dml                           # noqa: E402


def photo_support(ref_path, img_path, regions):
    """Annotate each region with photo_max/photo_mean: the blurred photometric
    difference inside that box, in reference pixel space. In place.

    The counterpart of dino_localizer.feature_support for the other direction -
    features propose and PIXELS veto. One subtract and one blur: no model, no
    per-camera state.
    """
    a, b, black = m._gray_pair(ref_path, img_path)
    _, d = m.photo_data(a, b, black)
    h, w = d.shape
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, max(x0 + 1, int(x1))), min(h, max(y0 + 1, int(y1)))
        p = d[y0:y1, x0:x1]
        r["photo_max"] = float(p.max()) if p.size else 0.0
        r["photo_mean"] = float(p.mean()) if p.size else 0.0
    return regions


def propose(spec: str, ref_path: str, img_path: str, camera: str = "tram_1762"):
    """(regions, info) for one variant spec. `camera` selects the per-camera
    Dinomaly checkpoint and is ignored by the specs that need no model."""
    spec = spec.strip()

    if spec == "shipped" or spec == "photo":
        regions, info = SHIPPED_LOCALIZE(ref_path, img_path)

    elif spec.startswith("ddgate") or spec.startswith("dpgate"):
        regions, info = dl.localize(ref_path, img_path)
        n0 = len(regions)
        thr = float(spec[6:])
        if spec.startswith("ddgate"):
            dml.support(ref_path, img_path, regions, camera=camera)
            regions = [r for r in regions if r["dinomaly_max"] > thr]
        else:
            photo_support(ref_path, img_path, regions)
            regions = [r for r in regions if r["photo_max"] > thr]
        info["vetoed"] = n0 - len(regions)

    elif spec.startswith("dinomaly"):
        zs, _, fs = spec[len("dinomaly"):].partition("@")
        regions, info = dml.localize(ref_path, img_path,
                                     z_thr=float(zs) if zs else None,
                                     abs_floor=float(fs) if fs else None,
                                     camera=camera)

    elif spec.startswith("bgate"):
        a, _, b = spec[5:].partition("+")
        regions, info = dl.localize_gated(ref_path, img_path, gate=float(a))
        dml.support(ref_path, img_path, regions, camera=camera)
        regions = [r for r in regions if r["dinomaly_max"] > float(b)]

    elif spec.startswith("dgate"):
        body = spec[5:]
        stat = "mean" if body.endswith("mean") else "max"
        thr = float(body.replace("mean", "") or dml.GATE_THRESHOLD)
        regions, info = dml.localize_gated(ref_path, img_path, gate=thr,
                                           stat=stat, camera=camera)

    elif spec.startswith("gate"):
        body = spec[4:]
        stat = "mean" if body.endswith("mean") else "max"
        thr = float(body.replace("mean", "") or dl.GATE_THRESHOLD)
        regions, info = dl.localize_gated(ref_path, img_path, gate=thr, stat=stat)

    elif spec.startswith("dino"):
        zs, _, fs = spec[4:].partition("@")
        regions, info = dl.localize(ref_path, img_path,
                                    z_thr=float(zs) if zs else dl.Z_THRESHOLD,
                                    abs_floor=float(fs) if fs else dl.ABS_FLOOR)

    else:
        raise ValueError(f"unknown localizer spec '{spec}'")

    info["spec"] = spec
    info["total"] = len(regions)
    return regions, info


def camera_of(case: dict) -> str:
    """Which per-camera Dinomaly checkpoint a case belongs to.

    The benchmark's reference key IS the camera for the fleet cameras
    ("1760-cam04", "3333-cam53"), so nothing is maintained in parallel. "real" is
    the tram_1762 camera; "variant" is a different scene with no model of its
    own and is deliberately scored against the tram_1762 checkpoint, because
    out-of-domain behaviour is one of the things the benchmark measures.
    """
    ref = case.get("reference", "")
    return ref if "-cam" in ref else "tram_1762"
