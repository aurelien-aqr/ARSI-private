"""How many pixels of offset can a mask absorb before it becomes wrong?

Each 3333 mask is artificially shifted by k pixels (8 directions) and compared
with the original: IoU, plus the glass area left visible. The curve gives a
numeric mounting tolerance instead of an arbitrary figure.

Output: docs/camera_alignment/_tolerance.jpg + tolerance.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MASKS = ROOT / "data" / "masks_labelme" / "3333"
OUT = ROOT / "docs" / "camera_alignment"
HFOV_DEG = 100.0

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_alignment_report import mask_from_json, deg_from_px, label  # noqa: E402

SHIFTS = [0, 5, 10, 15, 20, 30, 40, 60, 80, 100, 140]
DIRS = [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]


def main():
    W, H = 1280, 720
    masks = [mask_from_json(p, W, H) for p in sorted(MASKS.glob("*.json"))]
    curve = []
    for k in SHIFTS:
        ious, exposed = [], []
        for m in masks:
            for dx, dy in DIRS:
                M = np.float32([[1, 0, k * dx], [0, 1, k * dy]])
                s = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_NEAREST)
                inter = np.logical_and(s > 0, m > 0).sum()
                union = np.logical_or(s > 0, m > 0).sum()
                ious.append(inter / union if union else 1.0)
                exposed.append(float(np.logical_and(m > 0, s == 0).mean()) * 100)
        curve.append({
            "shift_px": k,
            "deg": round(deg_from_px(k, W, HFOV_DEG), 2),
            "iou_median": round(float(np.median(ious)), 3),
            "iou_p10": round(float(np.percentile(ious, 10)), 3),
            "glass_exposed_pct_median": round(float(np.median(exposed)), 2),
            "glass_exposed_pct_p90": round(float(np.percentile(exposed, 90)), 2),
        })
        print(curve[-1])

    (OUT / "tolerance.json").write_text(json.dumps(curve, indent=2))

    # --- plot -------------------------------------------------------------- #
    Wc, Hc, pad = 1100, 620, 90
    fig = np.full((Hc, Wc, 3), 20, np.uint8)
    x_of = lambda k: int(pad + (k / SHIFTS[-1]) * (Wc - 2 * pad))
    y_of = lambda v: int(Hc - pad - v * (Hc - 2 * pad))

    for v in np.arange(0, 1.01, 0.2):
        y = y_of(v)
        cv2.line(fig, (pad, y), (Wc - pad, y), (48, 48, 48), 1)
        label(fig, f"{v:.1f}", (pad - 46, y + 5), (120, 120, 120), 0.45)
    for k in SHIFTS:
        cv2.line(fig, (x_of(k), pad), (x_of(k), Hc - pad), (34, 34, 34), 1)
    for k in (0, 20, 40, 60, 80, 100, 140):
        label(fig, f"{k}", (x_of(k) - 10, Hc - pad + 24), (120, 120, 120), 0.45)
        label(fig, f"{deg_from_px(k, W, HFOV_DEG):.1f}d", (x_of(k) - 14, Hc - pad + 44), (85, 85, 85), 0.4)

    for key, col, txt in (("iou_median", (120, 220, 140), "median IoU"),
                          ("iou_p10", (90, 150, 245), "10th percentile IoU (worst case)")):
        pts = [(x_of(c["shift_px"]), y_of(c[key])) for c in curve]
        for p, q in zip(pts, pts[1:]):
            cv2.line(fig, p, q, col, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(fig, p, 3, col, -1)
        label(fig, txt, (x_of(60), pts[SHIFTS.index(60)][1] - 14), col, 0.5)

    # two mounting marks, read straight off the median curve
    by_shift = {c["shift_px"]: c for c in curve}
    tol_target, tol_limit = 5, 10
    for px, col, txt in ((tol_target, (120, 220, 140), "target"), (tol_limit, (245, 170, 60), "limit")):
        cv2.line(fig, (x_of(px), pad), (x_of(px), Hc - pad), col, 2)
        label(fig, f"{txt} {px} px ({deg_from_px(px, W, HFOV_DEG):.1f} deg) -> IoU {by_shift[px]['iou_median']:.2f}",
              (x_of(px) + 8, pad + 26 + (0 if txt == "target" else 24)), col, 0.5)
    # offset actually observed between 1760 and 3333, for scale
    cv2.line(fig, (x_of(70), pad), (x_of(70), Hc - pad), (90, 90, 230), 1)
    label(fig, "median observed 1760 -> 3333 : 71 px", (x_of(70) + 8, Hc - pad - 20), (110, 110, 235), 0.5)

    label(fig, "Mounting tolerance : IoU of a mask shifted by k pixels against the original",
          (pad, 40), (235, 235, 235), 0.6)
    label(fig, "15 3333 masks x 8 shift directions", (pad, 64), (140, 140, 140), 0.48)
    label(fig, "applied shift (pixels, and degrees at an assumed 100 deg field)",
          (pad, Hc - 20), (140, 140, 140), 0.48)
    cv2.imwrite(str(OUT / "_tolerance.jpg"), fig, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"target {tol_target} px, limit {tol_limit} px")


if __name__ == "__main__":
    main()
