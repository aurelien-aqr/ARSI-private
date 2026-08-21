"""Measure the framing offset between two trams filmed by the same cameras.

Measurement chain
-----------------
1. Median background per camera (tools/build_background_frames.py): removes
   passengers, moving shadows and noise, leaving only the tram structure.
2. SIFT matching between the 1760 background and the 3333 background of the
   counterpart camera, then robust estimation of three models (similarity,
   affine, homography) plus an independent phase correlation.
3. Every model is scored by the NCC of the gradient maps after registration;
   the best one is kept, and the disagreement between competing models serves
   as an error bar.
4. Physical reading: translation at the centre, in-plane rotation, scale, then
   conversion to pan/tilt degrees under an explicit field-of-view assumption.

Output: docs/camera_alignment/metrics.json + figures *.jpg
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MASKS = ROOT / "data" / "masks_labelme"
OUT = ROOT / "docs" / "camera_alignment"
BG = OUT / "bg"

# Assumed horizontal field of view of the lens (wide-angle dome, ~2.8 mm on
# 1/2.8"). Pixels are the measurement; degrees are derived from this figure.
HFOV_DEG = 100.0
HFOV_LOW, HFOV_HIGH = 90.0, 110.0

# 1760 camera <-> 3333 camera occupying the same physical position in the tram.
PAIRS = [(f"1760-cam{a:02d}", f"3333-cam{b}") for a, b in zip(range(1, 9), range(10, 18))]
PAIRS += [(f"1760-cam{a:02d}", f"3333-cam{b}") for a, b in zip(range(11, 18), range(50, 57))]


# --------------------------------------------------------------------------- #
# image preparation
# --------------------------------------------------------------------------- #
def _gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(3.0, (8, 8)).apply(g)


def _grad(img: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(_gray(img).astype(np.float32), (0, 0), 2.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    m = np.sqrt(gx * gx + gy * gy)
    return m / (m.max() + 1e-6)


def _to_h(W: np.ndarray) -> np.ndarray:
    return W if W.shape[0] == 3 else np.vstack([W, [0, 0, 1]])


def ncc_score(img_a: np.ndarray, img_b: np.ndarray, W: np.ndarray) -> float:
    """Gradient-map correlation once a has been registered onto b."""
    h, w = img_b.shape[:2]
    wa = cv2.warpPerspective(_grad(img_a), _to_h(W), (w, h), borderValue=float("nan"))
    ok = np.isfinite(wa)
    ok[:20], ok[-20:], ok[:, :20], ok[:, -20:] = False, False, False, False
    if ok.sum() < 0.4 * h * w:
        return -1.0
    x, y = wa[ok], _grad(img_b)[ok]
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))


# --------------------------------------------------------------------------- #
# estimation
# --------------------------------------------------------------------------- #
def estimate(img_a: np.ndarray, img_b: np.ndarray) -> dict:
    """Return the candidate 1760 -> 3333 models and their correspondences."""
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.01)
    ka, da = sift.detectAndCompute(_gray(img_a), None)
    kb, db = sift.detectAndCompute(_gray(img_b), None)
    cands: dict[str, np.ndarray] = {"identity": np.eye(3, dtype=np.float64)}
    src = dst = None
    n_good = 0

    if da is not None and db is not None:
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
        good = [p for p, q in pairs if p.distance < 0.8 * q.distance]
        n_good = len(good)
        if n_good >= 8:
            src = np.float32([ka[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
            dst = np.float32([kb[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
            kw = dict(method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=50000, confidence=0.9999)
            S, si = cv2.estimateAffinePartial2D(src, dst, **kw)
            A, ai = cv2.estimateAffine2D(src, dst, **kw)
            H, hi = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 4.0, maxIters=50000, confidence=0.9999)
            if S is not None:
                cands["similarity"] = _to_h(S)
            if A is not None:
                cands["affine"] = _to_h(A)
            if H is not None:
                cands["homography"] = H

    # phase correlation: pure translation, fully independent of SIFT
    ga, gb = _grad(img_a), _grad(img_b)
    win = cv2.createHanningWindow(ga.shape[::-1], cv2.CV_32F)
    (px, py), presp = cv2.phaseCorrelate(ga * win, gb * win)
    cands["phase"] = np.array([[1, 0, px], [0, 1, py], [0, 0, 1]], np.float64)

    scores = {k: ncc_score(img_a, img_b, W) for k, W in cands.items()}
    return {"cands": cands, "scores": scores, "src": src, "dst": dst,
            "n_matches": n_good, "phase_resp": float(presp)}


def read_similarity(W: np.ndarray, w: int, h: int):
    """dx, dy at the centre + in-plane rotation + scale, read off a 3x3 model."""
    c = W @ np.array([w / 2.0, h / 2.0, 1.0])
    c = c / c[2]
    dx, dy = float(c[0] - w / 2.0), float(c[1] - h / 2.0)
    a, b_ = W[0, 0], W[0, 1]
    scale = math.hypot(a, b_)
    rot = math.degrees(math.atan2(-b_, a))
    return dx, dy, rot, scale


def deg_from_px(shift_px: float, w: int, hfov: float) -> float:
    """Camera rotation equivalent to a shift measured at the image centre."""
    f = (w / 2.0) / math.tan(math.radians(hfov / 2.0))
    return math.degrees(math.atan2(shift_px, f))


def analyse_pair(name_a: str, name_b: str) -> dict:
    img_a = cv2.imread(str(BG / "1760" / f"{name_a}.png"))
    img_b = cv2.imread(str(BG / "3333" / f"{name_b}.png"))
    h, w = img_a.shape[:2]
    est = estimate(img_a, img_b)
    scores = est["scores"]

    best_name = max(scores, key=scores.get)
    best = est["cands"][best_name]
    best_score = scores[best_name]

    # "readable" model: the best similarity, or failing that the best model
    read_name = "similarity" if scores.get("similarity", -1) > 0.9 * best_score else best_name
    dx, dy, rot, scale = read_similarity(est["cands"][read_name], w, h)

    # uncertainty: spread between competing models (score > 80 % of the best)
    rivals = [k for k, s in scores.items() if k != "identity" and s > 0.8 * best_score and s > 0.15]
    d_all = np.array([read_similarity(est["cands"][k], w, h)[:2] for k in rivals]) if rivals else np.zeros((1, 2))
    spread = float(np.linalg.norm(d_all.std(axis=0))) if len(d_all) > 1 else 0.0

    # parallax: residual of the matched points under the best model. A pure
    # camera pivot registers exactly; a displaced mount leaves a residual that
    # depends on depth.
    resid = flow_med = flow_disp = None
    if est["src"] is not None:
        proj = cv2.perspectiveTransform(est["src"], best).reshape(-1, 2)
        d = np.linalg.norm(proj - est["dst"].reshape(-1, 2), axis=1)
        inl = d < 6.0
        n_inl = int(inl.sum())
        if n_inl >= 8:
            resid = float(np.median(d[inl]))
            mags = np.linalg.norm(est["dst"].reshape(-1, 2)[inl] - est["src"].reshape(-1, 2)[inl], axis=1)
            flow_med = float(np.median(mags))
            # relative flow spread: ~0 = pure pivot, >0.3 = the mount moved
            flow_disp = float((np.percentile(mags, 90) - np.percentile(mags, 10)) / max(flow_med, 1e-6))
    else:
        n_inl = 0

    # overlap of the common field of view
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, best).reshape(-1, 2).astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(warped, corners.reshape(-1, 2))
    overlap = float(inter) / (w * h) if inter else 0.0

    shift = math.hypot(dx, dy)
    return {
        "cam_1760": name_a, "cam_3333": name_b, "size": [w, h],
        "model": read_name, "best_model": best_name,
        "ncc": round(best_score, 3), "ncc_identity": round(scores["identity"], 3),
        "n_matches": est["n_matches"], "n_inliers": n_inl,
        "phase_resp": round(est["phase_resp"], 3),
        "dx": round(dx, 1), "dy": round(dy, 1), "shift_px": round(shift, 1),
        "shift_pct_w": round(100 * shift / w, 1),
        "rot_deg": round(rot, 2), "scale": round(scale, 4),
        "spread_px": round(spread, 1),
        "pan_deg": round(deg_from_px(dx, w, HFOV_DEG), 2),
        "tilt_deg": round(deg_from_px(dy, w, HFOV_DEG), 2),
        "pan_deg_lo": round(deg_from_px(dx, w, HFOV_LOW), 2),
        "pan_deg_hi": round(deg_from_px(dx, w, HFOV_HIGH), 2),
        "tilt_deg_lo": round(deg_from_px(dy, w, HFOV_LOW), 2),
        "tilt_deg_hi": round(deg_from_px(dy, w, HFOV_HIGH), 2),
        "total_deg": round(deg_from_px(shift, w, HFOV_DEG), 2),
        "parallax_resid_px": round(resid, 2) if resid else None,
        "flow_median_px": round(flow_med, 1) if flow_med else None,
        "flow_dispersion": round(flow_disp, 3) if flow_disp is not None else None,
        "fov_overlap": round(overlap, 3),
        "_H": best.tolist(),
        "_img": (img_a, img_b),
        "_est": est,
    }


# --------------------------------------------------------------------------- #
# labelme masks
# --------------------------------------------------------------------------- #
def mask_from_json(path: Path, w: int, h: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    if not path.exists():
        return m
    for shape in json.loads(path.read_text())["shapes"]:
        pts = np.array(shape["points"], np.float32)
        if shape["shape_type"] == "circle":
            c, edge = pts[0], pts[1]
            cv2.circle(m, tuple(np.round(c).astype(int)), int(round(np.linalg.norm(edge - c))), 255, -1)
        elif shape["shape_type"] == "rectangle":
            cv2.rectangle(m, tuple(np.round(pts[0]).astype(int)), tuple(np.round(pts[1]).astype(int)), 255, -1)
        else:
            cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 255)
    return m


def mask_stats(name_a: str, name_b: str, H, w: int, h: int):
    ma = mask_from_json(MASKS / "1760" / f"{name_a}.json", w, h)
    mb = mask_from_json(MASKS / "3333" / f"{name_b}.json", w, h)
    mw = cv2.warpPerspective(ma, np.array(H), (w, h), flags=cv2.INTER_NEAREST)

    def iou(x, y):
        u = np.logical_or(x > 0, y > 0).sum()
        return round(float(np.logical_and(x > 0, y > 0).sum()) / u, 3) if u else None

    n_a = len(json.loads((MASKS / "1760" / f"{name_a}.json").read_text())["shapes"])
    n_b = len(json.loads((MASKS / "3333" / f"{name_b}.json").read_text())["shapes"])
    out = {
        "n_shapes_1760": n_a, "n_shapes_3333": n_b,
        "area_1760_pct": round(float((ma > 0).mean()) * 100, 1),
        "area_3333_pct": round(float((mb > 0).mean()) * 100, 1),
        "iou_copied": iou(ma, mb),
        "iou_registered": iou(mw, mb),
        # under-coverage: glass that should have been masked and stays visible
        # -> the outside world scrolls through the inspected area = false alarms
        "glass_exposed_copied_pct": round(float(np.logical_and(mb > 0, ma == 0).mean()) * 100, 2),
        "glass_exposed_registered_pct": round(float(np.logical_and(mb > 0, mw == 0).mean()) * 100, 2),
        # over-coverage: interior masked by mistake -> anomalies made invisible
        "interior_masked_copied_pct": round(float(np.logical_and(ma > 0, mb == 0).mean()) * 100, 2),
        "interior_masked_registered_pct": round(float(np.logical_and(mw > 0, mb == 0).mean()) * 100, 2),
    }
    return out, ma, mb, mw


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def label(img, text, org=(12, 30), color=(255, 255, 255), scale=0.72):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def fig_align(img_a, img_b, r, path: Path):
    h, w = img_a.shape[:2]
    a, b = img_a.copy(), img_b.copy()
    for im, txt, col in ((a, f"{r['cam_1760']}  -  median background 1760", (0, 200, 255)),
                         (b, f"{r['cam_3333']}  -  median background 3333", (0, 255, 120))):
        for x in range(w // 8, w, w // 8):
            cv2.line(im, (x, 0), (x, h), (70, 70, 70), 1)
        for y in range(h // 6, h, h // 6):
            cv2.line(im, (0, y), (w, y), (70, 70, 70), 1)
        cv2.drawMarker(im, (w // 2, h // 2), col, cv2.MARKER_CROSS, 44, 2)
        label(im, txt, color=col)

    ga, gb = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    over = cv2.merge([ga, gb, ga])
    label(over, "raw overlay   magenta = 1760   green = 3333", scale=0.64)
    if abs(r["dx"]) + abs(r["dy"]) > 4:
        cv2.arrowedLine(over, (w // 2, h // 2), (int(w // 2 + r["dx"]), int(h // 2 + r["dy"])),
                        (0, 255, 255), 3, tipLength=0.2)
    label(over, f"dx={r['dx']:+.0f}px  dy={r['dy']:+.0f}px  roll={r['rot_deg']:+.2f}deg  "
                f"scale={r['scale']:.3f}  =>  {r['total_deg']:.1f}deg", (12, h - 18), (0, 255, 255), 0.64)

    gw = cv2.cvtColor(cv2.warpPerspective(img_a, np.array(r["_H"]), (w, h)), cv2.COLOR_BGR2GRAY)
    aligned = cv2.merge([gw, gb, gw])
    label(aligned, f"after alignment ({r['model']})  -  residual", scale=0.64)
    label(aligned, f"NCC {r['ncc_identity']:.2f} -> {r['ncc']:.2f}", (12, h - 18), (255, 255, 255), 0.64)

    cv2.imwrite(str(path), np.vstack([np.hstack([a, b]), np.hstack([over, aligned])]),
                [cv2.IMWRITE_JPEG_QUALITY, 86])


def fig_masks(img_b, ma, mb, mw, r, path: Path):
    h, w = img_b.shape[:2]
    canvas = (img_b * 0.5).astype(np.uint8)

    def draw(mask, color, thick=2):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, color, thick)

    draw(ma, (255, 90, 255))
    draw(mw, (0, 210, 255))
    draw(mb, (0, 255, 120))
    label(canvas, "magenta : 1760 mask copied as-is", (12, 28), (255, 90, 255), 0.6)
    label(canvas, "yellow  : same mask, auto-aligned", (12, 52), (0, 210, 255), 0.6)
    label(canvas, "green   : 3333 mask redrawn by hand", (12, 76), (0, 255, 120), 0.6)
    label(canvas, f"IoU copied {r['iou_copied']:.2f} -> aligned {r['iou_registered']:.2f}    "
                  f"glass left visible {r['glass_exposed_copied_pct']:.1f} % of frame    "
                  f"interior wrongly masked {r['interior_masked_copied_pct']:.1f} %",
          (12, h - 18), (255, 255, 255), 0.55)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 86])


def fig_flow(img_b, r, path: Path):
    est = r["_est"]
    if est["src"] is None:
        return
    h, w = img_b.shape[:2]
    canvas = cv2.addWeighted(img_b, 0.4, np.zeros_like(img_b), 0, 45)
    src = est["src"].reshape(-1, 2)
    dst = est["dst"].reshape(-1, 2)
    proj = cv2.perspectiveTransform(est["src"], np.array(r["_H"])).reshape(-1, 2)
    keep = np.linalg.norm(proj - dst, axis=1) < 6.0
    src, dst = src[keep], dst[keep]
    mags = np.linalg.norm(dst - src, axis=1)
    vmax = max(float(np.percentile(mags, 95)), 1.0) if len(mags) else 1.0
    for (x0, y0), (x1, y1), m in zip(src, dst, mags):
        t = min(m / vmax, 1.0)
        cv2.arrowedLine(canvas, (int(x0), int(y0)), (int(x1), int(y1)),
                        (int(255 * (1 - t)), int(90 + 110 * (1 - t)), int(255 * t)), 1, tipLength=0.3)
    label(canvas, f"displacement of the {len(src)} matched structural points (1760 -> 3333)",
          (12, 28), scale=0.6)
    if len(mags):
        label(canvas, f"median {np.median(mags):.0f} px   min {mags.min():.0f}   max {mags.max():.0f} px"
                      f"   -> uniform flow = pivot, varying flow = mount displaced",
              (12, 52), scale=0.55)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 86])


def fig_summary(rows: list[dict], path: Path):
    """Offset map: one arrow per camera, in image coordinates."""
    S, pad = 900, 70
    canvas = np.full((S + 2 * pad, S + 2 * pad, 3), 22, np.uint8)
    cx = cy = pad + S // 2
    lim = 180.0                      # px of offset spanned by a half-axis
    k = (S / 2) / lim

    # each ring is labelled at whatever bearing sits farthest from every arrow
    bearings = [math.atan2(r["dy"], r["dx"]) for r in rows]

    def clearest_angle() -> float:
        def gap(a):
            return min(abs(math.atan2(math.sin(a - b), math.cos(a - b))) for b in bearings)
        return max((math.radians(t) for t in range(0, 360, 5)), key=gap)

    for rpx in (50, 100, 150):
        cv2.circle(canvas, (cx, cy), int(rpx * k), (55, 55, 55), 1)
        a = clearest_angle()
        gx = int(cx + rpx * k * math.cos(a))
        gy = int(cy + rpx * k * math.sin(a))
        label(canvas, f"{rpx} px", (gx - 20, gy + 18), (105, 105, 105), 0.44)
    cv2.line(canvas, (pad, cy), (pad + S, cy), (80, 80, 80), 1)
    cv2.line(canvas, (cx, pad), (cx, pad + S), (80, 80, 80), 1)

    # labels are pushed along the radius until they no longer overlap
    placed: list[tuple[int, int]] = []
    for r in sorted(rows, key=lambda r: -r["shift_px"]):
        x = int(cx + r["dx"] * k)
        y = int(cy + r["dy"] * k)
        big = r["shift_px"] > 100
        col = (80, 90, 245) if big else (120, 220, 120)
        cv2.arrowedLine(canvas, (cx, cy), (x, y), col, 2, tipLength=0.12)
        cv2.circle(canvas, (x, y), 4, col, -1)

        norm = max(math.hypot(r["dx"], r["dy"]), 1e-6)
        ux, uy = r["dx"] / norm, r["dy"] / norm
        lx, ly = x + int(10 * ux) + 6, y + int(10 * uy) + 5
        for _ in range(24):
            if all(abs(lx - px) > 66 or abs(ly - py) > 17 for px, py in placed):
                break
            lx, ly = lx + int(14 * ux), ly + int(14 * uy)
        placed.append((lx, ly))
        cv2.line(canvas, (x, y), (lx - 4, ly - 4), (70, 70, 70), 1)
        label(canvas, r["cam_3333"].replace("3333-", ""), (lx, ly), col, 0.5)

    label(canvas, "Framing offset 1760 -> 3333, measured at the image centre",
          (pad, 34), (235, 235, 235), 0.66)
    label(canvas, "->  right = the 3333 scene is shifted right    "
                  "down = shifted down", (pad, 58), (150, 150, 150), 0.5)
    label(canvas, f"angular scale : 100 px ~ {deg_from_px(100, 1280, HFOV_DEG):.1f} deg "
                  f"(assumed {HFOV_DEG:.0f} deg field)", (pad, S + 2 * pad - 24), (150, 150, 150), 0.5)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])


def fig_contact_sheet(path: Path, rows: list[dict]):
    """Contact sheet: raw overlay of all 15 pairs, at a glance."""
    tw, th = 426, 240
    cols, rws = 5, 3
    sheet = np.full((rws * (th + 26), cols * tw, 3), 20, np.uint8)
    for i, r in enumerate(rows):
        a = cv2.imread(str(BG / "1760" / f"{r['cam_1760']}.png"))
        b = cv2.imread(str(BG / "3333" / f"{r['cam_3333']}.png"))
        ga = cv2.cvtColor(cv2.resize(a, (tw, th)), cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(cv2.resize(b, (tw, th)), cv2.COLOR_BGR2GRAY)
        tile = cv2.merge([ga, gb, ga])
        r_, c_ = divmod(i, cols)
        y0 = r_ * (th + 26)
        sheet[y0:y0 + th, c_ * tw:(c_ + 1) * tw] = tile
        label(sheet, f"{r['cam_1760']} / {r['cam_3333']}   {r['shift_px']:.0f} px  "
                     f"{r['total_deg']:.1f} deg  roll {r['rot_deg']:+.1f}",
              (c_ * tw + 6, y0 + th + 18), (220, 220, 220), 0.42)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name_a, name_b in PAIRS:
        r = analyse_pair(name_a, name_b)
        img_a, img_b = r["_img"]
        h, w = img_a.shape[:2]
        ms, ma, mb, mw = mask_stats(name_a, name_b, r["_H"], w, h)
        r.update(ms)

        fig_align(img_a, img_b, r, OUT / f"{name_b}_align.jpg")
        fig_masks(img_b, ma, mb, mw, r, OUT / f"{name_b}_masks.jpg")
        fig_flow(img_b, r, OUT / f"{name_b}_flow.jpg")

        for k in ("_img", "_est", "_H"):
            r.pop(k, None)
        rows.append(r)
        print(f"{name_a}->{name_b}: {r['model']:11s} dx={r['dx']:+7.1f} dy={r['dy']:+7.1f} "
              f"({r['shift_px']:5.1f}px = {r['total_deg']:4.1f}deg) roll={r['rot_deg']:+5.2f} "
              f"scale={r['scale']:.3f} NCC {r['ncc_identity']:.2f}->{r['ncc']:.2f} "
              f"±{r['spread_px']:.0f}px  IoU {r['iou_copied']:.2f}->{r['iou_registered']:.2f}")

    fig_summary(rows, OUT / "_offset_summary.jpg")
    fig_contact_sheet(OUT / "_contact_sheet.jpg", rows)
    (OUT / "metrics.json").write_text(json.dumps(rows, indent=2))
    print(f"\n-> {OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
