"""Static background per camera, by temporal median, for both 1760 and 3333.

A median background removes passengers, moving shadows and noise: only the
structure of the tram survives, which makes cross-tram matching far more
reliable than a single frame does.

Output: docs/camera_alignment/bg/{1760,3333}/<cam>.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VID = ROOT / "data" / "videos"
OUT = ROOT / "docs" / "camera_alignment" / "bg"

N_SAMPLES = 80


def median_background(path: Path, n: int = N_SAMPLES) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    idx = np.linspace(0, max(total - 2, 0), min(n, max(total // 2, 1))).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def main():
    (OUT / "1760").mkdir(parents=True, exist_ok=True)
    (OUT / "3333").mkdir(parents=True, exist_ok=True)

    for p in sorted((VID / "1760").glob("*.mp4")):
        dst = OUT / "1760" / f"{p.stem}.png"
        if dst.exists():
            continue
        bg = median_background(p)
        if bg is not None:
            cv2.imwrite(str(dst), bg)
            print("1760", p.stem, bg.shape)

    # for 3333, take the longest moment: more frames, better median
    by_cam: dict[str, Path] = {}
    for m in sorted((VID / "3333").iterdir()):
        for p in m.glob("*.mp4"):
            best = by_cam.get(p.stem)
            if best is None or p.stat().st_size > best.stat().st_size:
                by_cam[p.stem] = p
    for cam, p in sorted(by_cam.items()):
        dst = OUT / "3333" / f"{cam}.png"
        if dst.exists():
            continue
        bg = median_background(p)
        if bg is not None:
            cv2.imwrite(str(dst), bg)
            print("3333", cam, p.parent.name, bg.shape)


if __name__ == "__main__":
    sys.exit(main())
