"""Video -> frames extraction (docs/SPEC.md "Video -> frames")."""
import json
from pathlib import Path

import cv2

from .errors import FrameError


def probe(video_path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FrameError(f"cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {"path": str(video_path), "fps": fps, "frame_count": n,
                "duration_s": n / fps if fps else 0.0,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)}
    finally:
        cap.release()


def extract_frames(video_path, out_dir, every_n: int = None, every_s: float = None,
                   start_s: float = 0.0, end_s: float = None,
                   max_side: int = None, ext: str = "jpg") -> dict:
    """Write every selected frame as f%04d.<ext> under out_dir + a meta.json.

    Exactly one of every_n (keep 1 frame in N) / every_s (1 frame every N
    seconds) must be given. start_s/end_s trim; max_side downsizes keeping
    aspect. Returns the meta dict (also saved to out_dir/meta.json).
    """
    if (every_n is None) == (every_s is None):
        raise ValueError("give exactly one of every_n / every_s")
    info = probe(video_path)
    fps = info["fps"] or 25.0
    step = int(every_n) if every_n else max(1, round(float(every_s) * fps))
    first = int(start_s * fps)
    last = int(end_s * fps) if end_s is not None else (info["frame_count"] or 1 << 30) - 1

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    frames, idx = [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or idx > last:
                break
            if idx >= first and (idx - first) % step == 0:
                if max_side and max(frame.shape[:2]) > max_side:
                    scale = max_side / max(frame.shape[:2])
                    frame = cv2.resize(frame, (round(frame.shape[1] * scale),
                                               round(frame.shape[0] * scale)))
                name = f"f{idx:04d}.{ext}"
                cv2.imwrite(str(out_dir / name), frame)
                frames.append({"file": name, "index": idx, "time_s": idx / fps})
            idx += 1
    finally:
        cap.release()

    meta = {"video": info, "params": {"every_n": every_n, "every_s": every_s,
                                      "start_s": start_s, "end_s": end_s,
                                      "max_side": max_side},
            "frames": frames}
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return meta
