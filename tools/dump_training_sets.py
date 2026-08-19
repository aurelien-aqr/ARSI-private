#!/usr/bin/env python3
"""Dump what each Dinomaly checkpoint was actually trained on.

Written to docs/dino_models/training_sets.json and read by
tools/build_dino_doc.py, so the report's training table comes from the
checkpoints themselves rather than from a number typed by hand. Every model
stores its own frame list at save time (tools/dinomaly_train.py), which is what
makes this possible.

    venv/bin/python tools/dump_training_sets.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT = REPO_ROOT / "docs" / "dino_models" / "training_sets.json"

#: How to read a session out of a training frame's filename, per family.
SESSION_RE = (re.compile(r"tram_\d+_(v\d+)_"), re.compile(r"_(\d\d-\d\d-\d\d)_"))


def session_of(path: str) -> str:
    name = path.split("/")[-1]
    for rx in SESSION_RE:
        mt = rx.search(name)
        if mt:
            return mt.group(1)
    return "single run"


def main():
    import tools.dinomaly as dm
    rows = []
    for ckpt in sorted(dm.WEIGHTS_DIR.glob("dinomaly_*.pt")):
        camera = ckpt.stem[len("dinomaly_"):]
        _, meta = dm.load(camera)
        files = meta.get("frames") or []
        if not isinstance(files, list):
            continue
        sessions = Counter(session_of(f) for f in files)
        rows.append({
            "camera": camera,
            "frames": len(files),
            "sessions": sorted(sessions.items()),
            "n_sessions": len(sessions),
            "epochs": meta.get("epochs"),
            "loss_first": meta.get("loss_first"),
            "loss_last": meta.get("loss_last"),
            "size_mb": round(ckpt.stat().st_size / 1048576, 1),
        })
    rows.sort(key=lambda r: r["camera"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    total = sum(r["frames"] for r in rows)
    print(f"wrote {OUT}  ({len(rows)} models, {total} frames)\n")
    for r in rows:
        br = ", ".join(f"{k}={v}" for k, v in r["sessions"])
        print(f"  {r['camera']:<12} {r['frames']:>4} frames  "
              f"{r['n_sessions']} session(s)  [{br}]")


if __name__ == "__main__":
    main()
