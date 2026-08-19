#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - tools/dinomaly_train.py
#  Trains the Dinomaly decoder on this camera's NOMINAL frames.
#
#  The whole value of the reference-free direction is that the model sees many
#  sessions of "nothing wrong" instead of one reference frame. That makes the
#  training set the experiment, so it is built here explicitly and defensively:
#
#  LEAKAGE. The benchmark's 12 clean cases are frames of these same videos. A
#  model trained on them would be scored on its training data, and consecutive
#  frames are near-duplicates, so excluding the exact file is not enough - every
#  frame within HOLDOUT of a benchmark frame is dropped too. The exclusion list
#  is DERIVED from the benchmark ground truth (not hand-typed) so it cannot
#  drift away from the benchmark it protects.
#
#  CONTAMINATION. "Nominal" has to mean nominal:
#    v1  the reference session, clean throughout            -> train
#    v3  clean for the first frames; the GT anomalies at f0205/f0219 are staged
#        during it, and a person appears from ~f0229        -> train up to V3_MAX
#    v2  people walking through and staged objects from the start -> EXCLUDED
#    v4  clean, and the only DARK session - but 22 frames of which 3 are
#        benchmark negatives, so any use of it leaks       -> EXCLUDED, which
#        makes v4 the one fully held-out session at eval time
#  The consequence is stated rather than hidden: this model is trained on two
#  daylight sessions, so the v4 negatives test generalisation to an unseen
#  lighting, while the v1/v3 negatives only test unseen FRAMES of a seen session.
#
#  Run (CPU is enough - the encoder is frozen, only 15 M params train):
#      python tools/dinomaly_train.py --camera tram_1762 --epochs 40
# =============================================================================

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import tools.dinomaly as dm                                   # noqa: E402

MASKED_DIR = REPO_ROOT / "data" / "masked"
NOMINAL_DIR = REPO_ROOT / "data" / "nominal"


TRAIN_SESSIONS = ("v1", "v3")
V3_MAX = 150                # staging starts well before the GT anomaly at f0205
HOLDOUT = 15                # frames on either side of a benchmark frame
FRAME_RE = re.compile(r"tram_(\d+)_(v\d+)_f(\d+)")


def benchmark_frames():
    """{(session, index)} used by the benchmark, from the GT file itself."""
    from arsi_core import benchmarks
    _, gt = benchmarks.load()
    paths = [c["image"] for c in gt["cases"]] + list(gt["references"].values())
    out = set()
    for p in paths:
        mt = FRAME_RE.search(p)
        if mt:
            out.add((mt.group(2), int(mt.group(3))))
    return out


def nominal_frames(camera: str = "tram_1762", stride: int = 1):
    """The training list, with the reasons attached (printed, then stored in the
    checkpoint - a model is only as interpretable as the data behind it).

    Two sources, because the two camera families arrived differently:

      tram_1762  759 masked frames were already in data/masked/, so the session
                 rules and the benchmark holdout are applied HERE, below.
      1760 / 39T no frames existed at all. tools/build_nominal_frames.py cuts
                 them out of the source videos and applies the same holdout plus
                 a person filter at extraction time, so by the time they land in
                 data/nominal/<camera>/ the selection is already done. Read that
                 file for what each family can and cannot support - 1760 is
                 within-session only, and the 39T negatives are too.
    """
    if camera != "tram_1762":
        # `stride` is deliberately NOT applied here: build_nominal_frames.py
        # already samples one frame every 2 s, so the sampling rate has exactly
        # one owner per family. Applying it twice would silently halve the pool.
        pool = sorted((NOMINAL_DIR / camera).glob(f"{camera}_*.jpg"))
        if not pool:
            raise SystemExit(
                f"no nominal frames for '{camera}' under "
                f"{NOMINAL_DIR / camera}. Build them first:\n"
                f"    python tools/build_nominal_frames.py "
                f"--family {camera.split('-')[0]}")
        return pool, 0, set()
    held = benchmark_frames()
    picked, dropped = [], 0
    for f in sorted(MASKED_DIR.glob("tram_1762_v*_f*_masked.jpg")):
        mt = FRAME_RE.search(f.name)
        session, idx = mt.group(2), int(mt.group(3))
        if session not in TRAIN_SESSIONS:
            continue
        if session == "v3" and idx > V3_MAX:
            continue
        if any(s == session and abs(idx - i) <= HOLDOUT for s, i in held):
            dropped += 1
            continue
        picked.append(f)
    return picked[::stride], dropped, held


def _cache_path(files):
    """One file per (frame list, grid, layers). Landing in weights/ keeps it out
    of git and next to the checkpoint it belongs to."""
    key = hashlib.sha1(("|".join(str(f) for f in files)
                        + f"|{dm.INPUT_W}|{dm.LAYERS}").encode()).hexdigest()[:10]
    return dm.WEIGHTS_DIR / f".feat_{dm.INPUT_W}_{key}.pt"


def encode_all(files, use_cache: bool = True):
    """Frozen features for every training frame, kept as the two group targets
    only ([2, N, D] fp16 ~ 5 MB/frame): the encoder never trains, so this is
    computed once and the epochs then cost nothing but the decoder.

    Cached on disk because it is 7.5 min of CPU that does not depend on any
    hyper-parameter - sweeping the decoder should not re-pay for the encoder."""
    cache = _cache_path(files)
    if use_cache and cache.exists():
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        print(f"  features from cache: {cache.name}")
        return blob["feats"], tuple(blob["grid"])
    feats, valid = [], None
    t0 = time.time()
    for i, f in enumerate(files, 1):
        layers, v, grid = dm.encode(str(f))
        feats.append(torch.stack(dm.group_targets(layers)).half())
        valid = v if valid is None else valid
        if i % 10 == 0 or i == len(files):
            print(f"  encoded {i}/{len(files)}  ({time.time() - t0:.0f} s)", flush=True)
    out = torch.stack(feats)
    if use_cache:
        dm.WEIGHTS_DIR.mkdir(exist_ok=True)
        torch.save({"feats": out, "grid": list(grid)}, cache)
    return out, grid


def train(files, epochs=40, batch=4, lr=1e-3, seed=0, crop=(23, 40),
          use_cache=True):
    """Trains on random patch-grid CROPS, not whole frames. Measured on this
    laptop: a full 80x45 grid costs 3.6 s per step, so 40 epochs would be 5 h; a
    quarter-frame crop is ~0.9 s and doubles as augmentation (the paper trains on
    crops too). Inference still runs the FULL grid - the decoder has no
    positional parameters of its own, position lives in the encoder features."""
    torch.manual_seed(seed)
    feats, grid = encode_all(files, use_cache)    # [F, 2, N, D] fp16
    n = feats.shape[0]
    gh, gw = grid
    ch, cw = min(crop[0], gh), min(crop[1], gw)
    feats = feats.reshape(n, dm.N_GROUPS, gh, gw, -1)   # stays on CPU: the whole
    #   stack is ~5 MB/frame and only one crop per step is needed on the device
    model = dm.Dinomaly().to(dm.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, math.ceil(n / batch)) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    step, t0 = 0, time.time()
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            y0 = torch.randint(0, gh - ch + 1, (1,)).item()
            x0 = torch.randint(0, gw - cw + 1, (1,)).item()
            g = feats[idx][:, :, y0:y0 + ch, x0:x0 + cw].reshape(
                len(idx), dm.N_GROUPS, ch * cw, -1).float().to(dm.DEVICE)
            targets = [g[:, 0], g[:, 1]]
            preds = model(g.sum(1))               # bottleneck input = layer sum
            loss = dm.hard_cosine_loss(preds, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            tot += float(loss)
            nb += 1
        hist.append(tot / nb)
        print(f"  epoch {ep:3d}/{epochs}  loss {tot / nb:.4f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
    return model, hist, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="tram_1762")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", default="23x40", help="patch-grid crop, HxW")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--stride", type=int, default=2,
                    help="take one frame in N: consecutive frames are near-duplicates")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap the frame count")
    ap.add_argument("--no-cache", action="store_true",
                    help="re-encode instead of reusing the cached features")
    args = ap.parse_args()

    files, dropped, held = nominal_frames(args.camera, args.stride)
    if args.limit:
        files = files[:args.limit]
    by_session = {}
    for f in files:
        mt = FRAME_RE.search(f.name)
        # 1760 has one session; 39T names its moment between camera and t
        key = mt.group(2) if mt else (f.stem.split("_")[1] if "_t" in f.stem
                                      and len(f.stem.split("_")) > 2 else "run")
        by_session.setdefault(key, []).append(f)
    print(f"training frames: {len(files)}  "
          + "  ".join(f"{s}={len(v)}" for s, v in sorted(by_session.items())))
    print(f"held out around {len(held)} benchmark frames (+-{HOLDOUT}): "
          f"{dropped} frames dropped")
    print(f"grid {dm.INPUT_W}px wide, layers {dm.LAYERS}, device {dm.DEVICE}\n")

    crop = tuple(int(v) for v in args.crop.lower().split("x"))
    model, hist, grid = train(files, args.epochs, args.batch, args.lr, crop=crop,
                              use_cache=not args.no_cache)
    dm.save(model, args.camera, {
        "frames": [str(f.relative_to(REPO_ROOT)) for f in files],
        "n_frames": len(files), "sessions": sorted(by_session),
        "epochs": args.epochs, "stride": args.stride, "lr": args.lr,
        "crop": list(crop), "batch": args.batch,
        "loss_first": hist[0], "loss_last": hist[-1], "grid": list(grid),
        "excluded": {"sessions": ["v2 (people and staged objects)",
                                  "v4 (22 frames, 3 are benchmark negatives)"],
                     "holdout": HOLDOUT, "v3_max": V3_MAX},
    })
    print(f"\nsaved {dm.checkpoint_path(args.camera)}  "
          f"(loss {hist[0]:.4f} -> {hist[-1]:.4f})")


if __name__ == "__main__":
    main()
