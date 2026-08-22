#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - tools/dinomaly2_train.py
#  Trains the Dinomaly2 variant (tools/dinomaly2.py) on one camera's nominal
#  frames. Same data, same holdout, same crops and same schedule as
#  dinomaly_train.py - deliberately, because the only question this answers is
#  "do the five Dinomaly2 elements change anything ON OUR CAMERAS", and that
#  question dies if anything else moves at the same time.
#
#  What differs from dinomaly_train.py, and only this:
#    - the feature cache also stores the CLS token, without which context-aware
#      recentering is impossible;
#    - the cached targets are already recentered, so a training step costs the
#      same as v1's;
#    - the loss is dinomaly2.loose_loss with its 1000-step ratio warm-up.
#
#  Usage:  python tools/dinomaly2_train.py --camera 3333-cam53
# =============================================================================

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import tools.dinomaly as dm                                    # noqa: E402
import tools.dinomaly2 as d2                                   # noqa: E402
from tools.dinomaly_train import nominal_frames, FRAME_RE, HOLDOUT  # noqa: E402


def _cache_path(files):
    key = hashlib.sha1(("|".join(str(f) for f in files)
                        + f"|{dm.INPUT_W}|{dm.LAYERS}|car={d2.CAR}").encode()).hexdigest()[:10]
    return dm.WEIGHTS_DIR / f".feat2_{dm.INPUT_W}_{key}.pt"


def encode_all(files, use_cache: bool = True):
    """[F, 1 + N_GROUPS, N, D] fp16: the bottleneck input, then the recentered
    group targets. Recentering is folded in here because it depends only on the
    frozen encoder, so it belongs on the same side of the cache as the encoder."""
    cache = _cache_path(files)
    if use_cache and cache.exists():
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        print(f"  features from cache: {cache.name}")
        return blob["feats"], tuple(blob["grid"]), blob["valid"]
    feats, valid, grid = [], None, None
    t0 = time.time()
    for i, f in enumerate(files, 1):
        layers, cls, v, grid = d2.encode(str(f))
        row = [d2.bottleneck_input(layers)] + d2.group_targets(layers, cls)
        feats.append(torch.stack(row).half())
        valid = v if valid is None else valid
        if i % 10 == 0 or i == len(files):
            print(f"  encoded {i}/{len(files)}  ({time.time() - t0:.0f} s)", flush=True)
    out = torch.stack(feats)
    if use_cache:
        dm.WEIGHTS_DIR.mkdir(exist_ok=True)
        torch.save({"feats": out, "grid": list(grid), "valid": valid}, cache)
    return out, grid, valid


def train(files, epochs=40, batch=4, lr=1e-3, seed=0, crop=(23, 40),
          use_cache=True):
    torch.manual_seed(seed)
    feats, grid, valid = encode_all(files, use_cache)
    n, gh, gw = feats.shape[0], grid[0], grid[1]
    ch, cw = min(crop[0], gh), min(crop[1], gw)
    feats = feats.reshape(n, 1 + dm.N_GROUPS, gh, gw, -1)
    vgrid = valid.reshape(gh, gw)
    model = d2.Dinomaly2().to(dm.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, math.ceil(n / batch)) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    step, t0, hist = 0, time.time(), []
    for ep in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            y0 = torch.randint(0, gh - ch + 1, (1,)).item()
            x0 = torch.randint(0, gw - cw + 1, (1,)).item()
            g = feats[idx][:, :, y0:y0 + ch, x0:x0 + cw].reshape(
                len(idx), 1 + dm.N_GROUPS, ch * cw, -1).float().to(dm.DEVICE)
            v = vgrid[y0:y0 + ch, x0:x0 + cw].reshape(-1).to(dm.DEVICE)
            targets = [g[:, k + 1] for k in range(dm.N_GROUPS)]
            preds = model(g[:, 0])
            step += 1
            loss = d2.loose_loss(preds, targets, valid=v,
                                 p=d2.warm_ratio(step))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
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
    ap.add_argument("--crop", default="23x40")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    files, dropped, held = nominal_frames(args.camera, args.stride)
    if args.limit:
        files = files[:args.limit]
    by_session = {}
    for f in files:
        mt = FRAME_RE.search(f.name)
        key = mt.group(2) if mt else (f.stem.split("_")[1] if "_t" in f.stem
                                      and len(f.stem.split("_")) > 2 else "run")
        by_session.setdefault(key, []).append(f)
    print(f"[dinomaly2] training frames: {len(files)}  "
          + "  ".join(f"{s}={len(v)}" for s, v in sorted(by_session.items())))
    print(f"held out around {len(held)} benchmark frames (+-{HOLDOUT}): "
          f"{dropped} dropped")
    print(f"grid {dm.INPUT_W}px, layers {dm.LAYERS}, dropout {d2.DROPOUT}, "
          f"loose loss p={d2.LL_RATIO} factor={d2.LL_FACTOR}, device {dm.DEVICE}\n")

    crop = tuple(int(v) for v in args.crop.lower().split("x"))
    model, hist, grid = train(files, args.epochs, args.batch, args.lr, crop=crop,
                              use_cache=not args.no_cache)
    d2.save(model, args.camera, {
        "frames": [str(f.relative_to(REPO_ROOT)) for f in files],
        "n_frames": len(files), "sessions": sorted(by_session),
        "epochs": args.epochs, "stride": args.stride, "lr": args.lr,
        "crop": list(crop), "batch": args.batch,
        "loss_first": hist[0], "loss_last": hist[-1], "grid": list(grid),
    })
    print(f"\nsaved {d2.checkpoint_path(args.camera)}  "
          f"(loss {hist[0]:.4f} -> {hist[-1]:.4f})")


if __name__ == "__main__":
    main()
