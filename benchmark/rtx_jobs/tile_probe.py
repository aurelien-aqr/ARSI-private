#!/usr/bin/env python3
"""Does tiling a big region rescue the objects the judge rejects on 39T?

The four 39T frames the pipeline misses all show the same geometry: the object
IS inside a proposed region, but occupies 0.4-4.2 % of it, so the judge sees a
crop where a bottle is a dozen pixels. This re-judges those frames with every
large region cut into overlapping tiles, and asks whether any tile that lands on
a ground-truth object now comes back YES.

Diagnostic only: it never uses the GT to choose what to judge, only to score.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/Documents/ARSI-private"))

from PIL import Image                                  # noqa: E402
import vlm_05_reference_diff as m                      # noqa: E402
from arsi_core import benchmarks                       # noqa: E402

MODEL = "haervwe/GLM-4.6V-Flash-9B:latest"
CASES = ["39T_cam52_084637", "39T_cam53_084637",
         "39T_cam53_085124", "39T_cam54_083517"]

TILE_TRIGGER = 40_000    # px of bbox area above which a region is cut up
TILE = 160               # tile side in reference-space pixels
STEP = 120               # 25 % overlap, so an object on a seam still fits a tile
TILE_MARGIN = 20         # the shipped 40 px + 0.75 context would re-shrink the
TILE_CONTEXT = 0.25      # object inside its own tile, which is the whole point

m.MODEL_NAME = MODEL


def tiles_of(bbox):
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    if w * h <= TILE_TRIGGER:
        return [list(bbox)]
    xs = list(range(x0, max(x0 + 1, x1 - TILE + 1), STEP)) or [x0]
    ys = list(range(y0, max(y0 + 1, y1 - TILE + 1), STEP)) or [y0]
    if xs[-1] + TILE < x1:
        xs.append(x1 - TILE)
    if ys[-1] + TILE < y1:
        ys.append(y1 - TILE)
    out = []
    for ty in ys:
        for tx in xs:
            out.append([max(x0, tx), max(y0, ty),
                        min(x1, tx + TILE), min(y1, ty + TILE)])
    return out


def cov(inner, outer):
    """Share of `inner` covered by `outer`."""
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return ((x2 - x1) * (y2 - y1)
            / max(1, (inner[2] - inner[0]) * (inner[3] - inner[1])))


_, doc = benchmarks.load("ground_truth")
refs, ROOT = doc["references"], benchmarks.REPO_ROOT
by_id = {c["id"]: c for c in doc["cases"]}

m.check_model(MODEL)
grand = {"tiles": 0, "yes": 0, "inst": 0, "rescued": 0, "calls_s": 0.0}

for cid in CASES:
    case = by_id[cid]
    ref_path = str(ROOT / refs[case["reference"]])
    img_path = str(ROOT / case["image"])
    reference = Image.open(ref_path).convert("RGB")
    image = Image.open(img_path).convert("RGB")
    if image.size != reference.size:
        image = image.resize(reference.size)

    regions, loc = m.localize(ref_path, img_path)
    print(f"\n=== {cid}: {len(regions)} regions proposees "
          f"(base={loc['base']} +lo={loc['second']} +edge={loc['edge']} "
          f"-person={loc['person_veto']})")

    yes_boxes = []
    n_tiles = 0
    t0 = time.time()
    for r in regions:
        for tb in tiles_of(r["bbox"]):
            n_tiles += 1
            sub = {"bbox": tb, "area": (tb[2] - tb[0]) * (tb[3] - tb[1])}
            is_obj, label = m.classify_with_vlm(image, reference, sub,
                                                TILE_MARGIN, TILE_CONTEXT)
            if is_obj and not m.is_non_anomaly(label):
                yes_boxes.append((tb, label))
    dt = time.time() - t0
    print(f"    {n_tiles} tuiles jugees en {dt:.0f}s -> {len(yes_boxes)} YES")

    for ins in (case.get("instances") or []):
        grand["inst"] += 1
        hits = [(cov(ins["bbox"], tb), lb) for tb, lb in yes_boxes
                if cov(ins["bbox"], tb) > 0.3]
        if hits:
            grand["rescued"] += 1
            best = max(hits)
            print(f"    RECUPERE  {ins['label']:14} couverture {best[0]:.2f} "
                  f"-> \"{str(best[1])[:60]}\"")
        else:
            print(f"    toujours rate  {ins['label']:14} "
                  f"(aire {(ins['bbox'][2]-ins['bbox'][0])*(ins['bbox'][3]-ins['bbox'][1]):,} px)")

    grand["tiles"] += n_tiles
    grand["yes"] += len(yes_boxes)
    grand["calls_s"] += dt

print(f"\n===== TOTAL: {grand['rescued']}/{grand['inst']} instances recuperees | "
      f"{grand['tiles']} tuiles -> {grand['yes']} YES | "
      f"{grand['calls_s']:.0f}s ({grand['calls_s']/max(1,grand['tiles']):.2f}s/appel)")
json.dump(grand, open("/tmp/tile_probe_result.json", "w"), indent=2)
