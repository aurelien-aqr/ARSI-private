#!/usr/bin/env python3
"""Tile-size sweep over the whole 39T arm, judged end to end.

Arm 0 is the shipped behaviour re-run through this same script: it must
reproduce today's 39T numbers, which is what makes the other arms comparable.
Every arm applies the pipeline's own post-filters (is_non_anomaly,
is_implausible, dedupe_regions) so the difference is the tiling alone.
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
TRIGGER = 40_000          # bbox area above which a region gets cut up

# (name, tile side, margin, context) - tile=0 means "ship it whole"
ARMS = [
    ("baseline", 0, m.CROP_MARGIN, m.CROP_CONTEXT),
    ("tile160", 160, 20, 0.25),
]

m.MODEL_NAME = MODEL


def tiles_of(bbox, tile):
    x0, y0, x1, y1 = bbox
    if not tile or (x1 - x0) * (y1 - y0) <= TRIGGER:
        return [list(bbox)]
    step = int(tile * 0.75)
    xs = list(range(x0, max(x0 + 1, x1 - tile + 1), step)) or [x0]
    ys = list(range(y0, max(y0 + 1, y1 - tile + 1), step)) or [y0]
    if xs[-1] + tile < x1:
        xs.append(x1 - tile)
    if ys[-1] + tile < y1:
        ys.append(y1 - tile)
    return [[max(x0, tx), max(y0, ty), min(x1, tx + tile), min(y1, ty + tile)]
            for ty in ys for tx in xs]


def iou_cov(inner, outer):
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0
    it = (x2 - x1) * (y2 - y1)
    ia = (inner[2] - inner[0]) * (inner[3] - inner[1])
    oa = (outer[2] - outer[0]) * (outer[3] - outer[1])
    return it / (ia + oa - it), it / max(1, ia)


_, doc = benchmarks.load("ground_truth")
refs, ROOT = doc["references"], benchmarks.REPO_ROOT
cases = [c for c in doc["cases"] if not c["reference"].startswith("39T")]
m.check_model(MODEL)
print(f"{len(cases)} cas 1762, {sum(len(c.get('instances') or []) for c in cases)} instances\n",
      flush=True)

results = {}
for name, tile, margin, context in ARMS:
    t0 = time.time()
    calls = 0
    inst_tot = inst_hit = inst_strict = 0
    kept_tot = fp_boxes = 0
    frames_flagged = frames_anom = frames_clean = false_alarm_frames = 0

    for case in cases:
        ref_path = str(ROOT / refs[case["reference"]])
        img_path = str(ROOT / case["image"])
        reference = Image.open(ref_path).convert("RGB")
        image = Image.open(img_path).convert("RGB")
        if image.size != reference.size:
            image = image.resize(reference.size)

        regions, _ = m.localize(ref_path, img_path)
        kept = []
        for r in regions:
            for tb in tiles_of(r["bbox"], tile):
                calls += 1
                sub = {"bbox": tb,
                       "area": (tb[2] - tb[0]) * (tb[3] - tb[1])}
                is_obj, label = m.classify_with_vlm(image, reference, sub,
                                                    margin, context)
                if m.is_non_anomaly(label) or m.is_implausible(label, sub["area"]):
                    is_obj = False
                if is_obj:
                    sub["vlm_label"] = label
                    kept.append(sub)
        kept = m.dedupe_regions(kept)
        kept_tot += len(kept)

        instances = case.get("instances") or []
        if case.get("has_anomaly"):
            frames_anom += 1
            if kept:
                frames_flagged += 1
        else:
            frames_clean += 1
            if kept:
                false_alarm_frames += 1

        matched = set()
        for ins in instances:
            inst_tot += 1
            best_i = best_c = 0.0
            best_k = None
            for idx, k in enumerate(kept):
                i, c = iou_cov(ins["bbox"], k["bbox"])
                if c > best_c:
                    best_c, best_k = c, idx
                best_i = max(best_i, i)
            if best_c >= 0.5 or best_i >= 0.1:
                inst_hit += 1
                if best_k is not None:
                    matched.add(best_k)
            if best_i >= 0.3:
                inst_strict += 1
        fp_boxes += len(kept) - len(matched)

    dt = time.time() - t0
    results[name] = dict(
        tile=tile, calls=calls, seconds=round(dt, 1),
        inst=inst_tot, hit=inst_hit, strict=inst_strict,
        kept=kept_tot, fp=fp_boxes,
        recall=round(inst_hit / max(1, inst_tot), 3),
        strict_recall=round(inst_strict / max(1, inst_tot), 3),
        region_precision=round((kept_tot - fp_boxes) / max(1, kept_tot), 3),
        frame_recall=round(frames_flagged / max(1, frames_anom), 3),
        specificity=round(1 - false_alarm_frames / max(1, frames_clean), 3),
    )
    print(f"{name:9} tile={tile or '-':>4} | {calls:5} appels {dt:6.0f}s | "
          f"recall {results[name]['recall']:.3f} strict {results[name]['strict_recall']:.3f} | "
          f"gardees {kept_tot:3} dont {fp_boxes:3} FP -> precision {results[name]['region_precision']:.3f} | "
          f"frame recall {results[name]['frame_recall']:.3f} spec {results[name]['specificity']:.3f}",
          flush=True)
    json.dump(results, open("/tmp/tile_sweep_1762_result.json", "w"), indent=2)

print("\nfini", flush=True)
