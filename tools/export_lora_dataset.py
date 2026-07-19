#!/usr/bin/env python3
"""Build a LoRA fine-tuning dataset for the vlm_05 crop judge from
HUMAN-verified labels (docs/LORA_PLAN.md).

Sources, in order of trust:
1. App-job reviews (data/app/jobs/*/review.json, made in ARSI Studio's
   Review mode): every judged detection becomes a sample — TP -> "YES <label>",
   FP -> "NO"; every reviewer-drawn missed box becomes "YES <label>".
2. Optionally (--include-benchmark) the benchmark ground truth: the localizer
   runs on each case, regions matching a GT instance box (IoU >= 0.3) become
   "YES <gt label>", unmatched regions become "NO". WARNING: the 29-case
   benchmark is the eval set — training on it destroys it as an eval.
   Only use this to bootstrap once a bigger eval exists.

Each sample is the EXACT inference-time artifact: the reference|inspection
side-by-side crop rendered by vlm_05.render_crop_pair, paired with the
current vlm_05 PROMPT. Output is LLaMA-Factory sharegpt-format JSONL
(train.jsonl / val.jsonl, split by FRAME so crops of one frame never
straddle the split) plus the images/ directory and a stats report.

Usage:
  venv/bin/python tools/export_lora_dataset.py --out data/lora_dataset
  venv/bin/python tools/export_lora_dataset.py --out data/lora_dataset \
      --include-benchmark   # leakage warning above applies
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image                                    # noqa: E402

import vlm_05_reference_diff as vlm05                    # noqa: E402
from arsi_core import APP_DATA                           # noqa: E402

IOU_MATCH = 0.3
VAL_FRACTION = 0.1


def reanchor(path: str) -> Path:
    """Absolute paths recorded on another machine -> local data/ tree."""
    p = Path(path)
    if p.is_file():
        return p
    s = str(path)
    i = s.find("/data/")
    if i != -1 and (REPO_ROOT / s[i + 1:]).is_file():
        return REPO_ROOT / s[i + 1:]
    return p


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if not inter:
        return 0.0
    area = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / area


class Exporter:
    def __init__(self, out_dir: Path, prompt: str):
        self.out = out_dir
        self.img_dir = out_dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.prompt = prompt
        self.samples = []        # (frame_key, sample_dict)
        self.stats = Counter()

    def add(self, frame_key: str, insp: Image.Image, ref: Image.Image,
            bbox, answer: str, source: str):
        name = f"{frame_key}_{self.stats['total']:05d}.jpg"
        crop = vlm05.render_crop_pair(insp, ref, bbox,
                                      vlm05.CROP_MARGIN, vlm05.CROP_CONTEXT)
        crop.save(self.img_dir / name, quality=92)
        self.samples.append((frame_key, {
            "messages": [{"role": "user", "content": "<image>" + self.prompt},
                         {"role": "assistant", "content": answer}],
            "images": [f"images/{name}"]}))
        self.stats["total"] += 1
        self.stats["yes" if answer.startswith("YES") else "no"] += 1
        self.stats[f"src:{source}"] += 1

    def write(self):
        # the dataset dir must be self-sufficient for llamafactory-cli:
        # ship the dataset_info.json registration alongside the JSONL
        info = REPO_ROOT / "tools" / "lora" / "dataset_info.json"
        if info.exists():
            import shutil
            shutil.copy(info, self.out / "dataset_info.json")
        # frame-level split: one frame's crops all land on the same side
        train, val = [], []
        for key, sample in self.samples:
            h = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 100
            (val if h < VAL_FRACTION * 100 else train).append(sample)
        for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
            with open(self.out / name, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats = {**self.stats, "train": len(train), "val": len(val),
                 "prompt_sha1": hashlib.sha1(self.prompt.encode()).hexdigest()[:12]}
        with open(self.out / "stats.json", "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=1)
        return stats


def export_reviews(ex: Exporter):
    jobs_dir = APP_DATA / "jobs"
    n_jobs = 0
    for review_path in sorted(jobs_dir.glob("*/review.json")):
        job_dir = review_path.parent
        try:
            with open(job_dir / "results.json", encoding="utf-8") as fh:
                results = json.load(fh)
            with open(review_path, encoding="utf-8") as fh:
                review = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  ! {job_dir.name}: unreadable ({exc}) — skipped")
            continue
        ref_path = (results.get("config") or {}).get("reference")
        if not ref_path:
            print(f"  ! {job_dir.name}: no reference (whole-frame pipeline) — "
                  f"skipped, crop pairs need one")
            continue
        ref_path = reanchor(ref_path)
        if not ref_path.is_file():
            print(f"  ! {job_dir.name}: reference not found locally — skipped")
            continue
        ref_img = Image.open(ref_path).convert("RGB")
        by_id = {f["frame_id"]: f for f in results["frames"]}
        used = 0
        for frame_id, entry in review.get("frames", {}).items():
            if not entry.get("done"):
                continue          # only confirmed frames are trustworthy
            frame = by_id.get(frame_id)
            img_path = reanchor(frame["image"]) if frame else None
            if not frame or not img_path.is_file():
                continue
            insp = Image.open(img_path).convert("RGB")
            if insp.size != ref_img.size:
                insp = insp.resize(ref_img.size)
            dets = frame.get("detections") or []
            key = f"{job_dir.name}_{frame_id}"
            for idx, verdict in entry.get("verdicts", {}).items():
                d = dets[int(idx)]
                if not d.get("bbox"):
                    continue
                answer = f"YES {d['label']}" if verdict == "tp" else "NO"
                ex.add(key, insp, ref_img, d["bbox"], answer, "review")
                used += 1
            for m in entry.get("missed", []):
                ex.add(key, insp, ref_img, m["bbox"], f"YES {m['label']}",
                       "review-missed")
                used += 1
        if used:
            n_jobs += 1
            print(f"  {job_dir.name}: {used} samples")
    if not n_jobs:
        print("  (no reviewed jobs found — open a job in ARSI Studio, "
              "toggle Review, judge some frames)")


def export_benchmark(ex: Exporter):
    gt_path = REPO_ROOT / "benchmark" / "ground_truth.json"
    with open(gt_path, encoding="utf-8") as fh:
        gt = json.load(fh)
    refs = gt["references"]
    for case in gt["cases"]:
        img_path = REPO_ROOT / case["image"]
        ref_path = REPO_ROOT / refs[case["reference"]]
        if not img_path.exists() or not ref_path.exists():
            continue
        regions, _ = vlm05.localize(str(ref_path), str(img_path))
        ref_img = Image.open(ref_path).convert("RGB")
        insp = Image.open(img_path).convert("RGB")
        if insp.size != ref_img.size:
            insp = insp.resize(ref_img.size)
        instances = case.get("instances", [])
        for r in regions:
            best = max(instances, key=lambda t: iou(r["bbox"], t["bbox"]),
                       default=None)
            if best and iou(r["bbox"], best["bbox"]) >= IOU_MATCH:
                what = best.get("label") or best.get("note") or best.get("type", "anomaly")
                answer = f"YES {what}"
            else:
                answer = "NO"
            ex.add(f"bench_{case['id']}", insp, ref_img, r["bbox"], answer,
                   "benchmark")
        print(f"  {case['id']}: {len(regions)} regions")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/lora_dataset")
    ap.add_argument("--include-benchmark", action="store_true",
                    help="ALSO mine benchmark GT (destroys it as an eval set!)")
    args = ap.parse_args()

    out = Path(args.out)
    ex = Exporter(out, vlm05.PROMPT)
    print("== app-job reviews ==")
    export_reviews(ex)
    if args.include_benchmark:
        print("== benchmark GT (eval-set leakage — you were warned) ==")
        export_benchmark(ex)
    if not ex.samples:
        print("nothing to export")
        return 1
    stats = ex.write()
    print(f"\nwrote {stats['total']} samples -> {out} "
          f"(train {stats['train']} / val {stats['val']}; "
          f"YES {stats['yes']} / NO {stats['no']})")
    if stats["yes"] and stats["no"] and not 0.2 < stats["yes"] / stats["no"] < 5:
        print("! class balance is skewed — review more frames of the "
              "under-represented class before training")
    return 0


if __name__ == "__main__":
    sys.exit(main())
