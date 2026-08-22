#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - tools/aboda_qualitative.py
#  The proposal stage on ABODA, the one public dataset whose CLASS is ours.
#
#  WHY QUALITATIVE AND NOT A TABLE. docs/PUBLIC_DATASETS.md already settles this:
#  ABODA is "11 CCTV sequences of unattended objects [...] Semantically it is
#  exactly our class, but 11 sequences is too little to carry a main results
#  table, and the licence is not clearly stated anywhere. Use it for qualitative
#  checks." On top of that, the copy in dataset/ ships the eleven .avi files and
#  a README and NOTHING ELSE - there are no annotations to score against. The
#  event-level numbers our plan wants (precision / recall / F1 per abandoned
#  object) need the unified temporal annotations from Luna et al., Sensors
#  18(12):4290, which are not in the repo. See docs/paper/results.md §4 for what
#  is missing and what it would take.
#
#  WHAT THIS DOES. For each sequence: build a reference from the opening frames
#  (median, same construction as tools/cdnet_eval.py), then run the shipped
#  proposal stage at several timestamps and write the boxed frames out. The
#  output is figure material and a sanity check on a domain we never tuned for -
#  outdoor, night, crowds - not a measurement.
#
#  Usage:  python tools/aboda_qualitative.py --videos video1,video5 --at 0.5,0.8
# =============================================================================

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ABODA = Path(os.environ.get(
    "ABODA_ROOT",
    "/home/aurelien/Documents/vsb/dataset/ABandoned Objects DAtaset (ABODA)"))
OUT = REPO_ROOT / "docs" / "public_benchmarks" / "aboda"


def frame_count(video: Path) -> int:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, timeout=600)
    return int(r.stdout.strip() or 0)


def extract(video: Path, index: int, dest: Path) -> Path:
    """One frame by index, as PNG. -vf select is exact where -ss is not."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-vf", f"select=eq(n\\,{index})", "-vsync", "vfr", "-frames:v", "1",
         str(dest)], check=True, timeout=600)
    return dest


def reference(video: Path, n: int, span: int, work: Path) -> Path:
    """Median of `n` frames spread over the first `span` frames: the opening of
    each ABODA sequence is the closest thing it has to a clean plate."""
    idx = np.unique(np.linspace(0, max(span - 1, 1), n).astype(int))
    frames = [np.asarray(Image.open(extract(video, i, work / f"ref_{i:06d}.png"))
                         .convert("RGB"), dtype=np.uint8) for i in idx]
    med = np.median(np.stack(frames), axis=0).astype(np.uint8)
    out = work / "reference.png"
    Image.fromarray(med).save(out)
    return out


def draw(image_path: Path, regions, out_path: Path):
    im = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    for i, r in enumerate(regions, 1):
        x0, y0, x1, y1 = r["bbox"]
        d.rectangle([x0, y0, x1, y1], outline=(255, 60, 60), width=3)
        d.text((x0 + 4, max(0, y0 - 12)), str(i), fill=(255, 60, 60))
    im.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="", help="comma-separated stems, default all")
    ap.add_argument("--at", default="0.4,0.6,0.8",
                    help="fractions of the sequence to inspect")
    ap.add_argument("--localizer", default="photo")
    ap.add_argument("--ref-frames", type=int, default=15)
    ap.add_argument("--ref-span", type=int, default=150)
    args = ap.parse_args()

    import vlm_05_reference_diff as v5
    from arsi_core import localizers as loc
    v5.PERSON_FILTER = False        # ABODA is full of people; same reasoning as CDnet

    vids = sorted(ABODA.glob("video*.avi"), key=lambda p: int(p.stem[5:]))
    if args.videos:
        keep = set(args.videos.split(","))
        vids = [v for v in vids if v.stem in keep]
    fracs = [float(x) for x in args.at.split(",")]

    OUT.mkdir(parents=True, exist_ok=True)
    index = {}
    for v in vids:
        work = Path("/tmp/arsi_aboda") / v.stem
        work.mkdir(parents=True, exist_ok=True)
        n = frame_count(v)
        ref = reference(v, args.ref_frames, min(args.ref_span, n // 4 or 1), work)
        print(f"{v.stem}: {n} frames, reference from the first "
              f"{min(args.ref_span, n // 4 or 1)}", flush=True)
        index[v.stem] = {"frames": n, "shots": []}
        for f in fracs:
            i = min(n - 1, int(n * f))
            insp = extract(v, i, work / f"insp_{i:06d}.png")
            regions, info = loc.localize(args.localizer, v5, str(ref), str(insp))
            out = OUT / f"{v.stem}_f{i:06d}_{args.localizer.replace('+', '-')}.jpg"
            draw(insp, regions, out)
            print(f"   frame {i:6d} ({f:.0%})  {len(regions):3d} regions -> {out.name}",
                  flush=True)
            index[v.stem]["shots"].append(
                {"frame": i, "fraction": f, "n_regions": len(regions),
                 "image": str(out.relative_to(REPO_ROOT)),
                 "bboxes": [r["bbox"] for r in regions]})

    (OUT / "index.json").write_text(json.dumps(
        {"_about": ("Qualitative only - ABODA ships no annotations, so nothing "
                    "here is scored. Reference = median of the opening frames. "
                    "Person filter OFF."),
         "localizer": args.localizer, "sequences": index}, indent=1))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
