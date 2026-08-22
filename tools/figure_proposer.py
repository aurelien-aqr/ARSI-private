#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - tools/figure_proposer.py
#  Figure 2 of the paper: the same frame proposed by the two localizer families,
#  side by side, with ground truth drawn underneath.
#
#  WHY THIS FIGURE. docs/paper/results.md §0a makes its claim with a strict-IoU
#  column (42/73 vs 58/73) and one number: on tram 3333 the photometric diff's
#  worst box covers 98.9 % of a 1280x720 frame. A box that large is a HIT under
#  the lenient rule and useless to a judge that only ever sees the crop. That is
#  the whole argument of the paper in one image, and a table cannot show it.
#
#  Usage:
#     python tools/figure_proposer.py                       # the default case
#     python tools/figure_proposer.py --case 3333_cam54_084021
# =============================================================================

import argparse, json, sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

GT_COLOR   = (60, 220, 90)
BOX_COLOR  = (255, 60, 60)
HUGE_COLOR = (255, 170, 0)      # a box past HUGE_FRAC of the frame is the point
HUGE_FRAC  = 0.30
OUT = REPO_ROOT / "docs" / "paper" / "figures"


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    return i / float((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)


def panel(image_path, regions, instances, title, subtitle):
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im, "RGBA")

    for inst in instances:                       # ground truth first, underneath
        x0, y0, x1, y1 = inst["bbox"]
        d.rectangle([x0, y0, x1, y1], outline=GT_COLOR, width=4)
        best = max((iou(inst["bbox"], r["bbox"]) for r in regions), default=0.0)
        # The strict rule of the benchmark is IoU >= 0.3, so the reader can
        # check the verdict on the image itself instead of trusting the caption.
        tag = f"IoU {best:.2f} {'PASS' if best >= 0.3 else 'FAIL'}"
        f = _font(20)
        tw, th = d.textbbox((0, 0), tag, font=f)[2:]
        # Below the box, unless that would fall off the frame - the floor
        # instances sit at the very bottom of every tram view.
        ty = y1 + 4 if y1 + th + 6 <= H else max(0, y0 - th - 6)
        tx = min(x0, W - tw - 4)
        d.rectangle([tx - 3, ty - 2, tx + tw + 3, ty + th + 2], fill=(0, 0, 0, 190))
        d.text((tx, ty), tag,
               fill=GT_COLOR if best >= 0.3 else (255, 110, 110), font=f)

    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        frac = ((x1 - x0) * (y1 - y0)) / float(W * H)
        col = HUGE_COLOR if frac >= HUGE_FRAC else BOX_COLOR
        d.rectangle([x0, y0, x1, y1], outline=col, width=3)
        if frac >= HUGE_FRAC:
            d.rectangle([x0, y0, x1, y1], fill=col[:3] + (40,))
            d.text((x0 + 8, y0 + 8), f"{frac:.1%} of the frame",
                   fill=col, font=_font(22))

    band = 64
    out = Image.new("RGB", (W, H + band), (255, 255, 255))
    out.paste(im, (0, band))
    d2 = ImageDraw.Draw(out)
    d2.text((12, 8),  title,    fill=(20, 20, 20), font=_font(26))
    d2.text((12, 38), subtitle, fill=(90, 90, 90), font=_font(18))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="3333_cam55_083517")
    ap.add_argument("--out", default=str(OUT / "fig2_proposers.jpg"))
    args = ap.parse_args()

    import vlm_05_reference_diff as m
    import tools.dino_localizer as dl

    gt = json.loads((REPO_ROOT / "benchmark" / "datasets" / "ground_truth.json").read_text())
    cases = gt["cases"].values() if isinstance(gt["cases"], dict) else gt["cases"]
    case = next(c for c in cases if c["id"] == args.case)
    ref = gt["references"][case["reference"]]
    ref = ref if isinstance(ref, str) else (ref.get("image") or ref.get("path"))
    insp, inst = case["image"], case["instances"]

    left, _ = m.localize(ref, insp)
    right, _ = dl.localize(ref, insp)
    print(f"{args.case}: photo {len(left)} regions, dino {len(right)} regions", flush=True)

    a = panel(insp, left, inst, "(a)  Photometric difference",
              f"{len(left)} proposals - the largest covers "
              f"{max(((b[2]-b[0])*(b[3]-b[1])) for b in [r['bbox'] for r in left]) / (1280*720):.1%} "
              f"of the frame")
    b = panel(insp, right, inst, "(b)  DINOv2 patch features",
              f"{len(right)} proposals - ground truth in green")

    W = a.width + b.width + 16
    fig = Image.new("RGB", (W, a.height), (255, 255, 255))
    fig.paste(a, (0, 0))
    fig.paste(b, (a.width + 16, 0))
    fig.thumbnail((2200, 2200))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.save(args.out, quality=92)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
