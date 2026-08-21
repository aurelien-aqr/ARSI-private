#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - tools/build_nominal_frames.py
#  Extracts the NOMINAL training pool for the reference-free localizer
#  (tools/dinomaly_train.py) on the 1760 and 3333 cameras, from the source videos.
#
#  WHY this file exists. Dinomaly was only ever trained on tram_1762, where 759
#  masked frames were already sitting in data/masked/. On 1760 and 3333 the only
#  images in the repo are the 39 benchmark frames themselves, so "train a model
#  per camera" first means "build a nominal set per camera" - and, as in
#  dinomaly_train.py, THE TRAINING SET IS THE EXPERIMENT. It is therefore built
#  here, explicitly, with the exclusions derived from the benchmark rather than
#  typed by hand.
#
#  WHAT IS AND IS NOT AVAILABLE (measured 2026-08-19, durations from the videos):
#
#  3333 has nine moments, and no leak-free pool. Four of them carry the staged
#  anomalies of the benchmark, 08-55-37 is the reference moment and 08-59-54 is
#  a benchmark negative; the three that are unlabelled last 12.4 s, 7.8 s and
#  10.4 s - ~30 s in total, not a training set. So we train on 08-55-37 and
#  08-59-54 minus the holdout, and the consequence is STATED rather than hidden:
#  the 14 anomalous 3333 cases stay genuinely out-of-session, but the 7 negatives
#  come from the sessions the model was trained on, so 3333 specificity is
#  optimistic and must be read as such.
#
#  1760 has ONE ten-minute run per camera and the benchmark samples it from
#  t=70 s to t=570 s. Every nominal frame therefore shares that run's light.
#  1760 measures within-session behaviour only; it cannot support any
#  cross-session claim.
#
#  THE PERSON FILTER IS NOT OPTIONAL. These are service runs with passengers -
#  build_1760_benchmark.py already had to reject cam04 at t=20 s for a passenger
#  on the phone. A blind sample would teach the model that people are nominal,
#  which is the one thing the pipeline must keep flagging. Frames where the
#  shipped YOLOv8n detector finds anyone are dropped, and the count is printed.
#
#  Frames are masked HERE, once, with the same MaskSpec the benchmark builders
#  use: every consumer must see the pixels the pipeline compares.
#
#  Run:  venv/bin/python tools/build_nominal_frames.py --family 1760
#        venv/bin/python tools/build_nominal_frames.py --family 3333
# =============================================================================

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from arsi_core import benchmarks                                # noqa: E402
from arsi_core.masking import MaskSpec                          # noqa: E402
from PIL import Image                                           # noqa: E402
import vlm_05_reference_diff as m                               # noqa: E402

VIDEOS = REPO_ROOT / "data" / "videos"
MASKS = REPO_ROOT / "data" / "masks_labelme"
OUT_ROOT = REPO_ROOT / "data" / "nominal"

STRIDE = 2.0          # seconds between sampled frames; 25 fps neighbours are
#                       near-duplicates and cost encoder time for no new signal
HOLDOUT = 15.0        # seconds on either side of a benchmark frame

#: The 3333 moments used as nominal. The other seven are anomalous or too short -
#: see the header. Both of these are ALSO benchmark sessions, hence the holdout.
NOMINAL_MOMENTS_3333 = ("08-55-37", "08-59-54")

#: (camera, moment) pairs the person filter cannot save. cam52 at 08-59-54 has
#: two passengers AND a bag at their feet (build_3333_benchmark.py, which refused
#: to label it for that reason): YOLO would drop the frames showing the people,
#: and keep the ones where the bag sits alone - teaching the model that a left
#: bag is nominal, on the very camera whose benchmark is about left objects.
#: Human knowledge the detector cannot recover, so it is written down.
EXCLUDE_3333 = {("3333-cam52", "08-59-54")}

#: data/benchmark_1760/1760-cam04_t120.jpg  ->  ("1760-cam04", None, 120)
#: data/benchmark_3333/3333-cam53_08-35-17_t60.jpg -> ("3333-cam53", "08-35-17", 60)
RE_1760 = re.compile(r"(1760-cam\d+)_t(\d+)")
RE_3333 = re.compile(r"(3333-cam\d+)_(\d\d-\d\d-\d\d)_t(\d+)")

_masks = {}


def mask_for(camera: str) -> MaskSpec:
    """camera is '1760-cam04' / '3333-cam53' - the benchmark's own reference key."""
    if camera not in _masks:
        family = camera.split("-")[0]
        path = MASKS / family / f"{camera}.json"
        _masks[camera] = MaskSpec.from_labelme(json.loads(path.read_text()),
                                               name=camera)
    return _masks[camera]


def benchmark_times():
    """{camera: {(moment, seconds)}} used by the benchmark, READ FROM the ground
    truth so the exclusion cannot drift away from the benchmark it protects."""
    _, gt = benchmarks.load()
    paths = [c["image"] for c in gt["cases"]] + list(gt["references"].values())
    out = {}
    for p in paths:
        mt = RE_3333.search(p)
        if mt:
            out.setdefault(mt.group(1), set()).add((mt.group(2), int(mt.group(3))))
            continue
        mt = RE_1760.search(p)
        if mt:
            out.setdefault(mt.group(1), set()).add((None, int(mt.group(2))))
    return out


def video_for(camera: str, moment):
    if moment is None:
        return VIDEOS / "1760" / f"{camera}.mp4"
    hits = sorted((VIDEOS / "3333").glob(f"*_{moment}"))
    if not hits:
        raise FileNotFoundError(f"no 3333 moment ending in {moment}")
    return hits[0] / f"{camera}.mp4"


def sample_times(path: Path, held, moment):
    """Timestamps to extract: every STRIDE seconds, minus the HOLDOUT windows."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    dur = n / fps
    bad = [s for mo, s in held if mo == moment]
    out = []
    t = STRIDE
    while t < dur - 1.0:
        if all(abs(t - b) > HOLDOUT for b in bad):
            out.append(t)
        t += STRIDE
    return out, dur, len(bad)


def extract(camera: str, moment, times, out_dir: Path):
    """Grab, mask, person-filter, write. Returns (kept, dropped_person)."""
    path = video_for(camera, moment)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    spec = mask_for(camera)
    tag = f"{moment}_" if moment else ""
    kept, dropped = 0, 0
    for t in times:
        dst = out_dir / f"{camera}_{tag}t{int(round(t)):04d}.jpg"
        if dst.exists():
            kept += 1
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, fr = cap.read()
        if not ok:
            continue
        img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        spec.apply(img).save(dst, quality=92)
        # the person check runs on the MASKED frame, the same pixels the
        # pipeline sees; a passenger is inside the tram, never behind a window
        if m.person_boxes(str(dst), img.size):
            dst.unlink()
            dropped += 1
        else:
            kept += 1
    cap.release()
    return kept, dropped


def build(family: str):
    held = benchmark_times()
    cams = sorted(c for c in held if c.startswith(family + "-"))
    if not cams:
        raise SystemExit(f"no benchmark camera for family '{family}'")
    moments = NOMINAL_MOMENTS_3333 if family == "3333" else (None,)
    total_kept = total_drop = 0
    for cam in cams:
        out_dir = OUT_ROOT / cam
        out_dir.mkdir(parents=True, exist_ok=True)
        cam_kept = cam_drop = 0
        for moment in moments:
            if (cam, moment) in EXCLUDE_3333:
                print(f"  {cam} {moment:<10} EXCLUDED (passengers + a bag on the "
                      f"floor; not nominal)", flush=True)
                continue
            times, dur, n_held = sample_times(video_for(cam, moment),
                                              held[cam], moment)
            k, d = extract(cam, moment, times, out_dir)
            cam_kept += k
            cam_drop += d
            label = moment or "single run"
            print(f"  {cam} {label:<10} {dur:6.1f}s  {len(times):4d} sampled "
                  f"(-{n_held} benchmark frames x +-{HOLDOUT:.0f}s)  "
                  f"kept {k:4d}  person-dropped {d:3d}", flush=True)
        print(f"{cam}: {cam_kept} nominal frames, {cam_drop} dropped\n")
        total_kept += cam_kept
        total_drop += cam_drop
    print(f"{family}: {total_kept} nominal frames over {len(cams)} cameras, "
          f"{total_drop} dropped for a person "
          f"({100 * total_drop / max(1, total_kept + total_drop):.0f}%)")
    print(f"written under {OUT_ROOT.relative_to(REPO_ROOT)}/<camera>/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["1760", "3333"])
    args = ap.parse_args()
    build(args.family)


if __name__ == "__main__":
    main()
