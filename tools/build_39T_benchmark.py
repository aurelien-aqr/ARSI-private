#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - tools/build_39T_benchmark.py
#  Builds benchmark/datasets/39T.json and the masked frames it points at,
#  from the 39T multi-camera footage (data/videos/39T/<moment>/39T-camNN.mp4).
#
#  WHY a second benchmark. Everything measured until 2026-08-16 came from ONE
#  camera of tram 1762, filmed in July: 29 cases, one viewpoint, one tram. Every
#  verdict in docs/DECISIONS.md rests on it. This one covers 4 viewpoints of a
#  different tram, and its negatives are frames of OTHER runs of the line, so
#  "clean" here means clean under a different sun and a different position on
#  the track - the deployment case.
#
#  THE LABELS ARE IN THIS FILE (CASES below), on purpose: the frames are
#  reproducible from the videos, so the only thing worth versioning by hand is
#  the human judgement. Re-running this script rebuilds everything.
#
#  HOW THEY WERE MADE (2026-08-16, by Claude, unreviewed by a human):
#    1. one frame per camera per moment, masked, laid out as a 6-up strip;
#    2. objects spotted by eye on the strip - NOT by running our own localizer,
#       which would have made the ground truth agree with the system under test;
#    3. every candidate confirmed by cropping the SAME rectangle out of the
#       reference and out of the inspection frame and looking at both. Two
#       candidates died there: a blue seat cover (present in both) and a lit
#       floor patch. The tram's own yellow validators fail the same test;
#    4. boxes read off a 100 px coordinate grid, generously, in reference pixel
#       space (1280x720).
#
#  ONE FRAME PER CASE, at t=60 s. The staged objects do not move during a run,
#  but the first seconds of several moments still show the staff placing them -
#  a fixed, late timestamp avoids labelling a scene that is still changing.
#
#  Run:  python tools/build_39T_benchmark.py
# =============================================================================

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from arsi_core import benchmarks                                # noqa: E402
from arsi_core.masking import MaskSpec                          # noqa: E402
from PIL import Image                                           # noqa: E402

VIDEOS = REPO_ROOT / "data" / "videos" / "39T"
MASKS = REPO_ROOT / "data" / "masks_labelme" / "39T"
OUT_DIR = REPO_ROOT / "data" / "benchmark_39T"
GT_OUT = REPO_ROOT / "benchmark" / "datasets" / "39T.json"

REF_MOMENT = "08-55-37"      # verified empty on all 15 cameras
T_CASE = 60                  # seconds into the video
T_SECOND_NEGATIVE = 120      # same run, 60 s later: does the reference age?

#  (camera, moment) -> [(type, label, [x0, y0, x1, y1]), ...]
CASES = {
    ("cam52", "08-46-37"): [("object", "purple bag on seat", [190, 660, 300, 720]),
                            ("object", "small white item on far seat", [525, 55, 580, 92])],
    ("cam52", "08-51-24"): [("object", "small dark item on seat", [460, 398, 515, 440])],
    ("cam53", "08-35-17"): [("object", "yellow bag on flip seat", [163, 198, 268, 270]),
                            ("object", "green item on rail", [893, 178, 942, 218])],
    ("cam53", "08-40-21"): [("object", "yellow bag on flip seat", [165, 198, 268, 272]),
                            ("object", "green item on rail", [895, 180, 940, 212])],
    ("cam53", "08-46-37"): [("object", "green item on flip seat", [172, 193, 235, 252])],
    ("cam53", "08-51-24"): [("object", "pink item hanging on pole", [443, 178, 492, 278])],
    ("cam54", "08-35-17"): [("litter", "plastic bag on floor", [125, 600, 260, 700])],
    ("cam54", "08-40-21"): [("litter", "plastic bag on floor", [128, 598, 278, 698])],
    ("cam54", "08-46-37"): [("object", "cloth on seat", [208, 122, 340, 182]),
                            ("object", "laptop on far-left seat", [0, 290, 120, 360])],
    ("cam54", "08-51-24"): [("object", "flat item on seat", [200, 145, 340, 210]),
                            ("object", "green cloth on seat", [80, 260, 150, 315]),
                            ("litter", "plastic bag on floor", [420, 600, 590, 710])],
    ("cam55", "08-35-17"): [("object", "grey laptop on seat", [412, 190, 520, 300]),
                            ("litter", "plastic bag on floor", [365, 580, 480, 705])],
    ("cam55", "08-40-21"): [("object", "grey laptop on seat", [418, 188, 516, 302]),
                            ("litter", "plastic bag on floor", [363, 578, 480, 706])],
    ("cam55", "08-46-37"): [("object", "white laptop on seat", [443, 273, 580, 348]),
                            ("object", "cloth on right seat", [838, 250, 975, 378])],
    ("cam55", "08-51-24"): [("object", "green item on seat", [545, 272, 610, 398]),
                            ("object", "white laptop on seat", [805, 288, 955, 362])],
}

#  Moments verified empty for that camera - the cross-session negatives.
CLEAN = [("cam53", "08-59-54"), ("cam54", "08-59-54"), ("cam55", "08-59-54")]

#  Not labelled in this pass, and deliberately absent rather than called clean:
#  cam52 at 08-35-17 / 08-40-21 (small distant items I could not confirm) and
#  cam52 at 08-59-54 (two passengers, one bag at their feet - a person case, not
#  a left-object case). The other 11 cameras of this tram are untouched.

_masks = {}


def mask_for(cam: str) -> MaskSpec:
    if cam not in _masks:
        path = MASKS / f"39T-{cam}.json"
        _masks[cam] = MaskSpec.from_labelme(json.loads(path.read_text()),
                                            name=f"39T-{cam}")
    return _masks[cam]


def moment_dir(moment: str) -> Path:
    hits = sorted(VIDEOS.glob(f"*_{moment}"))
    if not hits:
        raise FileNotFoundError(f"no 39T moment ending in {moment} under {VIDEOS}")
    return hits[0]


def frame(cam: str, moment: str, t: int = T_CASE) -> str:
    """Extract, mask, store. Returns the repo-relative path for the GT file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / f"39T-{cam}_{moment}_t{t}.jpg"
    if not dst.exists():
        raw = OUT_DIR / f"_raw_{dst.name}"
        subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i",
                        str(moment_dir(moment) / f"39T-{cam}.mp4"),
                        "-frames:v", "1", "-y", str(raw)], check=True)
        # masked HERE, once: every consumer (benchmark, Studio, reports) must see
        # the same pixels the pipeline compares, windows blacked out included
        mask_for(cam).apply(Image.open(raw).convert("RGB")).save(dst)
        raw.unlink()
    return f"data/benchmark_39T/{dst.name}"


def build():
    cams = sorted({c for c, _ in CASES} | {c for c, _ in CLEAN})
    refs = {f"39T-{c}": frame(c, REF_MOMENT) for c in cams}
    cases = []
    for (cam, moment), insts in sorted(CASES.items()):
        cases.append({
            "id": f"39T_{cam}_{moment.replace('-', '')}",
            "image": frame(cam, moment), "reference": f"39T-{cam}",
            "source": "real", "has_anomaly": True,
            "types": sorted({t for t, _, _ in insts}),
            "note": f"39T, {cam}, moment {moment}, t={T_CASE}s (tram circulating)",
            "instances": [{"type": t, "label": lab, "bbox": b} for t, lab, b in insts],
        })
    for cam, moment in CLEAN:
        cases.append({
            "id": f"39T_{cam}_{moment.replace('-', '')}_clean",
            "image": frame(cam, moment), "reference": f"39T-{cam}",
            "source": "real", "has_anomaly": False, "types": [], "instances": [],
            "note": "clean, a DIFFERENT run of the line than the reference",
        })
    for cam in cams:
        cases.append({
            "id": f"39T_{cam}_ref_t{T_SECOND_NEGATIVE}_clean",
            "image": frame(cam, REF_MOMENT, T_SECOND_NEGATIVE),
            "reference": f"39T-{cam}", "source": "real", "has_anomaly": False,
            "types": [], "instances": [],
            "note": f"clean, SAME run as the reference, {T_SECOND_NEGATIVE - T_CASE}s "
                    f"later - the cheapest possible negative",
        })
    gt = {"_about": ABOUT, "name": "tram 39T — four cameras, 2026-08-11 capture",
          "references": refs, "cases": cases}
    # Through benchmarks.save: it validates before writing and snapshots the
    # current file into benchmark/datasets/.backups/ first. That matters now that
    # the labels are corrected by hand in the Studio — re-running this builder
    # REPLACES those corrections, and the backup is what gets them back.
    benchmarks.save("39T", gt)
    n_inst = sum(len(c["instances"]) for c in cases)
    n_anom = sum(1 for c in cases if c["has_anomaly"])
    print(f"{GT_OUT.relative_to(REPO_ROOT)}: {len(cases)} cases "
          f"({n_anom} anomalous, {len(cases) - n_anom} clean), {n_inst} instances, "
          f"{len(cams)} cameras")
    print(f"frames in {OUT_DIR.relative_to(REPO_ROOT)}: "
          f"{len(list(OUT_DIR.glob('*.jpg')))}")


ABOUT = (
    "Ground truth for the 39T multi-camera footage (recorded 2026-08-11), built "
    "2026-08-16 by tools/build_39T_benchmark.py. A SECOND protocol next to "
    "benchmark/datasets/tram1762.json, which covers one camera of tram 1762 only. Four viewpoints "
    "of tram 39T (cam52-55); every image is the t=60 s frame of its video, masked "
    "with that camera's own mask, so it is exactly what the pipeline sees. Each "
    "camera has its own reference: itself during moment 08-55-37, verified empty on "
    "all 15 interior cameras. The other moments are separate runs of the line, so a "
    "negative here is a cross-session negative (different sun, different position on "
    "the track). instances = object-level ground truth in reference pixel space "
    "(1280x720), boxes intentionally generous; people are NOT instances. "
    "Labelled by Claude and NOT yet reviewed by a human: objects were found by eye, "
    "never by running our own localizer, and each one was confirmed by comparing the "
    "same crop in the reference and in the inspection frame - a test that rejected "
    "two candidates (a seat cover present in both frames, and a sunlit floor patch). "
    "Treat it as a draft to correct in the Studio's Benchmark screen.")


if __name__ == "__main__":
    build()
