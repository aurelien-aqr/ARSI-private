#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - tools/build_3333_benchmark.py
#
#  "3333"* is a PLACEHOLDER tram id, not a fleet number: the real vehicle number
#  of the 2026-08-11 capture is unknown. It was filed under "39T" until
#  2026-08-21, which was worse - 39T is the Skoda type, and tram 1760 is a 39T
#  too. Rename both the folders and this file the day the number turns up.
#  The canonical wording of the note lives in arsi_core.TRAM_ID_NOTE.
#
#  Builds the 3333 cameras of benchmark/datasets/ground_truth.json and the masked
#  frames they point at, from the 3333 multi-camera footage
#  (data/videos/3333/<moment>/3333-camNN.mp4). It MERGES into the one benchmark:
#  every case on a 3333 camera is replaced, the rest of the ground truth is left
#  alone.
#
#  WHY these cameras. Everything measured until 2026-08-16 came from ONE camera of
#  tram 1762, filmed in July: 29 cases, one viewpoint, one tram. Every verdict in
#  docs/DECISIONS.md rests on it. These 4 viewpoints are a different tram, and
#  their negatives are frames of OTHER runs of the line, so "clean" here means
#  clean under a different sun and a different position on the track - the
#  deployment case.
#
#  THE LABELS ARE IN THIS FILE (CASES below), on purpose: the frames are
#  reproducible from the videos, so the only thing worth versioning by hand is
#  the human judgement. Re-running this script rebuilds everything.
#
#  HOW THEY WERE MADE (2026-08-16, by Claude, from the footage alone):
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
#  Run:  python tools/build_3333_benchmark.py
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

VIDEOS = REPO_ROOT / "data" / "videos" / "3333"
MASKS = REPO_ROOT / "data" / "masks_labelme" / "3333"
OUT_DIR = REPO_ROOT / "data" / "benchmark_3333"

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
        path = MASKS / f"3333-{cam}.json"
        _masks[cam] = MaskSpec.from_labelme(json.loads(path.read_text()),
                                            name=f"3333-{cam}")
    return _masks[cam]


def moment_dir(moment: str) -> Path:
    hits = sorted(VIDEOS.glob(f"*_{moment}"))
    if not hits:
        raise FileNotFoundError(f"no 3333 moment ending in {moment} under {VIDEOS}")
    return hits[0]


def frame(cam: str, moment: str, t: int = T_CASE) -> str:
    """Extract, mask, store. Returns the repo-relative path for the GT file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / f"3333-{cam}_{moment}_t{t}.jpg"
    if not dst.exists():
        raw = OUT_DIR / f"_raw_{dst.name}"
        subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i",
                        str(moment_dir(moment) / f"3333-{cam}.mp4"),
                        "-frames:v", "1", "-y", str(raw)], check=True)
        # masked HERE, once: every consumer (benchmark, Studio, reports) must see
        # the same pixels the pipeline compares, windows blacked out included
        mask_for(cam).apply(Image.open(raw).convert("RGB")).save(dst)
        raw.unlink()
    return f"data/benchmark_3333/{dst.name}"


def build():
    cams = sorted({c for c, _ in CASES} | {c for c, _ in CLEAN})
    refs = {f"3333-{c}": frame(c, REF_MOMENT) for c in cams}
    cases = []
    for (cam, moment), insts in sorted(CASES.items()):
        cases.append({
            "id": f"3333_{cam}_{moment.replace('-', '')}",
            "image": frame(cam, moment), "reference": f"3333-{cam}",
            "source": "real", "has_anomaly": True,
            "types": sorted({t for t, _, _ in insts}),
            "note": f"3333, {cam}, moment {moment}, t={T_CASE}s (tram circulating)",
            "instances": [{"type": t, "label": lab, "bbox": b} for t, lab, b in insts],
        })
    for cam, moment in CLEAN:
        cases.append({
            "id": f"3333_{cam}_{moment.replace('-', '')}_clean",
            "image": frame(cam, moment), "reference": f"3333-{cam}",
            "source": "real", "has_anomaly": False, "types": [], "instances": [],
            "note": "clean, a DIFFERENT run of the line than the reference",
        })
    for cam in cams:
        cases.append({
            "id": f"3333_{cam}_ref_t{T_SECOND_NEGATIVE}_clean",
            "image": frame(cam, REF_MOMENT, T_SECOND_NEGATIVE),
            "reference": f"3333-{cam}", "source": "real", "has_anomaly": False,
            "types": [], "instances": [],
            "note": f"clean, SAME run as the reference, {T_SECOND_NEGATIVE - T_CASE}s "
                    f"later - the cheapest possible negative",
        })
    # There is ONE benchmark, so this builder MERGES its cameras into it rather
    # than writing a file of its own: it replaces every case whose reference is a
    # 3333 camera and leaves the rest of the ground truth alone. Through
    # benchmarks.save, which validates first and snapshots the current file into
    # benchmark/datasets/.backups/ - re-running this REPLACES whatever was
    # corrected by hand on these cameras, and the backup is what gets it back.
    ds_id, gt = benchmarks.load()
    kept = [c for c in gt["cases"] if not c.get("reference", "").startswith("3333-")]
    gt["references"].update(refs)
    gt["cases"] = kept + cases
    benchmarks.save(ds_id, gt)
    n_inst = sum(len(c["instances"]) for c in cases)
    n_anom = sum(1 for c in cases if c["has_anomaly"])
    print(f"{benchmarks.dataset_path(ds_id).relative_to(REPO_ROOT)}: "
          f"{len(cases)} 3333 cases ({n_anom} anomalous, {len(cases) - n_anom} "
          f"clean), {n_inst} instances, {len(cams)} cameras, merged into "
          f"{len(gt['cases'])} cases total")
    print(f"frames in {OUT_DIR.relative_to(REPO_ROOT)}: "
          f"{len(list(OUT_DIR.glob('*.jpg')))}")


#: The provenance of these cameras is written in the merged dataset's `_about`
#: (benchmark/datasets/ground_truth.json) and is deliberately NOT rewritten
#: here: a rebuild replaces the 3333 cases, not the description of the whole
#: benchmark.


if __name__ == "__main__":
    build()
