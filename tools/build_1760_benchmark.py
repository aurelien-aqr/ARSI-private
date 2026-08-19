#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - tools/build_1760_benchmark.py
#  Builds the 1760 cameras of benchmark/datasets/ground_truth.json and the masked
#  frames they point at, from data/videos/1760/1760-camNN.mp4. It MERGES into the
#  one benchmark: every case on a 1760 camera is replaced, the rest is left alone.
#  Same contract as tools/build_39T_benchmark.py - read that one first.
#
#  WHY these frames. Specificity - the rate at which an empty tram raises an
#  alarm - is the number that decides deployment, and it rested on 12 clean 1762
#  frames and 7 clean 39T ones. These are 18 clean frames of a THIRD tram
#  (DPO-1760, 04/08/2026 09:00-09:10, a moving run of the line) on three interior
#  cameras, each paired with a reference from its own run 50-500 s earlier: a
#  negative here means clean under a visibly different sun. No anomalous case on
#  purpose - this measures false alarms only.
#
#  HOW THEY WERE CHOSEN (2026-08-19). One frame per camera every 50 s was laid
#  out as a 12-up strip per camera and read by eye. cam04 at t=20 s was rejected
#  (a passenger is on the phone in it); every frame kept here was seen. The three
#  cameras are the interior views with the most different framings; the five that
#  point at the door and the ground (cam01/02/03/11/12) are not usable.
#
#  MASKS come from data/masks_labelme/1760/1760-camNN.json - the same place and
#  the same format build_39T_benchmark.py reads, hand-drawn, ~26 polygons per
#  camera. On a tram in motion the world outside changes completely between two
#  frames, so an unmasked window is a guaranteed change region: the first version
#  of these frames was written UNMASKED by mistake, and the second through the
#  coarse 5-zone masks under data/app/masks/ - neither is the reference.
#
#  Run:  venv/bin/python tools/build_1760_benchmark.py
# =============================================================================

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from arsi_core import benchmarks                                # noqa: E402
from arsi_core.masking import MaskSpec                          # noqa: E402
from PIL import Image                                           # noqa: E402

VIDEOS = REPO_ROOT / "data" / "videos" / "1760"
MASKS = REPO_ROOT / "data" / "masks_labelme" / "1760"
OUT_DIR = REPO_ROOT / "data" / "benchmark_1760"

CAMS = ["cam04", "cam13", "cam06"]
T_REF = 70                                    # seconds into the run
T_CLEAN = [120, 220, 320, 420, 520, 570]      # spread over the ten minutes

_masks = {}


def mask_for(cam: str) -> MaskSpec:
    if cam not in _masks:
        path = MASKS / f"1760-{cam}.json"
        _masks[cam] = MaskSpec.from_labelme(json.loads(path.read_text()),
                                            name=f"1760-{cam}")
    return _masks[cam]


def frame(cam: str, t: int) -> str:
    """Extract, mask, store. Returns the repo-relative path for the GT file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / f"1760-{cam}_t{t:03d}.jpg"
    if not dst.exists():
        raw = OUT_DIR / f"_raw_{dst.name}"
        subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i",
                        str(VIDEOS / f"1760-{cam}.mp4"),
                        "-frames:v", "1", "-q:v", "2", "-y", str(raw)], check=True)
        # masked HERE, once: every consumer (benchmark, Studio, reports) must see
        # the same pixels the pipeline compares, windows blacked out included
        mask_for(cam).apply(Image.open(raw).convert("RGB")).save(dst)
        raw.unlink()
    return f"data/benchmark_1760/{dst.name}"


def build():
    refs = {f"1760-{c}": frame(c, T_REF) for c in CAMS}
    cases = [{
        "id": f"1760_{cam}_t{t:03d}",
        "image": frame(cam, t),
        "reference": f"1760-{cam}",
        "source": "real",
        "has_anomaly": False,
        "types": [],
        "note": f"clean, SAME run as the reference, {t - T_REF}s later "
                f"(tram circulating, the light through the windows moves)",
        "instances": [],
    } for cam in CAMS for t in T_CLEAN]

    ds_id, gt = benchmarks.load()
    kept = [c for c in gt["cases"] if not c.get("reference", "").startswith("1760-")]
    gt["references"].update(refs)
    gt["cases"] = kept + cases
    benchmarks.save(ds_id, gt)
    print(f"{benchmarks.dataset_path(ds_id).relative_to(REPO_ROOT)}: "
          f"{len(cases)} 1760 cases (0 anomalous, {len(cases)} clean), "
          f"{len(CAMS)} cameras, merged into {len(gt['cases'])} cases total")
    print(f"frames in {OUT_DIR.relative_to(REPO_ROOT)}: "
          f"{len(list(OUT_DIR.glob('*.jpg')))}")


if __name__ == "__main__":
    build()
