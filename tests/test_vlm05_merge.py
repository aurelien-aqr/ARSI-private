"""Pre-VLM region merge in vlm_05: it must absorb fragments of one object without
ever chaining neighbours into a frame-sized box.

The blob case is the one worth guarding. It passes every lenient check — a box
covering the frame "overlaps" every ground-truth instance — so only an explicit
assertion on the merged box size catches a regression here. See the MERGE_*
comment block in vlm_05_reference_diff.py for the measured sweep.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import vlm_05_reference_diff as m   # noqa: E402


def region(x0, y0, x1, y1, fill=1.0, salience=1.0):
    """A region whose changed-pixel area is `fill` of its box."""
    return {"bbox": [x0, y0, x1, y1], "salience": salience,
            "area": int((x1 - x0) * (y1 - y0) * fill)}


def test_merges_a_fragment_into_its_object():
    # a phone box and the sliver of seat edge 10 px to its right
    regions = [region(100, 100, 200, 160), region(210, 100, 230, 160)]
    out = m.merge_regions(regions, gap=24, min_fill=0.5)
    assert len(out) == 1
    assert out[0]["bbox"] == [100, 100, 230, 160]
    # the merged area is the CHANGED pixels of both, not the box around them
    assert out[0]["area"] == sum(r["area"] for r in regions)


def test_keeps_distant_objects_apart():
    regions = [region(100, 100, 200, 200), region(900, 700, 1000, 800)]
    assert len(m.merge_regions(regions, gap=24, min_fill=0.5)) == 2


def test_fill_guard_refuses_a_sparse_union():
    # two thin bars far apart on both axes but within gap on neither -> the union
    # box would be mostly empty, which is how a chain starts
    regions = [region(0, 0, 400, 20), region(0, 300, 400, 320)]
    assert len(m.merge_regions(regions, gap=400, min_fill=0.5)) == 2


def test_never_chains_into_a_frame_sized_box():
    # 12 sparse regions strung across a 1920x1080 frame, each within gap of the
    # next: a naive transitive merge swallows the whole frame (the real_f0205
    # failure). Every merged box must stay far below the frame.
    frame = 1920 * 1080
    regions = [region(i * 150, 400, i * 150 + 130, 500, fill=0.35) for i in range(12)]
    out = m.merge_regions(regions, gap=24, min_fill=0.5)
    biggest = max((b[2] - b[0]) * (b[3] - b[1])
                  for b in (r["bbox"] for r in out))
    assert biggest < frame * 0.25


def test_merged_region_carries_the_strongest_salience():
    # the cap ranks by salience: a merged region must not be diluted by its
    # dimmer neighbour, or a bright small object drops below a lighting blob
    regions = [region(100, 100, 200, 160, salience=9.0),
               region(210, 100, 230, 160, salience=0.5)]
    out = m.merge_regions(regions, gap=24, min_fill=0.5)
    assert out[0]["salience"] == 9.0


def test_shipped_defaults_are_the_swept_values():
    # the sweep that chose these is recorded in the MERGE_* comment block; a
    # silent edit here changes every box the judge sees
    assert (m.MERGE_REGIONS, m.MERGE_GAP, m.MERGE_MIN_FILL) == (True, 24, 0.50)
