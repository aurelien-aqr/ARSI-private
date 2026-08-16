"""Dinomaly localizer: the contract and the failure mode, not the accuracy.

Accuracy lives in benchmark/eval_localization.py (--variants dinomaly,dgate),
which needs a trained checkpoint. What must hold with or without one is that the
reference-free localizer speaks vlm_05's language, and that a missing checkpoint
fails with a sentence that says how to make one.
"""
import numpy as np
import pytest

import tools.dinomaly as dm
import tools.dino_localizer as dl
import vlm_05_reference_diff as m


def test_missing_checkpoint_says_how_to_train_one():
    with pytest.raises(FileNotFoundError) as exc:
        dm.load("no_such_camera")
    msg = str(exc.value)
    assert "dinomaly_train.py" in msg and "no_such_camera" in msg


def test_patch_map_becomes_vlm05_regions(monkeypatch):
    """Every feature localizer goes through the SHIPPED post-processing. A patch
    blob must come out as a region in REFERENCE pixel space, so the boxes of
    dino, dinomaly and photo are all in the one space the judge crops from."""
    monkeypatch.setattr(m, "PERSON_FILTER", False)     # no YOLO in a unit test
    gh, gw = 45, 80
    over = np.zeros((gh, gw), dtype=bool)
    over[20:24, 30:36] = True                          # one blob, mid-frame
    salience = over.astype(np.float32) * 10.0
    size = (1920, 1080)

    regions, info = dl.regions_from_patches(over, salience, "", size, "dinomaly",
                                            {}, 0.0)

    assert info["patches_over"] == 24 and info["total"] == len(regions) >= 1
    x0, y0, x1, y1 = regions[0]["bbox"]
    assert regions[0]["channel"] == "dinomaly"
    assert 0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]
    # the blob sits at 30-36/80 of the width and 20-24/45 of the height; the
    # box must land there, not merely somewhere inside the frame
    assert 600 < x0 < 800 and 1000 > y0 > 400


def test_hard_mining_optimises_the_worst_points_only():
    """The loss must ignore what the decoder already reconstructs: otherwise a
    perfect reconstruction of an anomaly is rewarded like everything else, which
    is exactly how a reconstruction detector stops detecting."""
    import torch
    t = torch.nn.functional.normalize(torch.randn(1, 100, 8), dim=-1)
    p = t.clone()
    p[0, :5] = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)  # 5 bad

    all_points = float(dm.hard_cosine_loss([p], [t], q=0.0))
    hardest = float(dm.hard_cosine_loss([p], [t], q=0.9))
    assert hardest > all_points * 5      # dominated by the 5 broken points
