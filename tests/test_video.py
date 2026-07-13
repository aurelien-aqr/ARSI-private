import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from arsi_core.video import extract_frames, probe  # noqa: E402


@pytest.fixture
def tiny_video(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    if not w.isOpened():
        pytest.skip("no MJPG codec available")
    for i in range(20):
        frame = np.full((48, 64, 3), i * 10, dtype=np.uint8)
        w.write(frame)
    w.release()
    return path


def test_probe(tiny_video):
    info = probe(tiny_video)
    assert info["frame_count"] == 20
    assert info["fps"] == pytest.approx(10.0)
    assert (info["width"], info["height"]) == (64, 48)


def test_extract_every_n(tiny_video, tmp_path):
    meta = extract_frames(tiny_video, tmp_path / "out", every_n=6)
    assert [f["index"] for f in meta["frames"]] == [0, 6, 12, 18]
    assert (tmp_path / "out" / "f0006.jpg").exists()
    assert (tmp_path / "out" / "meta.json").exists()


def test_extract_every_s_with_trim(tiny_video, tmp_path):
    # 10 fps, 1 frame per 0.5 s from 0.5 s to 1.5 s -> indices 5, 10, 15
    meta = extract_frames(tiny_video, tmp_path / "out", every_s=0.5,
                          start_s=0.5, end_s=1.5)
    assert [f["index"] for f in meta["frames"]] == [5, 10, 15]


def test_requires_exactly_one_step_param(tiny_video, tmp_path):
    with pytest.raises(ValueError):
        extract_frames(tiny_video, tmp_path / "out")
    with pytest.raises(ValueError):
        extract_frames(tiny_video, tmp_path / "out", every_n=2, every_s=1.0)
