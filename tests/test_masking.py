from PIL import Image

from arsi_core.masking import MaskSpec


def spec(size=(400, 300)):
    return MaskSpec(name="t", image_size=list(size), zones=[
        {"id": "z1", "label": "window",
         "polygon": [[10, 10], [110, 10], [110, 60], [10, 60]]}])


def test_apply_blacks_out_zone_only():
    img = Image.new("RGB", (400, 300), (200, 200, 200))
    out = spec().apply(img)
    assert out.getpixel((50, 30)) == (0, 0, 0)          # inside zone
    assert out.getpixel((200, 200)) == (200, 200, 200)  # outside untouched
    assert img.getpixel((50, 30)) == (200, 200, 200)    # original untouched


def test_apply_scales_to_other_resolution():
    img = Image.new("RGB", (800, 600), (200, 200, 200))  # 2x the drawn size
    out = spec().apply(img)
    assert out.getpixel((100, 60)) == (0, 0, 0)          # scaled zone centre
    assert out.getpixel((240, 140)) == (200, 200, 200)   # just past scaled edge


def test_hash_stable_and_content_sensitive():
    a, b = spec(), spec()
    assert a.hash == b.hash
    b.zones[0]["polygon"][0] = [11, 10]
    assert a.hash != b.hash
    # label-only changes don't invalidate verdict caches
    c = spec()
    c.zones[0]["label"] = "renamed"
    assert a.hash == c.hash


def test_save_load_roundtrip(tmp_path):
    path = spec().save(tmp_path / "m.json")
    loaded = MaskSpec.load(path)
    assert loaded.to_dict() == spec().to_dict()
    assert loaded.hash == spec().hash
