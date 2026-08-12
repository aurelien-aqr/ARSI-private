import pytest
from PIL import Image

from arsi_core.errors import ArsiError
from arsi_core.masking import MaskSpec, labelme_skipped
from arsi_core.video import camera_slug


def labelme(shapes, w=400, h=300):
    return {"version": "5.5.0", "flags": {}, "shapes": shapes,
            "imagePath": "f0001.jpg", "imageData": None,
            "imageWidth": w, "imageHeight": h}


def shape(kind, points, label="window"):
    return {"label": label, "points": points, "group_id": None,
            "shape_type": kind, "flags": {}}


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


# ------------------------------------------------------------ LabelMe interop

def test_from_labelme_polygon_keeps_pixel_coords():
    doc = labelme([shape("polygon", [[10, 10], [110, 10], [110, 60], [10, 60]])])
    m = MaskSpec.from_labelme(doc, name="imported", camera="1760-cam05")
    assert m.image_size == [400, 300]
    assert m.camera == "1760-cam05"
    assert m.zones[0]["label"] == "window"
    assert m.zones[0]["polygon"] == [[10, 10], [110, 10], [110, 60], [10, 60]]
    # same geometry as a hand-drawn mask => same cache hash
    assert m.hash == spec().hash


def test_from_labelme_rectangle_becomes_four_corners():
    doc = labelme([shape("rectangle", [[10, 10], [110, 60]])])
    m = MaskSpec.from_labelme(doc, name="i")
    assert m.zones[0]["polygon"] == [[10, 10], [110, 10], [110, 60], [10, 60]]
    img = Image.new("RGB", (400, 300), (200, 200, 200))
    out = m.apply(img)
    assert out.getpixel((50, 30)) == (0, 0, 0)
    assert out.getpixel((200, 200)) == (200, 200, 200)


def test_from_labelme_circle_is_approximated_and_filled():
    doc = labelme([shape("circle", [[200, 150], [250, 150]])])
    m = MaskSpec.from_labelme(doc, name="i")
    assert len(m.zones[0]["polygon"]) == 32
    out = m.apply(Image.new("RGB", (400, 300), (200, 200, 200)))
    assert out.getpixel((200, 150)) == (0, 0, 0)      # centre
    assert out.getpixel((200, 60)) == (200, 200, 200)  # well outside the radius


def test_open_shapes_are_skipped_and_reported():
    doc = labelme([shape("polygon", [[0, 0], [10, 0], [10, 10]]),
                   shape("linestrip", [[0, 0], [5, 5]], label="rail"),
                   shape("point", [[3, 3]], label="dot")])
    m = MaskSpec.from_labelme(doc, name="i")
    assert len(m.zones) == 1
    kinds = {s["shape_type"] for s in labelme_skipped(doc)}
    assert kinds == {"linestrip", "point"}


def test_from_labelme_rejects_unusable_files():
    with pytest.raises(ArsiError):                       # not a LabelMe file
        MaskSpec.from_labelme({"shapes": []}, name="i")
    with pytest.raises(ArsiError):                       # nothing enclosing area
        MaskSpec.from_labelme(labelme([shape("point", [[1, 1]])]), name="i")


def test_labelme_roundtrip_preserves_geometry():
    original = spec()
    back = MaskSpec.from_labelme(original.to_labelme("f0001.jpg"),
                                 name=original.name, camera=original.camera)
    assert back.image_size == original.image_size
    assert back.zones[0]["polygon"] == original.zones[0]["polygon"]
    assert back.hash == original.hash          # survives a trip through LabelMe


def test_to_labelme_shape_is_valid():
    doc = spec().to_labelme("f0001.jpg")
    assert doc["imageWidth"] == 400 and doc["imageHeight"] == 300
    assert doc["imagePath"] == "f0001.jpg"
    s = doc["shapes"][0]
    assert s["shape_type"] == "polygon" and s["label"] == "window"
    assert s["points"] == [[10.0, 10.0], [110.0, 10.0], [110.0, 60.0], [10.0, 60.0]]


# ------------------------------------------------------------ camera identity

def test_camera_slug_from_upload_names():
    assert camera_slug("1760-cam05.mp4") == "1760-cam05"
    assert camera_slug("tram 1762 (v1).MP4") == "tram_1762_v1"
    assert camera_slug("") == "camera"
    assert camera_slug("....mp4") == "camera"
