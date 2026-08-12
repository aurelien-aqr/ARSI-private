"""Camera-wide masks: named JSON presets of polygons drawn once on any frame
of a fixed camera and applied to every frame (docs/SPEC.md "Masking").

The existing data/masked/ frames follow exactly this convention (pure-black
window contours); this module reproduces that external step inside the app.

LabelMe interop: the team annotates in LabelMe, whose format carries the same
information under different keys (`shapes[].points` vs `zones[].polygon`, both
in image pixels). `from_labelme`/`to_labelme` convert both ways so a mask can
be drawn in either tool; the app keeps ownership of applying it and of the
verdict-cache hash, which LabelMe knows nothing about.
"""
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from . import APP_DATA
from .errors import ArsiError

MASKS_DIR = APP_DATA / "masks"

LABELME_VERSION = "5.5.0"
#: LabelMe shape types that describe a closed area, so can become a zone.
#: Open shapes (line, linestrip, point) enclose nothing and are skipped.
LABELME_AREA_SHAPES = ("polygon", "rectangle", "circle")


def _rectangle_polygon(points):
    """LabelMe stores a rectangle as two opposite corners."""
    (x0, y0), (x1, y1) = points[:2]
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _circle_polygon(points, segments=32):
    """LabelMe stores a circle as centre + a point on the rim."""
    (cx, cy), (ex, ey) = points[:2]
    r = math.hypot(ex - cx, ey - cy)
    return [[cx + r * math.cos(2 * math.pi * i / segments),
             cy + r * math.sin(2 * math.pi * i / segments)]
            for i in range(segments)]


def _coord(v):
    """Integral values stay ints: the verdict-cache hash serialises the
    polygons, so 10 and 10.0 would hash differently and a mask round-tripped
    through LabelMe would needlessly invalidate every cached verdict."""
    v = round(float(v), 2)
    return int(v) if v == int(v) else v


def _shape_polygon(shape):
    """Polygon for one LabelMe shape, or None when it cannot become a zone."""
    kind = shape.get("shape_type", "polygon")
    pts = [[float(x), float(y)] for x, y in shape.get("points", [])]
    if kind == "polygon":
        return pts if len(pts) >= 3 else None
    if kind == "rectangle":
        return _rectangle_polygon(pts) if len(pts) >= 2 else None
    if kind == "circle":
        return _circle_polygon(pts) if len(pts) >= 2 else None
    return None


def labelme_skipped(data: dict):
    """Shapes the conversion drops, as ``[{label, shape_type, reason}]``.

    Reported to the user rather than silently ignored: a mask that lost a zone
    on import would black out less than the annotator intended."""
    out = []
    for shape in data.get("shapes", []):
        if _shape_polygon(shape) is not None:
            continue
        kind = shape.get("shape_type", "polygon")
        reason = ("open shape, encloses no area" if kind not in LABELME_AREA_SHAPES
                  else "not enough points")
        out.append({"label": shape.get("label", ""), "shape_type": kind,
                    "reason": reason})
    return out


@dataclass
class MaskSpec:
    name: str
    image_size: list                    # [w, h] of the frame the zones were drawn on
    zones: list = field(default_factory=list)   # [{"id", "label", "polygon": [[x, y], ...]}]
    camera: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "MaskSpec":
        return cls(name=d["name"], image_size=list(d["image_size"]),
                   zones=list(d.get("zones", [])), camera=d.get("camera", ""))

    @classmethod
    def load(cls, path) -> "MaskSpec":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def from_labelme(cls, data: dict, name: str, camera: str = "") -> "MaskSpec":
        """Build a mask from a LabelMe annotation file.

        LabelMe points are already in image pixels, like `zones[].polygon`, so
        only the keys change. Shapes that enclose no area are dropped — call
        `labelme_skipped()` to tell the user which."""
        try:
            w, h = int(data["imageWidth"]), int(data["imageHeight"])
        except (KeyError, TypeError, ValueError):
            raise ArsiError("not a LabelMe file: missing imageWidth/imageHeight")
        if w <= 0 or h <= 0:
            raise ArsiError(f"invalid LabelMe image size: {w}x{h}")

        zones = []
        for shape in data.get("shapes", []):
            poly = _shape_polygon(shape)
            if poly is None:
                continue
            idx = len(zones) + 1
            zones.append({"id": str(idx),
                          "label": shape.get("label") or f"Zone {idx}",
                          "polygon": [[_coord(x), _coord(y)] for x, y in poly]})
        if not zones:
            raise ArsiError("no usable zone in this LabelMe file "
                            "(polygon, rectangle or circle shapes are required)")
        return cls(name=name, image_size=[w, h], zones=zones, camera=camera)

    def to_labelme(self, image_path: str = "") -> dict:
        """Export as a LabelMe annotation file, so the team can reopen and edit
        a Studio mask in their usual tool. Every zone becomes a polygon —
        rectangles and circles imported earlier stay expanded, which LabelMe
        reads back fine."""
        return {
            "version": LABELME_VERSION,
            "flags": {},
            "shapes": [{"label": z.get("label") or f"Zone {i + 1}",
                        "points": [[float(x), float(y)] for x, y in z["polygon"]],
                        "group_id": None,
                        "description": "",
                        "shape_type": "polygon",
                        "flags": {},
                        "mask": None}
                       for i, z in enumerate(self.zones)],
            "imagePath": image_path,
            "imageData": None,
            "imageHeight": self.image_size[1],
            "imageWidth": self.image_size[0],
        }

    def to_dict(self) -> dict:
        return {"name": self.name, "camera": self.camera,
                "image_size": self.image_size, "zones": self.zones}

    def save(self, path=None) -> Path:
        path = Path(path) if path else MASKS_DIR / f"{self.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1, ensure_ascii=False)
        return path

    @property
    def hash(self) -> str:
        """Stable content hash — joins the vlm_05 verdict-cache key, so editing
        a mask invalidates cached verdicts exactly like a prompt change."""
        canon = json.dumps({"size": self.image_size,
                            "polys": sorted(z["polygon"] for z in self.zones)},
                           separators=(",", ":"))
        return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:12]

    def apply(self, image: Image.Image) -> Image.Image:
        """Return a copy with every zone filled pure black. Zones are scaled
        when the image size differs from the size they were drawn on (same
        camera, different export resolution)."""
        out = image.copy()
        draw = ImageDraw.Draw(out)
        sx = image.width / self.image_size[0]
        sy = image.height / self.image_size[1]
        for zone in self.zones:
            poly = [(x * sx, y * sy) for x, y in zone["polygon"]]
            if len(poly) >= 3:
                draw.polygon(poly, fill=(0, 0, 0))
        return out

    def apply_file(self, src, dst) -> Path:
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as img:
            self.apply(img.convert("RGB")).save(dst)
        return dst


def list_masks():
    if not MASKS_DIR.exists():
        return []
    return [MaskSpec.load(p) for p in sorted(MASKS_DIR.glob("*.json"))]
