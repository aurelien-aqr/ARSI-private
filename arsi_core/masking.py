"""Camera-wide masks: named JSON presets of polygons drawn once on any frame
of a fixed camera and applied to every frame (docs/SPEC.md "Masking").

The existing data/masked/ frames follow exactly this convention (pure-black
window contours); this module reproduces that external step inside the app.
"""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from . import APP_DATA

MASKS_DIR = APP_DATA / "masks"


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
