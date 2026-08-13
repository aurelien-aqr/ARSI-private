"""Region proposal ("localizer") as a selectable stage, independent of the judge.

vlm_05 is two stages that fail for different reasons and are improved by
different work: something proposes candidate regions, then a VLM judges each
one. The judge has always been selectable (that is the model choice); this
module makes the proposal stage selectable too, so the two can be swept
independently instead of being one frozen half of a pipeline.

All localizers share vlm_05's contract exactly - (regions, info) with bboxes in
REFERENCE pixel space - and reuse its post-processing (person veto, salience
cap, merge). Choosing one changes which regions reach the VLM, nothing else.

Measured on the 29-case object-level benchmark (GLM x conservative, see
benchmark/README.md "DINOv2 feature gate"):

                       instances   strict     region     regions sent
                       localized      IoU  precision       to the VLM
    photo                  45/45    37/45      0.730              559
    photo+dino             45/45    37/45      0.815              243
    dino (z4, floor .08)   44/45    36/45        n/a              533

DEFAULT stays "photo": it is what every job in the history and every published
benchmark number was run with, and switching the default would silently change
what a re-run means. "photo+dino" carries the recommended badge instead.

Importing this module must stay cheap: torch is imported only when a DINOv2
localizer is actually selected. The dependency direction (arsi_core -> tools/)
mirrors adapters.py depending on the vlm_0x scripts: the scripts and probes are
the implementation, arsi_core is the contract the app and CLI talk to.
"""
import importlib.util
from pathlib import Path

from .errors import FrameError

DEFAULT = "photo"

LOCALIZERS = {
    "photo": {
        "name": "Pixel diff (multi-channel)",
        "summary": "The shipped change detector: photometric difference against "
                   "the clean reference at two thresholds plus an added-edge "
                   "channel, person veto, salience cap, merge.",
        "measured": "Localizes all 45 benchmark instances. Its noise is purely "
                    "photometric: a cross-session empty frame still yields 15-37 "
                    "candidate regions from lighting and white balance alone.",
        "needs": [],
    },
    "photo+dino": {
        "name": "Pixel diff + DINOv2 gate",
        "summary": "The pixel diff proposes the boxes, then DINOv2 patch "
                   "features veto the ones with no semantic change - a lighting "
                   "shift on an empty seat looks identical to the features.",
        "measured": "Same 45/45 instances and the same box quality as the pixel "
                    "diff, with region precision 0.730 -> 0.815 and 57% fewer "
                    "VLM calls (559 -> 243 on the benchmark). Recommended.",
        "needs": ["torch"],
        "recommended": True,
    },
    "dino": {
        "name": "DINOv2 features only (AnomalyDINO)",
        "summary": "No pixel comparison at all: each patch is scored by cosine "
                   "distance to the nominal reference patch at the same grid "
                   "position (AnomalyDINO, WACV 2025, specialised for a fixed "
                   "camera).",
        "measured": "Halves the region count on cross-session frames (105 -> 52) "
                    "where exposure differs, but boxes are quantised to the 24 px "
                    "patch grid, so strict-IoU box quality drops (37/45 -> 32/45). "
                    "Use it to see the feature signal on its own.",
        "needs": ["torch"],
    },
}


def names():
    return list(LOCALIZERS)


def catalog():
    """Registry for /api/localizers, with availability resolved."""
    out = []
    for key, spec in LOCALIZERS.items():
        ok, why = _availability(key)
        # `measured` is deliberately NOT shipped: the cards show the name and
        # the summary only. The numbers live in this file and in
        # benchmark/README.md, where they can be kept honest.
        out.append({"key": key, "name": spec["name"], "summary": spec["summary"],
                    "recommended": bool(spec.get("recommended")),
                    "available": ok, "unavailable_reason": why,
                    "first_use_download_mb": 0 if _weights_cached() or not spec["needs"] else 84})
    return out


def _weights_cached() -> bool:
    try:
        import torch
        ckpt = (Path(torch.hub.get_dir()) / "checkpoints"
                / "dinov2_vits14_reg4_pretrain.pth")
        return ckpt.exists()
    except Exception:
        return False


def _availability(name: str):
    spec = LOCALIZERS.get(name)
    if spec is None:
        return False, f"unknown localizer '{name}'"
    if "torch" in spec["needs"]:
        if importlib.util.find_spec("torch") is None:
            return False, ("PyTorch is not installed in this environment "
                           "(pip install torch), so DINOv2 features are "
                           "unavailable")
    return True, ""


def availability(name: str):
    """(ok, reason) for one localizer - what the API turns into a 409."""
    return _availability(name)


def check(name: str):
    """Validate a choice before a job starts. Raises FrameError (the SPEC's
    'input is unusable' class) so the API turns it into one clear message
    instead of failing every frame in turn."""
    ok, why = _availability(name)
    if not ok:
        raise FrameError(why)
    return name


def warmup(name: str):
    """Load whatever the localizer needs, once, before the frame loop - a failed
    weight download must fail the job with one message, not N frame errors."""
    if name in ("photo+dino", "dino"):
        try:
            _dino()._load_model()
        except Exception as exc:
            raise FrameError(
                f"cannot load the DINOv2 backbone for localizer '{name}': "
                f"{type(exc).__name__}: {exc}. The first use downloads ~84 MB "
                f"from dl.fbaipublicfiles.com; pick the 'photo' localizer to run "
                f"offline.") from exc


def _dino():
    import tools.dino_localizer as dl
    return dl


def localize(name: str, module, reference: str, inspection: str, params: dict = None):
    """Run the chosen localizer. `module` is the vlm_05 module as configured by
    the caller (thresholds/person filter overrides already applied), so every
    localizer honours the job's advanced params.

    Note on the verdict cache: the key is (reference, image, bbox, model,
    prompt, mask) and deliberately does NOT include the localizer. A box is a
    box - if two localizers propose identical coordinates the crop pair the VLM
    sees is identical, so the cache hit is correct and makes localizer A/Bs
    nearly free.
    """
    params = params or {}
    if name == "photo":
        regions, info = module.localize(reference, inspection)
        info["localizer"] = name
        return regions, info

    dl = _dino()
    if name == "dino":
        regions, info = dl.localize(
            reference, inspection,
            z_thr=float(params.get("DINO_Z", dl.Z_THRESHOLD)),
            abs_floor=float(params.get("DINO_FLOOR", dl.ABS_FLOOR)))
    elif name == "photo+dino":
        regions, info = dl.localize_gated(
            reference, inspection,
            gate=float(params.get("DINO_GATE", dl.GATE_THRESHOLD)))
    else:
        raise FrameError(f"unknown localizer '{name}'")
    info["localizer"] = name
    return regions, info
