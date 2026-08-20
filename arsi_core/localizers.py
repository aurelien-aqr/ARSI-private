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

THAT TABLE IS ONE CAMERA. Re-measured 2026-08-19 on all 68 cases (3 trams,
9 views, benchmark/README.md "Three localizer families on THREE trams"), the
ranking inverts, and it inverts on the column the table above cannot see:

                       instances   strict     regions sent
                       localized      IoU       to the VLM
    photo                  65/73    42/73             1192
    photo+dino             65/73    42/73              727
    dino (z4, floor .08)   69/73    58/73             1228

On tram 39T the pixel diff scores 5/26 strict against "dino"'s 19/26 - it boxes
almost the whole frame - and no gate can repair that, because a gate deletes
regions and never redraws one. "photo+dino" remains the right pick on the 1762
camera and is a pure cost lever everywhere; "dino" is the quality pick on the
fleet cameras, at no region saving.

Run END TO END through the shipped judge (GLM-4.6V-Flash-9B x conservative) on
the same 68 cases, 2026-08-19:

                       regions   frame F1   frame recall   region precision
    photo                 1192      0.931          0.871              0.694
    photo+dino             727      0.931          0.871              0.773
    dino                  1228      0.984          0.968              0.896

Object recall is 55/73 in ALL THREE - better boxes do not make the judge find
more instances. What they change is everything else: three more anomalous frames
flagged, and kept-but-wrong regions down from 30 to 7. Frame specificity is
1.000 throughout, so no arm trades false alarms for this.

THE RECOMMENDED BADGE THEREFORE MOVED to "dino" on 2026-08-19. It is a measured
trade, not a preference: +69 % VLM calls for +0.097 frame recall and +0.123
region precision. Revert by moving `recommended` back to "photo+dino" if judge
cost, not miss rate, is the binding constraint on a deployment.

DEFAULT stays "photo" regardless: it is what every job in the history and every
published benchmark number was run with, and switching the default would
silently change what a re-run means.

A FOURTH arm, "dino+dinomaly" (`ddgate0.05` in tools/localizer_specs.py), adds a
per-camera Dinomaly model as a VETO on AnomalyDINO's boxes:

                       regions   frame F1   frame recall   region precision
    dino                  1228      0.984          0.968              0.896
    dino+dinomaly          654      0.984          0.968              0.896

Every quality column is EQUAL to "dino", to the digit, and that is the expected
result rather than a disappointment: a veto only deletes boxes, so "dino" is its
ceiling, and reaching that ceiling means none of the 574 deleted boxes held a
true instance. It is a pure cost lever - 47 % fewer judge calls - and it costs
one Dinomaly forward pass per frame to save them.

It is NOT recommended, for one reason that has nothing to do with the numbers:
it needs a checkpoint trained on nominal footage of THAT camera
(weights/dinomaly_<camera>.pt, tools/dinomaly_train.py). Eight exist; a ninth
camera has none. So the veto is applied when a checkpoint for the frame's camera
is found and SILENTLY SKIPPED otherwise - the arm degrades to plain "dino",
which is the recommended localizer anyway, and says so in info["dinomaly"].
Never let it degrade to something worse than what the user could have picked.

The camera is inferred from the reference PATH (see `resolve_camera`), or named
outright with the DINOMALY_CAMERA param. See docs/dino_models/.

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
        "measured": "Localizes all 45 instances of the 1762 benchmark, but on "
                    "the full 68-case set it boxes only 42/73 at strict IoU, and "
                    "5/26 on 39T where its worst box covers 98.9% of the frame. "
                    "Its noise is purely photometric: a cross-session empty frame "
                    "still yields 15-37 candidate regions from lighting alone.",
        "needs": [],
    },
    "photo+dino": {
        "name": "Pixel diff + DINOv2 gate",
        "summary": "The pixel diff proposes the boxes, then DINOv2 patch "
                   "features veto the ones with no semantic change - a lighting "
                   "shift on an empty seat looks identical to the features.",
        "measured": "Same 45/45 instances and the same box quality as the pixel "
                    "diff, with region precision 0.730 -> 0.815 and 57% fewer "
                    "VLM calls (559 -> 243) on the 1762 benchmark. Across all 68 "
                    "cases it is a pure cost lever: 42/73 strict IoU, exactly the "
                    "ungated pixel diff, because a veto never redraws a box.",
        "needs": ["torch"],
    },
    "dino": {
        "name": "DINOv2 features only (AnomalyDINO)",
        "summary": "No pixel comparison at all: each patch is scored by cosine "
                   "distance to the nominal reference patch at the same grid "
                   "position (AnomalyDINO, WACV 2025, specialised for a fixed "
                   "camera). The most accurate boxes once more than one camera "
                   "is scored, at no saving in VLM calls.",
        "measured": "Loses on the 1762 camera alone (strict IoU 32/45 vs 37/45, "
                    "patch-grid boxes) and WINS across the fleet: 58/73 strict "
                    "against the pixel diff's 42/73 on the 68-case set, and 19/26 "
                    "against 5/26 on 39T. It sends no fewer regions to the VLM "
                    "(1228 vs 1192), so it buys box quality, not cost. End to "
                    "end it lifts frame F1 0.931 -> 0.984 and region precision "
                    "0.694 -> 0.896 at unchanged object recall (55/73).",
        "needs": ["torch"],
        "recommended": True,
    },
    "dino+dinomaly": {
        "name": "AnomalyDINO + Dinomaly veto",
        "summary": "AnomalyDINO draws the same boxes as above, then a model "
                   "trained on nominal footage of this camera deletes the ones "
                   "it can reconstruct - it has seen the scene without them. "
                   "Identical accuracy to AnomalyDINO for half the VLM calls, "
                   "on the cameras that have a trained model.",
        "measured": "Equal to 'dino' on every quality column of the 68-case set "
                    "(frame F1 0.984, region precision 0.896, 55/73 instances, "
                    "46/73 strict) for 654 regions against 1228 - a veto cannot "
                    "beat what it filters, and matching it means none of the 574 "
                    "deleted boxes held a true instance. Costs one Dinomaly "
                    "forward per frame. Falls back to plain 'dino' on a camera "
                    "with no checkpoint.",
        "needs": ["torch", "checkpoint"],
    },
}

#: Checkpoint naming contract, duplicated from tools/dinomaly.checkpoint_path
#: ON PURPOSE: availability is asked for on every /api/localizers call and this
#: module promises to stay torch-free until a feature localizer is actually
#: selected, while importing tools.dinomaly costs a torch import. The two
#: spellings are pinned together by test_checkpoint_glob_matches_dinomaly.
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
CKPT_GLOB = "dinomaly_*.pt"


def names():
    return list(LOCALIZERS)


def catalog(reference: str = None):
    """Registry for /api/localizers, with availability resolved.

    `reference` is the reference image the job will use, when the caller knows
    it. It changes nothing about availability - it resolves `reference_note`,
    which is how an arm says "I am selectable, and on THIS input I will quietly
    do less than my name says". Only 'dino+dinomaly' has such a mode, and the
    picker sits directly above the reference chooser, so the warning is the
    difference between learning this before the run and after it.
    """
    out = []
    for key, spec in LOCALIZERS.items():
        ok, why = _availability(key)
        # `measured` is deliberately NOT shipped: the cards show the name and
        # the summary only. The numbers live in this file and in
        # benchmark/README.md, where they can be kept honest.
        note, note_ok = _reference_note(key, reference)
        out.append({"key": key, "name": spec["name"], "summary": spec["summary"],
                    "recommended": bool(spec.get("recommended")),
                    "available": ok, "unavailable_reason": why,
                    "reference_note": note, "reference_ok": note_ok,
                    "first_use_download_mb": 0 if _weights_cached() or not spec["needs"] else 84})
    return out


def _reference_note(key: str, reference: str):
    """(text, ok) about what this arm will actually do on `reference`."""
    if "checkpoint" not in LOCALIZERS.get(key, {}).get("needs", []):
        return "", True
    cams = checkpoint_cameras()
    if not cams:
        return "", True                   # already reported as unavailable
    if not reference:
        return f"Trained cameras: {', '.join(sorted(cams))}.", True
    cam = resolve_camera(reference)
    if cam:
        return f"Dinomaly model found for this reference: {cam}.", True
    return (f"No Dinomaly model matches this reference, so the veto cannot fire: "
            f"this runs exactly as '{LOCALIZERS['dino']['name']}' - same boxes, "
            f"same number of VLM calls. Trained cameras: "
            f"{', '.join(sorted(cams))}. Train one for this camera with "
            f"tools/dinomaly_train.py, or name the reference after a trained "
            f"camera."), False


def checkpoint_cameras():
    """Cameras with a trained Dinomaly checkpoint, longest name first so
    `resolve_camera` prefers the most specific match."""
    if not WEIGHTS_DIR.is_dir():
        return []
    names_ = [p.stem[len("dinomaly_"):] for p in WEIGHTS_DIR.glob(CKPT_GLOB)]
    return sorted(names_, key=lambda c: (-len(c), c))


def resolve_camera(reference: str, params: dict = None):
    """Which Dinomaly checkpoint a frame's reference belongs to, or None.

    The reference PATH is the evidence, because it is the one thing every
    caller already has: the app names an uploaded reference after its camera,
    and the benchmark's reference paths are literally
    `data/benchmark_39T/39T-cam53_...jpg`. `camera_of` in
    tools/localizer_specs.py reads the same fact off the benchmark's reference
    KEY - same convention, the copy each caller can actually see.

    Returning None is a normal answer, not a failure: the 'variant' scene has no
    model of its own, and `localize` degrades to plain 'dino' rather than
    scoring it against some other camera's model. (The offline benchmark DOES
    deliberately score variant against tram_1762, because out-of-domain
    behaviour is a thing it measures - a shipped job has no such reason.)
    """
    named = (params or {}).get("DINOMALY_CAMERA")
    if named:
        return str(named)
    hay = Path(reference).name
    for cam in checkpoint_cameras():
        if cam in hay:
            return cam
    return None


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
    if "checkpoint" in spec["needs"] and not checkpoint_cameras():
        # NO checkpoint at all means the veto could never fire on any frame, so
        # the arm is just 'dino' with extra words - do not offer it. ONE
        # checkpoint is enough to offer it: the per-frame fallback handles the
        # cameras that are not covered.
        return False, ("no Dinomaly checkpoint has been trained yet "
                       f"({WEIGHTS_DIR}/{CKPT_GLOB}). Train one on nominal "
                       "footage of a camera first: "
                       "python tools/dinomaly_train.py --camera <name>")
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


def warmup(name: str, references=()):
    """Load whatever the localizer needs, once, before the frame loop - a failed
    weight download must fail the job with one message, not N frame errors.

    `references` is the job's reference images. They are only needed by
    'dino+dinomaly', to resolve and load the per-camera checkpoints up front:
    the same reason as the backbone, plus one the backbone does not have -
    whether the veto will fire at all is a fact about this job that the operator
    should learn when it starts, not infer from a per-frame info key afterwards.
    Returns a note about that, or "".
    """
    if name not in ("photo+dino", "dino", "dino+dinomaly"):
        return ""
    try:
        _dino()._load_model()
    except Exception as exc:
        raise FrameError(
            f"cannot load the DINOv2 backbone for localizer '{name}': "
            f"{type(exc).__name__}: {exc}. The first use downloads ~84 MB "
            f"from dl.fbaipublicfiles.com; pick the 'photo' localizer to run "
            f"offline.") from exc
    if name != "dino+dinomaly":
        return ""

    cams = {resolve_camera(r) for r in references if r}
    covered = sorted(c for c in cams if c)
    for cam in covered:
        try:
            _dinomaly()._model(cam)
        except Exception as exc:
            raise FrameError(
                f"cannot load the Dinomaly checkpoint '{cam}' for localizer "
                f"'{name}': {type(exc).__name__}: {exc}") from exc
    if not references:
        return ""
    if not covered:
        return ("no Dinomaly checkpoint matches this job's reference - the veto "
                "will not fire and every frame runs as plain 'dino'")
    if None in cams:
        return (f"Dinomaly veto active for {', '.join(covered)}; the other "
                f"references have no checkpoint and run as plain 'dino'")
    return f"Dinomaly veto active for {', '.join(covered)}"


def _dino():
    import tools.dino_localizer as dl
    return dl


def _dinomaly():
    import tools.dinomaly_localizer as dml
    return dml


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
        return _tagged(regions, info, name)

    dl = _dino()
    if name in ("dino", "dino+dinomaly"):
        regions, info = dl.localize(
            reference, inspection,
            z_thr=float(params.get("DINO_Z", dl.Z_THRESHOLD)),
            abs_floor=float(params.get("DINO_FLOOR", dl.ABS_FLOOR)))
        if name == "dino+dinomaly":
            _dinomaly_veto(regions, info, reference, inspection, params)
    elif name == "photo+dino":
        regions, info = dl.localize_gated(
            reference, inspection,
            gate=float(params.get("DINO_GATE", dl.GATE_THRESHOLD)))
    else:
        raise FrameError(f"unknown localizer '{name}'")
    return _tagged(regions, info, name)


def _tagged(regions, info, name):
    """Stamp the arm's name and re-state the region count on the way out.

    `total` is re-derived rather than trusted because the sub-localizers set it
    BEFORE their veto step runs, so a gated arm would otherwise report the
    ungated count - a silently wrong cost number in the one field named for it.
    (The offline path pins the same key the same way, tools/localizer_specs.py.)
    """
    info["localizer"] = name
    info["total"] = len(regions)
    return regions, info


def _dinomaly_veto(regions, info, reference, inspection, params):
    """Delete the AnomalyDINO boxes the nominal model rebuilds. In place, and
    only if this camera has a model - see `resolve_camera`.

    The kept boxes keep their coordinates exactly, which is the property the
    whole cheap-A/B story rests on: a 'dino' run and a 'dino+dinomaly' run share
    every verdict in the cache for the boxes that survive, so the second arm
    costs only the frames the first one already paid for."""
    camera = resolve_camera(reference, params)
    info["dinomaly_camera"] = camera
    if camera is None:
        info["dinomaly"] = "no checkpoint for this reference - ran as 'dino'"
        info["vetoed"] = 0
        return regions
    dml = _dinomaly()
    gate = float(params.get("DINOMALY_GATE", dml.GATE_THRESHOLD))
    dml.support(reference, inspection, regions, camera=camera)
    kept = [r for r in regions if r["dinomaly_max"] > gate]
    info["vetoed"] = len(regions) - len(kept)
    info["dinomaly_gate"] = gate
    regions[:] = kept
    return regions
