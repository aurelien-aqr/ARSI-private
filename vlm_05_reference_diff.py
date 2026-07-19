#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - vlm_05_reference_diff.py
#  Change-detection of anomalies (forgotten objects, graffiti, vandalism) in a
#  tram interior.
#
#  The tram camera is FIXED, so a clean reference frame and the inspection frame
#  are pixel-aligned. Instead of asking an open-vocabulary detector to localize
#  tiny objects (which it misses - a wallet on the floor is invisible to
#  YOLO-World even at very low confidence), this script LOCALIZES purely from
#  the reference:
#     1) DIFF     the (grayscale) inspection against the reference, ignoring the
#                 already-masked black window areas, and threshold it into a
#                 change mask.
#     2) BLOBS    group the changed pixels into regions (connected components)
#                 and keep the reasonably sized ones as candidate objects.
#     3) CLASSIFY each candidate region with the local VLM (qwen2.5vl:7b via
#                 Ollama): is it an anomaly - a forgotten object, graffiti, or
#                 damage - (not a person, not a reflection)? and if so, what is it?
#
#  Differencing has high recall (it flags everything that changed) but also
#  flags people and lighting/reflection changes - the VLM is what tells those
#  apart from genuine anomalies.
#
#  Hardware target : x86 Ubuntu + NVIDIA RTX 3080 Ti (12 GB VRAM)
#  Model           : qwen2.5vl:7b  (served locally by Ollama)
#
#  Run from the repository root:   python vlm_05_reference_diff.py
# =============================================================================

import sys
import re
import json
from collections import deque
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import ollama

# Resolve every path relative to the repository root. [not USER CONFIG]
REPO_ROOT = Path(__file__).resolve().parent
def _p(path):
    p = Path(path)
    return str(p if p.is_absolute() else REPO_ROOT / p)

# =============================================================================
#  USER CONFIG  ---  the ONLY part you are meant to edit
# =============================================================================

# 1) Clean reference image (empty, undamaged tram - SAME fixed camera viewpoint).
REFERENCE_PATH  = "data/reference/tram_1762_v1_f0227_masked_reference.jpg"

# 2) Inspection image to be checked (windows already masked black).
INSPECTION_PATH = "data/masked/tram_1762_v2_f0143_masked.jpg"

# Where to save the annotated image and the detections JSON.
OUTPUT_PATH      = "results/tram_1762_v2_f0143_refdiff.jpg"
OUTPUT_JSON_PATH = "results/tram_1762_v2_f0143_refdiff.json"

# --- Change detection --------------------------------------------------------
# A pixel counts as "changed" when the blurred grayscale difference exceeds
# DIFF_THRESHOLD. Higher = fewer/only-stronger changes (misses faint objects);
# lower = more sensitive but more lighting/reflection noise.
DIFF_THRESHOLD = 40

# Gaussian blur radius applied to the difference before thresholding - smooths
# out per-pixel noise so a real object forms one solid blob.
BLUR_RADIUS = 3

# Changed pixels are grouped into connected regions on a downscaled mask (for
# speed). DILATE merges regions separated by small gaps so a fragmented object
# becomes a single box. MIN_AREA (in full-resolution pixels) drops specks -
# raise it to ignore small reflections, lower it to catch smaller objects.
# NOTE: these are per-camera / per-resolution knobs - retune them when you point
# the script at a different camera (a busier or differently-lit environment will
# usually want a slightly higher DIFF_THRESHOLD and MIN_AREA).
DOWNSCALE = 4
DILATE    = 2
MIN_AREA  = 500

# Upper size gate, in full-res px. This ONLY guards against degenerate regions that
# span almost the whole frame (a global exposure/white-balance shift). It must NOT
# be set low: real anomalies can be large - a coat draped over a seat measured
# ~102000 px and a wall covered in graffiti ~365000 px in this framing, so an
# aggressive cap silently deletes them (that was the original bug). The VLM, not a
# size threshold, is what rejects a whole-seat lighting change (it looks identical
# on both sides of the reference|now crop). Set to 0 to disable entirely.
MAX_AREA  = 400000

# Safety cap for rush hour: a crowded frame can differ from the empty reference
# in dozens of regions (people everywhere), and each region costs one VLM call.
# Only the MAX_REGIONS best-ranked changed regions are classified (applied AFTER
# the MAX_AREA gate above). Raise it if you would rather be exhaustive than fast.
# Ranking is by SALIENCE (mean diff intensity x sqrt(area)), not raw area: with
# area-ranking a dim whole-seat lighting blob could push a small bright phone
# out of the cap in a busy cross-session frame.
MAX_REGIONS = 25

# --- Multi-channel localization (measured on benchmark/ground_truth.json) -----
# A single global threshold cannot both catch faint anomalies and keep the
# region count sane: LOWERING the base threshold to 25 merges busy frames into
# giant blobs that the MAX_AREA gate then deletes (real_f0112 went 4/4 -> 0/4).
# So the base detector at DIFF_THRESHOLD stays UNTOUCHED and extra channels only
# ADD candidate boxes (region-LIST union, never mask union - masks would merge):
#
#  channel 2: the photometric diff again at SECOND_PASS_THRESHOLD. Catches
#     low-contrast solid objects (a dark bottle on the dark floor appears at
#     thr 30-35 but not at 40). Additions that overlap no base region are kept,
#     best SECOND_PASS_MAX_ADD by salience.
#  channel 3: added-EDGE-energy relu(|grad insp| - |grad ref|) restricted to
#     areas where the reference is locally FLAT, blurred and thresholded.
#     Sensor/JPEG noise is symmetric between the two frames and cancels; a
#     faint tag ADDS one-sided stroke edges on a flat panel (the ZORK tag sits
#     12x above the strongest noise box in this domain; the photometric diff
#     put it BELOW noise). Catches faint graffiti that no global photometric
#     threshold can reach without flooding.
#
# Measured on the GT: localization recall 41/45 -> 45/45 (ZORK + XRP faint
# tags and both floor bottles recovered; the XRP tag was first thought
# unreachable, but that probe measured a misplaced GT box - the channels box
# the real tag and every judge names it), at +52% candidate regions on anomaly
# frames. Set a threshold to 0 to disable that channel.
SECOND_PASS_THRESHOLD = 30
SECOND_PASS_MAX_ADD   = 8
EDGE_THRESHOLD        = 1.5
EDGE_FLAT_THR         = 6.0     # reference |grad| above this = not a flat surface
EDGE_MAX_ADD          = 4

# --- Person filter -------------------------------------------------------------
# A cheap person detector (YOLOv8-nano, ~2-6 s/frame on CPU, ~20 ms on GPU) runs
# once per inspection frame; candidate regions mostly contained in a person box
# (intersection-over-region-area >= PERSON_IOA) are dropped BEFORE the VLM.
# This is what cleanly separates "jacket worn by a passenger" (inside the person
# box -> vetoed) from "jacket forgotten on a seat" (no person box -> kept), which
# no label blacklist can do - a forgotten jacket is a real anomaly. Also saves
# the VLM calls those regions would have cost. Verified on GT: vetoes person
# regions on real_f0219 / gpt_11 while losing ZERO ground-truth instances.
# If ultralytics or the weights are unavailable the filter degrades gracefully
# (warns once and keeps every region).
PERSON_FILTER  = True
PERSON_CONF    = 0.35
PERSON_IOA     = 0.6
PERSON_WEIGHTS = "yolov8n.pt"   # auto-downloaded by ultralytics on first use

# --- VLM classification ------------------------------------------------------
# If True, crop each changed region and send it to the local VLM.
USE_VLM = True

# What the VLM step does (same idea as vlm_04):
#   "label"  - the VLM only NAMES each region; every region is kept and boxed.
#   "filter" - regions the VLM answers NO to (people, reflections, empty) are
#              dropped, keeping only genuine anomalies.
VLM_MODE = "filter"

# Context padding around each region before sending it to the VLM (fixed pixels
# PLUS a fraction of the region size), so small crops keep their surroundings.
CROP_MARGIN  = 40
CROP_CONTEXT = 0.75

# Minimum side (px) of each half of the side-by-side crop before it is sent to
# the VLM. Tiny regions (a phone on a seat is ~50 px) are upscaled to at least
# this size so the VLM has enough detail to judge them - without it, qwen3 tends
# to dismiss small dark objects as reflections and answer NO. Set to 0 to send
# crops at native size.
MIN_CROP_SIDE = 320

# Safety net for crowded frames: even when told to answer NO for people, the 7B
# VLM sometimes answers YES and then names a body part ("hair", "jacket on a
# person"). Any region whose VLM label contains one of these words is dropped,
# regardless of the YES/NO. This is semantic (not pixel positions), so it keeps
# working when the script is pointed at a different camera.
NEGATIVE_LABELS = ["person", "people", "passenger", "man", "woman", "child",
                   "hair", "head", "face", "hand", "arm", "leg", "body",
                   "shoe", "foot", "skin"]

# The diff is symmetric, so it also lights up things that were in the reference and
# are now GONE (a seat folded, a reflection that moved). The VLM sometimes narrates
# that as "X disappeared / removed / is missing" and answers YES - but a
# DISAPPEARANCE is not an abandoned-object/graffiti/damage anomaly, so drop any
# label that describes something leaving rather than appearing.
DISAPPEAR_LABELS = ["disappear", "removed", "missing", "gone", "no longer",
                    "empty seat", "nothing"]

# Precision guard against VLM hallucinations on empty-seat / lighting regions. Two
# label-vs-region sanity checks (measured on this data, chosen to drop 0 true
# positives): a NAMED small object cannot be huge - the worst false positive was a
# whole empty seat (89184 px) called "phone" - and a bare "YES" with no object name
# is a low-confidence guess (every real detection here named its object). Graffiti
# is deliberately NOT size-checked: a real tag measured as small as 656 px.
SMALL_OBJECT_WORDS = ["phone", "wallet", "purse", "coin", "key", "card", "sim",
                      "earbud", "ring", "watch", "pen", "glasses", "lighter"]
SMALL_OBJECT_MAX_AREA = 6000    # px; a genuine phone/wallet region was <= ~4200 px


def is_implausible(label: str, area: int) -> bool:
    """True if a YES verdict is likely a hallucination: an unnamed 'YES', or a
    small-object name pinned to a region far too large to be that object."""
    if not label.strip():
        return True
    if area > SMALL_OBJECT_MAX_AREA and (_words(label) & set(SMALL_OBJECT_WORDS)):
        return True
    return False

# The question sent to the VLM for each changed region. The VLM is shown the SAME
# spot twice, side by side: LEFT = the clean reference, RIGHT = now. Asking "what
# ANOMALY appeared on the right" (a relative judgement) instead of "is this an
# object" (an absolute one) is what rejects reflections/lighting - they look the
# same on both sides - and what makes the decision portable to other cameras.
#
# TWO prompts are provided:
#   PROMPT_LENIENT - the original: high recall, but the benchmark showed it also
#                    hallucinates objects/graffiti on empty seats & lighting and
#                    flags a person's own clothing (region precision ~0.75 after
#                    the plausibility filter).
#   PROMPT         - conservative anti-hallucination rewrite (DEFAULT): adds
#                    explicit NO conditions for same-scene-different-lighting, for
#                    anything a person wears/holds, and for metal scratches/glare
#                    mistaken as graffiti, and an "if unsure, answer NO" rule. Meant
#                    to raise precision; A/B it against PROMPT_LENIENT on GPU
#                    (changing the prompt invalidates benchmark/cache.json, so it
#                    forces a full re-run - minutes on GPU, hours on CPU).
PROMPT_LENIENT = """You are shown the SAME spot in a tram twice, side by side:
- LEFT half  = the clean reference (normal, undamaged tram).
- RIGHT half = the same spot now.
Did an ANOMALY appear on the right that is NOT on the left? An anomaly is any of:
- a forgotten object (bag, backpack, phone, wallet, bottle, package),
- graffiti or a tag / scribble drawn on a surface,
- damage or vandalism (a torn or slashed seat, a broken part).
Answer NO if the only difference is a person or body part, a shadow, a
reflection, or a lighting/exposure change.
Reply with YES or NO, then name what appeared in 2-4 words."""

PROMPT = """You are shown the SAME spot in a tram twice, side by side:
- LEFT half  = the clean reference (normal, empty, undamaged tram).
- RIGHT half = the same spot now.

Report ONLY a clear new ANOMALY that is present on the RIGHT and absent on the
LEFT. An anomaly is one of:
- a forgotten object (bag, backpack, phone, wallet, bottle, package),
- litter left behind (can, bottle, wrapper, paper, food),
- graffiti: painted letters, a tag or a scribble on a surface,
- damage / vandalism: a torn or slashed seat, a broken part.

Answer NO (it is NOT an anomaly) if ANY of these hold:
- the RIGHT shows the SAME empty seat, floor, wall, pole or panel as the LEFT and
  only the brightness, shadow, reflection or colour differs,
- the difference is a person, or anything a person is wearing, holding or sitting
  on,
- it is a scratch, glare or reflection on metal or glass (these are NOT graffiti).

Be conservative: answer YES only if you can clearly SEE and NAME a specific new
object, marking or damage. If you are unsure, answer NO.

Reply with YES or NO, then name what appeared in 2-4 words."""

# Classifier model (served by Ollama). Both options below are ~6 GB and fit the
# 12 GB GPU. qwen3-vl:8b-instruct is markedly better at REJECTING people and
# reflections in the side-by-side comparison (qwen2.5vl:7b tends to answer YES
# and then name a body part), so it is the default here.
# Alternative: "qwen2.5vl:7b".
MODEL_NAME = "qwen3-vl:8b-instruct"

# =============================================================================
#  HARDWARE-LOCK  ---  DO NOT CHANGE
# =============================================================================
NUM_CTX     = 4096
NUM_PREDICT = 512
TEMPERATURE = 0.1
# =============================================================================


def check_model(model_name: str) -> None:
    """Verify that the Ollama server is reachable and the model is installed."""
    try:
        data = ollama.list()
    except Exception as exc:
        print("ERROR: could not reach the Ollama server.")
        print("Start it (in another terminal) with:  ollama serve")
        print(f"(details: {exc})")
        sys.exit(1)

    models = getattr(data, "models", None)
    if models is None and isinstance(data, dict):
        models = data.get("models", [])
    names = []
    for m in models or []:
        name = getattr(m, "model", None) or getattr(m, "name", None)
        if name is None and isinstance(m, dict):
            name = m.get("model") or m.get("name")
        if name:
            names.append(name)

    # Ollama resolves a tag-less name to ":latest" - accept both forms, so
    # "owner/model" passes when "owner/model:latest" is installed (an exact
    # string compare here aborted valid model-sweep runs on the GPU machine).
    if model_name not in names and f"{model_name}:latest" not in names:
        print(f"ERROR: model '{model_name}' is not installed.")
        print(f"Install it with:  ollama pull {model_name}")
        sys.exit(1)


def _gray_pair(reference_path: str, inspection_path: str):
    """Grayscale float arrays of (reference, inspection-resized-to-reference) and
    the both-black mask of the already-masked window areas."""
    ref = Image.open(reference_path).convert("L")
    insp = Image.open(inspection_path).convert("L")
    if ref.size != insp.size:
        insp = insp.resize(ref.size)
    a = np.asarray(ref, dtype=np.float32)
    b = np.asarray(insp, dtype=np.float32)
    black = (a < 12) & (b < 12)
    return a, b, black


def _blur_f(arr, radius, scale=1.0):
    """Gaussian blur of a float array via PIL (uint8 transport; `scale` keeps
    sub-integer detail for low-amplitude maps)."""
    u8 = np.clip(arr * scale, 0, 255).astype(np.uint8)
    out = np.asarray(Image.fromarray(u8).filter(ImageFilter.GaussianBlur(radius)),
                     dtype=np.float32)
    return out / scale


def photo_data(a, b, black, thr=None):
    """Photometric change: (mask, blurred-diff map). The vlm_05 base detector."""
    thr = DIFF_THRESHOLD if thr is None else thr
    d = np.abs(a - b)
    d[black] = 0
    d = _blur_f(d, BLUR_RADIUS)
    return d > thr, d


def edge_energy_map(a, b, black):
    """Added-edge energy: relu(|grad insp| - |grad ref|) where the reference is
    locally flat, blurred. Symmetric sensor/JPEG noise cancels in the one-sided
    subtraction; faint strokes drawn on a flat panel survive (see the channel-3
    comment in USER CONFIG)."""
    gya, gxa = np.gradient(a)
    ga = np.hypot(gxa, gya)
    gyb, gxb = np.gradient(b)
    gb = np.hypot(gxb, gyb)
    ga_smooth = _blur_f(ga, 3, scale=8.0)
    e = np.maximum(gb - ga, 0.0)
    e[ga_smooth > EDGE_FLAT_THR] = 0.0   # only trust flat-reference surfaces
    e[black] = 0.0
    return _blur_f(e, 6, scale=8.0)


def change_mask(reference_path: str, inspection_path: str):
    """Boolean full-resolution mask of pixels that changed vs the reference
    (base photometric channel only - kept for backward compatibility)."""
    a, b, black = _gray_pair(reference_path, inspection_path)
    mask, _ = photo_data(a, b, black)
    return mask


def salience(region, dmap) -> float:
    """Region rank key: mean diff intensity x sqrt(area). Prefers small-but-sharp
    real objects over large dim lighting blobs when the MAX_REGIONS cap bites."""
    x0, y0, x1, y1 = region["bbox"]
    patch = dmap[y0:y1, x0:x1]
    mean = float(patch.mean()) if patch.size else 0.0
    return mean * (region["area"] ** 0.5)


def _ioa(inner, outer) -> float:
    """Intersection area over `inner`'s area (how much of inner lies in outer)."""
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / area if area > 0 else 0.0


_person_model = None
_person_warned = False


def person_boxes(inspection_path: str, ref_size):
    """Person boxes from YOLOv8-nano, scaled into the REFERENCE pixel space (the
    inspection may have a different native size). Returns [] and warns once if
    ultralytics / the weights are unavailable, so the pipeline still runs."""
    global _person_model, _person_warned
    if not PERSON_FILTER:
        return []
    try:
        if _person_model is None:
            from ultralytics import YOLO
            _person_model = YOLO(_p(PERSON_WEIGHTS))
        res = _person_model.predict(inspection_path, classes=[0],
                                    conf=PERSON_CONF, verbose=False)[0]
        sx = ref_size[0] / res.orig_shape[1]
        sy = ref_size[1] / res.orig_shape[0]
        return [(int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))
                for x0, y0, x1, y1 in res.boxes.xyxy.tolist()]
    except Exception as exc:
        if not _person_warned:
            print(f"WARNING: person filter unavailable ({type(exc).__name__}: "
                  f"{exc}); continuing without it.")
            _person_warned = True
        return []


def _boxes_overlap(box_a, box_b) -> bool:
    """Lenient overlap (IoU > 0.1 or either centre inside the other box) - the
    same rule the benchmark uses to match regions to ground-truth instances."""
    if _iou(box_a, box_b) > 0.1:
        return True
    for inner, outer in ((box_a, box_b), (box_b, box_a)):
        cx = (inner[0] + inner[2]) / 2
        cy = (inner[1] + inner[3]) / 2
        if outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]:
            return True
    return False


def localize(reference_path: str, inspection_path: str):
    """Full multi-channel localization: base photometric regions (person-vetoed,
    then capped to MAX_REGIONS by salience), plus bounded additions from the
    low-threshold and added-edge channels (region-list union, so the proven base
    channel is never disturbed). The channel additions ride ON TOP of the cap -
    capping the merged list instead was measured to evict a small real object
    (the far phone in real_f0219) once 12 additions crowded in. Worst-case VLM
    budget per frame = MAX_REGIONS + SECOND_PASS_MAX_ADD + EDGE_MAX_ADD.

    Returns (regions, info) where info carries per-channel counts for logging.
    """
    a, b, black = _gray_pair(reference_path, inspection_path)
    persons = person_boxes(inspection_path, (a.shape[1], a.shape[0]))
    info = {"persons": len(persons), "person_veto": 0}

    def veto(rs):
        if not persons:
            return rs
        kept = [r for r in rs
                if not any(_ioa(r["bbox"], p) >= PERSON_IOA for p in persons)]
        info["person_veto"] += len(rs) - len(kept)
        return kept

    base_mask, base_dmap = photo_data(a, b, black)
    base = find_regions(base_mask, DOWNSCALE, DILATE, MIN_AREA, MAX_AREA)
    info["base"] = len(base)
    for r in base:
        r["channel"] = "photo"
        r["salience"] = salience(r, base_dmap)
    base = veto(base)
    base.sort(key=lambda r: -r["salience"])
    capped = len(base) > MAX_REGIONS
    regions = base[:MAX_REGIONS]

    info["second"] = 0
    if SECOND_PASS_THRESHOLD:
        mask2, dmap2 = photo_data(a, b, black, thr=SECOND_PASS_THRESHOLD)
        adds = [r for r in find_regions(mask2, DOWNSCALE, DILATE, MIN_AREA, MAX_AREA)
                if not any(_boxes_overlap(r["bbox"], q["bbox"]) for q in base)]
        for r in adds:
            r["channel"] = "photo_lo"
            r["salience"] = salience(r, dmap2)
        adds = veto(adds)
        adds.sort(key=lambda r: -r["salience"])
        regions += adds[:SECOND_PASS_MAX_ADD]
        info["second"] = min(len(adds), SECOND_PASS_MAX_ADD)

    info["edge"] = 0
    if EDGE_THRESHOLD:
        emap = edge_energy_map(a, b, black)
        adds = [r for r in find_regions(emap > EDGE_THRESHOLD,
                                        DOWNSCALE, DILATE, MIN_AREA, MAX_AREA)
                if not any(_boxes_overlap(r["bbox"], q["bbox"]) for q in regions)]
        for r in adds:
            r["channel"] = "edge"
            # edge energies live on a ~0-8 scale vs 0-255 photometric: rescale so
            # the salience fields stay comparable across channels in the output.
            r["salience"] = salience(r, emap) * 30.0
        adds = veto(adds)
        adds.sort(key=lambda r: -r["salience"])
        regions += adds[:EDGE_MAX_ADD]
        info["edge"] = min(len(adds), EDGE_MAX_ADD)

    info["total"] = info["base"] + info["second"] + info["edge"]
    info["capped"] = capped
    return regions, info


def find_regions(mask, downscale: int, dilate: int, min_area: int, max_area: int = 0):
    """Group changed pixels into bounding boxes via connected components.

    Works on a downscaled copy of the mask for speed, then scales the boxes back
    up. Returns a list of {"bbox": [x0,y0,x1,y1], "area": px} in full-res pixels.
    Regions smaller than min_area (specks) or, when max_area > 0, larger than
    max_area (whole-seat lighting shifts) are dropped.
    """
    h, w = mask.shape
    small = mask[:h // downscale * downscale, :w // downscale * downscale]
    small = small.reshape(h // downscale, downscale,
                          w // downscale, downscale).max(axis=(1, 3))
    if dilate > 0:  # merge nearby fragments (dilation via PIL max filter)
        img = Image.fromarray((small * 255).astype(np.uint8))
        img = img.filter(ImageFilter.MaxFilter(2 * dilate + 1))
        small = np.asarray(img) > 0

    sh, sw = small.shape
    labels = np.zeros((sh, sw), dtype=np.int32)
    regions = []
    cur = 0
    for i in range(sh):
        for j in range(sw):
            if not small[i, j] or labels[i, j]:
                continue
            cur += 1
            q = deque([(i, j)])
            labels[i, j] = cur
            minx = maxx = j
            miny = maxy = i
            area = 0
            while q:
                y, x = q.popleft()
                area += 1
                minx, maxx = min(minx, x), max(maxx, x)
                miny, maxy = min(miny, y), max(maxy, y)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < sh and 0 <= nx < sw and small[ny, nx] and not labels[ny, nx]:
                            labels[ny, nx] = cur
                            q.append((ny, nx))
            full_area = area * downscale * downscale
            if full_area < min_area:
                continue
            if max_area and full_area > max_area:
                continue
            regions.append({
                "bbox": [minx * downscale, miny * downscale,
                         (maxx + 1) * downscale, (maxy + 1) * downscale],
                "area": full_area,
            })
    regions.sort(key=lambda r: -r["area"])
    return regions


def render_crop_pair(image, reference, bbox, margin: int, context: float):
    """Reference|inspection side-by-side crop of a padded bbox — the exact
    image the judge sees. Factored out so training-data export (tools/
    export_lora_dataset.py) renders samples identically to inference."""
    width, height = image.size
    x0, y0, x1, y1 = bbox
    pad_x = margin + int(context * (x1 - x0))
    pad_y = margin + int(context * (y1 - y0))
    cx0 = max(0, int(x0) - pad_x)
    cy0 = max(0, int(y0) - pad_y)
    cx1 = min(width, int(x1) + pad_x)
    cy1 = min(height, int(y1) + pad_y)
    box = (cx0, cy0, cx1, cy1)
    insp_crop = image.crop(box)
    ref_crop = reference.crop(box)

    # Upscale tiny crops so the VLM gets enough detail on small objects (a phone
    # crop is otherwise dismissed as a reflection). Same factor on both halves.
    long_side = max(insp_crop.size)
    if MIN_CROP_SIDE and long_side < MIN_CROP_SIDE:
        scale = MIN_CROP_SIDE / long_side
        new_size = (round(insp_crop.width * scale), round(insp_crop.height * scale))
        insp_crop = insp_crop.resize(new_size)
        ref_crop = ref_crop.resize(new_size)

    # Paste reference and inspection side by side with a thin white separator.
    sep = 6
    cw, ch = insp_crop.size
    combined = Image.new("RGB", (cw * 2 + sep, ch), (255, 255, 255))
    combined.paste(ref_crop, (0, 0))
    combined.paste(insp_crop, (cw + sep, 0))
    return combined


def classify_with_vlm(image, reference, region, margin: int, context: float):
    """Crop the region from BOTH images and ask the VLM what appeared.

    The same padded box is cut from the reference (LEFT) and the inspection
    (RIGHT) and pasted side by side, so the VLM compares the two directly. This
    relative judgement rejects reflections/lighting (identical on both sides) and
    is what carries over to a different camera.

    Returns (is_object, vlm_label). Saves the crop to a temp file because the
    Ollama client takes image paths (same call pattern as vlm_01/02/03/04).
    """
    combined = render_crop_pair(image, reference, region["bbox"], margin, context)

    crop_path = _p("results/_crop_tmp.jpg")
    Path(crop_path).parent.mkdir(parents=True, exist_ok=True)
    combined.save(crop_path)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": PROMPT,
            "images": [crop_path],
        }],
        think=False,  # skip chain-of-thought; write the answer straight into `content`
        options={
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "temperature": TEMPERATURE,
        },
    )

    text = response["message"]["content"].strip()
    is_object = text.upper().lstrip().startswith("YES")
    label = text.splitlines()[0].strip() if text else ""
    for prefix in ("YES", "NO"):
        if label.upper().startswith(prefix):
            label = label[len(prefix):].lstrip(" .,:;-").strip()
            break
    # If the model put the bare verdict on its own line ("YES\nA backpack"), the
    # object name is on the next line - take it, so a real detection is not left
    # nameless (an empty name is later treated as a low-confidence guess).
    if not label:
        for line in text.splitlines()[1:]:
            if line.strip():
                label = line.strip()
                break
    return is_object, label


def _words(text: str):
    """Lower-case word tokens of a label (for whole-word matching)."""
    return set(re.findall(r"[a-z]+", text.lower()))


def is_non_anomaly(label: str) -> bool:
    """True if a VLM label describes a person/body part or a DISAPPEARANCE rather
    than a genuine appearing anomaly. Such regions are dropped even on a YES.
    Body-part words are matched as WHOLE WORDS - matching "face" as a substring
    would wrongly kill "graffiti on sur-face" or "hand" would kill "handle"."""
    if _words(label) & set(NEGATIVE_LABELS):
        return True
    low = label.lower()                       # DISAPPEAR_LABELS holds phrases too
    return any(neg in low for neg in DISAPPEAR_LABELS)


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def dedupe_regions(regions, iou_thr: float = 0.4):
    """Merge kept regions that overlap a lot (the VLM often fires 2-3 boxes on one
    object, e.g. a backpack + its strap). Keeps the largest of each overlapping
    group so one physical anomaly is reported once."""
    kept = []
    for r in sorted(regions, key=lambda x: -x["area"]):
        if all(_iou(r["bbox"], k["bbox"]) <= iou_thr for k in kept):
            kept.append(r)
    return kept


def area_color(area):
    """Map a region area (px) to a colour from green (small) to red (large)."""
    if area < 1000:
        return (0, 200, 0)      # green
    if area < 3000:
        return (255, 200, 0)    # amber
    if area < 8000:
        return (255, 120, 0)    # orange
    return (220, 0, 0)          # red


def main() -> None:
    inspection_path = _p(INSPECTION_PATH)
    reference_path = _p(REFERENCE_PATH)
    for label, path in (("inspection", inspection_path), ("reference", reference_path)):
        if not Path(path).exists():
            print(f"ERROR: {label} image not found: {path}")
            print("Put images into data/ and edit the paths in USER CONFIG.")
            sys.exit(1)

    if USE_VLM:
        check_model(MODEL_NAME)

    print(f"Inspection : {inspection_path}")
    print(f"Reference  : {reference_path}")
    print(f"VLM        : {MODEL_NAME if USE_VLM else '(disabled)'}\n")

    # Load both images and put the inspection in the reference's pixel space. The
    # camera is fixed, so a same-sized frame is already aligned; an inspection of a
    # DIFFERENT size (e.g. an exported/AI-edited frame) is uniformly resized onto
    # the reference so every downstream coordinate - the change mask, the VLM crops
    # and the drawn boxes - lives in ONE coordinate space. Without this the crops
    # and boxes (computed in reference space) would be misplaced on a differently
    # sized inspection.
    reference = Image.open(reference_path).convert("RGB")
    image = Image.open(inspection_path).convert("RGB")
    if image.size != reference.size:
        ar_ins = image.size[0] / image.size[1]
        ar_ref = reference.size[0] / reference.size[1]
        if abs(ar_ins - ar_ref) > 0.02:
            print(f"WARNING: inspection aspect ratio {ar_ins:.3f} differs from "
                  f"reference {ar_ref:.3f}; the resize will distort the image. The "
                  f"diff only makes sense if both share the same camera framing.")
        print(f"Note: inspection {image.size} resized to reference "
              f"{reference.size} to align them for differencing.")
        image = image.resize(reference.size)

    # --- 1+2) Localize changed regions ---------------------------------------
    regions, loc = localize(reference_path, inspection_path)
    print(f"Change detection: base {loc['base']} region(s) "
          f"+ {loc['second']} low-threshold + {loc['edge']} edge-channel, "
          f"{loc['person_veto']} vetoed by {loc['persons']} person box(es).")
    if loc["capped"]:
        print(f"Rush-hour cap: only the {MAX_REGIONS} most salient base regions "
              f"(of {loc['base']}) are classified; channel additions ride on top.")

    # --- 3) VLM classification -----------------------------------------------
    kept = []
    if USE_VLM:
        for r in regions:
            is_object, label = classify_with_vlm(image, reference, r,
                                                 CROP_MARGIN, CROP_CONTEXT)
            # Override a YES that is a person/"disappeared" label or an implausible
            # hallucination (unnamed, or a small object named on a huge region).
            if is_non_anomaly(label) or is_implausible(label, r["area"]):
                is_object = False
            r["vlm_label"] = label
            r["vlm_is_object"] = is_object
            if VLM_MODE == "filter" and not is_object:
                continue
            kept.append(r)
        # Collapse the 2-3 boxes the VLM often fires on one object into one.
        if VLM_MODE == "filter":
            before = len(kept)
            kept = dedupe_regions(kept)
            print(f"VLM (filter): kept {len(kept)} object(s) of "
                  f"{len(regions)} changed region(s) "
                  f"({before - len(kept)} duplicate box(es) merged).")
        else:
            n_yes = sum(1 for r in kept if r["vlm_is_object"])
            print(f"VLM (label): kept all {len(kept)} region(s); "
                  f"the VLM called {n_yes} of them an object.")
    else:
        for r in regions:
            r["vlm_label"] = ""
            r["vlm_is_object"] = None
        kept = regions

    # --- 4) Output -----------------------------------------------------------
    print("\n----- ANOMALIES ----------------------------------------------")
    if not kept:
        print("(none)")
    for r in kept:
        print(f"  - area={r['area']:<6} bbox={r['bbox']}  vlm='{r.get('vlm_label', '')}'")
    print("-------------------------------------------------------------")

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for r in kept:
        x0, y0, x1, y1 = (int(v) for v in r["bbox"])
        colour = area_color(r["area"])
        tag = r.get("vlm_label") or "object"
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=3)
        draw.text((x0 + 3, max(0, y0 - 20)), tag, fill=colour, font=font)

    output_path = _p(OUTPUT_PATH)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"\nAnnotated image saved to: {output_path}")

    output_json_path = _p(OUTPUT_JSON_PATH)
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as fh:
        json.dump(kept, fh, indent=2, ensure_ascii=False)
    print(f"Detections JSON saved to: {output_json_path}")


if __name__ == "__main__":
    main()
