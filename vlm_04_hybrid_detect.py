#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - vlm_04_hybrid_detect.py
#  Hybrid detection of forgotten personal objects in a tram interior.
#
#  Two-stage pipeline (localize -> confirm):
#     1) LOCALIZE   with an open-vocabulary DETECTOR (YOLO-World, ultralytics).
#     2) FILTER      (optional) against a clean reference image: keep only
#                    objects that are NEW compared to the reference.
#     3) CONFIRM     (optional) each surviving candidate with the local VLM
#                    (qwen2.5vl:7b via Ollama) by cropping the box and asking
#                    a short yes/no + label question.
#
#  This is a POC. There is NO person detection, NO tracking, and NO temporal
#  "owner walked away" logic - it only finds candidate objects, optionally keeps
#  the new ones, and optionally asks the VLM to confirm/label each crop.
#
#  Hardware target : x86 Ubuntu + NVIDIA RTX 3080 Ti (12 GB VRAM)
#  Detector        : YOLO-World (weights auto-downloaded by ultralytics)
#  Model (confirm) : qwen2.5vl:7b  (served locally by Ollama)
#
#  Run from the repository root:   python vlm_04_hybrid_detect.py
# =============================================================================

import sys
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import ollama

# Resolve every path relative to the repository root. [not USER CONFIG]
REPO_ROOT = Path(__file__).resolve().parent
def _p(path):
    p = Path(path)
    return str(p if p.is_absolute() else REPO_ROOT / p)

# =============================================================================
#  USER CONFIG  ---  the ONLY part you are meant to edit
# =============================================================================

# 1) Clean reference image (empty, undamaged tram interior).
REFERENCE_PATH  = "data/reference/tram_1762_v1_f0227_masked_reference.jpg"

# 2) Inspection image to be checked (windows already masked black).
INSPECTION_PATH = "data/masked/tram_1762_v2_f0143_masked.jpg"

# Where to save the annotated image and the detections JSON.
OUTPUT_PATH      = "results/tram_1762_v2_f0032_hybrid.jpg"
OUTPUT_JSON_PATH = "results/tram_1762_v2_f0032_hybrid.json"

# --- Detector (YOLO-World) ---------------------------------------------------
# Open-vocabulary weights, auto-downloaded by ultralytics on first run.
# Lighter option (faster, less accurate): "yolov8s-worldv2.pt".
DETECTOR_WEIGHTS = "yolov8x-worldv2.pt"

# Open-vocabulary classes to look for (personal objects a passenger may forget).
DETECTOR_CLASSES = ["cell phone", "wallet", "handbag", "backpack",
                    "bag", "suitcase", "laptop"]

# Confidence threshold for keeping a detection. These objects are SMALL and
# often partly occluded, so we keep it LOW on purpose: a low threshold trades
# more false positives for fewer missed objects - the reference filter and the
# VLM confirmation step below are what prune the false positives back out.
DETECTOR_CONF = 0.02

# Inference resolutions (longest side) fed to the detector - the image is run
# once per size and the results are merged. No single scale sees every object:
# a far backpack only appears at 1280, while a phone on a seat only appears at
# 640 (at 1280 it is smoothed away). Running BOTH recovers both. More scales =
# better recall but slower; use a single value like [1280] if you need speed.
IMGSZS = [640, 1280]

# --- Reference filtering -----------------------------------------------------
# If True, run the detector on the reference too and keep only inspection boxes
# with NO same-class overlap in the reference (i.e. the genuinely NEW objects).
USE_REFERENCE = True

# Two same-class boxes count as "the same object" when their IoU is above this.
REFERENCE_IOU = 0.3

# --- VLM confirmation --------------------------------------------------------
# If True, crop each surviving candidate and send it to the local VLM.
USE_VLM = True

# What the VLM step does:
#   "label"  - the VLM only NAMES each object; every reference-filtered
#              candidate is kept and boxed even if the VLM says NO or mislabels
#              it. Use this when the goal is "box every abandoned object" and the
#              detector + reference filter already decide what counts.
#   "filter" - the VLM also DECIDES: candidates it answers NO to are dropped.
#              Stricter (fewer false positives) but it will discard hard objects
#              (a tiny phone the VLM cannot recognize even with context).
VLM_MODE = "label"

# Context around each crop before sending it to the VLM. Small objects sent as
# a tight crop lose their surroundings and the VLM misreads them (a wallet on a
# seat becomes "a laptop"); giving it the seat context around the object fixes
# most confirmations. The crop is padded by CROP_MARGIN pixels AND by
# CROP_CONTEXT x the box size, so tiny boxes get proportionally more context.
CROP_MARGIN  = 40
CROP_CONTEXT = 0.75

# The question sent to the VLM for each cropped candidate.
PROMPT = """Is this a personal object a passenger might have forgotten
(phone, wallet, bag)? Answer YES or NO, then name the object in 2-3 words."""

# =============================================================================
#  HARDWARE-LOCK  ---  DO NOT CHANGE
# =============================================================================
MODEL_NAME  = "qwen3-vl:8b-instruct"   # locked to this machine's GPU (RTX 3080 Ti)
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


def load_detector(weights: str, classes):
    """Load a YOLO-World detector and set its open-vocabulary classes."""
    try:
        from ultralytics import YOLO
    except Exception as exc:
        print("ERROR: could not import ultralytics.")
        print("Install it with:  pip install ultralytics   (or: pip install -r requirements.txt)")
        print(f"(details: {exc})")
        sys.exit(1)

    try:
        model = YOLO(_p(weights) if Path(_p(weights)).exists() else weights)
        model.set_classes(classes)
    except Exception as exc:
        print(f"ERROR: could not load the YOLO-World detector '{weights}'.")
        print("On first run ultralytics downloads the weights - check your connection.")
        print(f"(details: {exc})")
        sys.exit(1)
    return model


def detect(model, image_path: str, classes, conf: float, imgszs):
    """Run the detector on one image at several scales and merge the results.

    Each detection is a dict: {"label", "confidence", "bbox": [x0,y0,x1,y1]}
    with bbox in ABSOLUTE pixels of that image. The image is predicted once per
    size in imgszs; boxes from all scales are pooled, then merged
    class-agnostically so the same object seen at two scales (or emitted under
    two close classes, e.g. "cell phone" and "wallet") collapses to one, keeping
    the highest-confidence label - localization is the detector's job, final
    naming is the VLM's.
    """
    detections = []
    for imgsz in imgszs:
        results = model.predict(image_path, conf=conf, imgsz=imgsz, verbose=False)
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                cls_idx = int(b.cls[0])
                label = classes[cls_idx] if 0 <= cls_idx < len(classes) else str(cls_idx)
                x0, y0, x1, y1 = (float(v) for v in b.xyxy[0])
                detections.append({
                    "label": label,
                    "confidence": float(b.conf[0]),
                    "bbox": [x0, y0, x1, y1],
                })
    return merge_duplicates(detections)


def merge_duplicates(detections, iou_thr: float = 0.6):
    """Class-agnostic NMS: drop lower-confidence boxes that overlap a kept one."""
    kept = []
    for d in sorted(detections, key=lambda x: -x["confidence"]):
        if all(iou(d["bbox"], k["bbox"]) <= iou_thr for k in kept):
            kept.append(d)
    return kept


def iou(box_a, box_b) -> float:
    """Intersection-over-union of two [x0, y0, x1, y1] pixel boxes."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def filter_new(candidates, reference, iou_thr: float):
    """Keep only candidates with NO same-class match (IoU > thr) in reference."""
    kept = []
    for c in candidates:
        is_new = True
        for r in reference:
            if r["label"] == c["label"] and iou(c["bbox"], r["bbox"]) > iou_thr:
                is_new = False
                break
        if is_new:
            kept.append(c)
    return kept


def confirm_with_vlm(image, candidate, margin: int, context: float):
    """Crop the candidate box (with context) and ask the VLM to confirm/label it.

    Returns (is_confirmed, vlm_label). Saves the crop to a temp file because the
    Ollama client takes image paths (same call pattern as vlm_01/02/03).
    """
    width, height = image.size
    x0, y0, x1, y1 = candidate["bbox"]
    # Pad by a fixed margin PLUS a fraction of the box size, so small objects get
    # proportionally more of their surroundings (that context is what lets the
    # VLM tell a forgotten wallet from, say, a laptop).
    pad_x = margin + int(context * (x1 - x0))
    pad_y = margin + int(context * (y1 - y0))
    cx0 = max(0, int(x0) - pad_x)
    cy0 = max(0, int(y0) - pad_y)
    cx1 = min(width, int(x1) + pad_x)
    cy1 = min(height, int(y1) + pad_y)
    crop = image.crop((cx0, cy0, cx1, cy1))

    crop_path = _p("results/_crop_tmp.jpg")
    Path(crop_path).parent.mkdir(parents=True, exist_ok=True)
    crop.save(crop_path)

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
    is_confirmed = text.upper().lstrip().startswith("YES")
    # Keep the reply compact for the report: first line, with the leading
    # YES/NO and its punctuation stripped so only the object name remains.
    label = text.splitlines()[0].strip() if text else ""
    for prefix in ("YES", "NO"):
        if label.upper().startswith(prefix):
            label = label[len(prefix):].lstrip(" .,:;-").strip()
            break
    return is_confirmed, label


def severity_color(confidence):
    """Map a 0-1 detector confidence to a colour from green (low) to red (high)."""
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return (255, 255, 255)
    if c < 0.15:
        return (0, 200, 0)      # green
    if c < 0.30:
        return (160, 200, 0)    # yellow-green
    if c < 0.50:
        return (255, 200, 0)    # amber
    if c < 0.70:
        return (255, 120, 0)    # orange
    return (220, 0, 0)          # red


def main() -> None:
    inspection_path = _p(INSPECTION_PATH)
    if not Path(inspection_path).exists():
        print(f"ERROR: inspection image not found: {inspection_path}")
        print("Upload your images into  data/masked/  and edit INSPECTION_PATH.")
        sys.exit(1)

    reference_path = _p(REFERENCE_PATH)
    if USE_REFERENCE and not Path(reference_path).exists():
        print(f"ERROR: reference image not found: {reference_path}")
        print("Upload a clean reference into  data/reference/  and edit REFERENCE_PATH,")
        print("or set USE_REFERENCE = False in the USER CONFIG block.")
        sys.exit(1)

    if USE_VLM:
        check_model(MODEL_NAME)

    print(f"Inspection : {inspection_path}")
    print(f"Reference  : {reference_path if USE_REFERENCE else '(disabled)'}")
    print(f"Detector   : {DETECTOR_WEIGHTS}")
    print(f"VLM        : {MODEL_NAME if USE_VLM else '(disabled)'}\n")

    # --- 1) Localize ---------------------------------------------------------
    model = load_detector(DETECTOR_WEIGHTS, DETECTOR_CLASSES)
    candidates = detect(model, inspection_path, DETECTOR_CLASSES, DETECTOR_CONF, IMGSZS)
    print(f"Detector found {len(candidates)} candidate object(s) "
          f"(conf >= {DETECTOR_CONF}, scales {IMGSZS}).")

    # --- 2) Reference filtering ----------------------------------------------
    if USE_REFERENCE:
        ref_dets = detect(model, reference_path, DETECTOR_CLASSES, DETECTOR_CONF, IMGSZS)
        before = len(candidates)
        candidates = filter_new(candidates, ref_dets, REFERENCE_IOU)
        print(f"Reference filter: {before} -> {len(candidates)} new object(s) "
              f"(dropped {before - len(candidates)} already present in reference).")

    # --- 3) VLM step: label every candidate, and (in "filter" mode) also drop
    #        the ones the VLM rejects. In "label" mode nothing is dropped here.
    image = Image.open(inspection_path).convert("RGB")
    confirmed = []
    if USE_VLM:
        for c in candidates:
            ok, label = confirm_with_vlm(image, c, CROP_MARGIN, CROP_CONTEXT)
            c["vlm_label"] = label
            c["vlm_confirmed"] = ok
            if VLM_MODE == "filter" and not ok:
                continue
            confirmed.append(c)
        if VLM_MODE == "filter":
            print(f"VLM (filter): kept {len(confirmed)} of {len(candidates)} "
                  f"candidate(s).")
        else:
            n_yes = sum(1 for c in confirmed if c["vlm_confirmed"])
            print(f"VLM (label): kept all {len(confirmed)} candidate(s); "
                  f"the VLM recognized {n_yes} of them.")
    else:
        for c in candidates:
            c["vlm_label"] = ""
            c["vlm_confirmed"] = None
        confirmed = candidates

    # --- 4) Output -----------------------------------------------------------
    print("\n----- CONFIRMED OBJECTS --------------------------------------")
    if not confirmed:
        print("(none)")
    for c in confirmed:
        bbox = [round(v, 1) for v in c["bbox"]]
        print(f"  - {c['label']:<12} conf={c['confidence']:.2f}  bbox={bbox}"
              f"  vlm='{c.get('vlm_label', '')}'")
    print("-------------------------------------------------------------")

    # Draw the boxes (mirror vlm_03: colour + label above each box).
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for c in confirmed:
        x0, y0, x1, y1 = (int(v) for v in c["bbox"])
        colour = severity_color(c["confidence"])
        tag = c.get("vlm_label") or c["label"]
        label = f"{tag} ({c['confidence']:.2f})"
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=3)
        draw.text((x0 + 3, max(0, y0 - 20)), label, fill=colour, font=font)

    output_path = _p(OUTPUT_PATH)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"\nAnnotated image saved to: {output_path}")

    output_json_path = _p(OUTPUT_JSON_PATH)
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as fh:
        json.dump(confirmed, fh, indent=2, ensure_ascii=False)
    print(f"Detections JSON saved to: {output_json_path}")


if __name__ == "__main__":
    main()
