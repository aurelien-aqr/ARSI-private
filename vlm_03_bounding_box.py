#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - vlm_03_bounding_box.py
#  Bounding-box detection of anomalies in a tram interior.
#
#  Sends ONE image to the model and asks for a JSON list of detected anomalies
#  (graffiti / vandalism / forgotten objects), each with a normalized bounding
#  box and a severity score. Boxes are drawn with Pillow and the annotated
#  image is saved into results/.
#
#  Hardware target : x86 Ubuntu + NVIDIA RTX 3080 Ti (12 GB VRAM)
#  Model           : qwen2.5vl:7b  (served locally by Ollama)
#
#  Run from the repository root:   python vlm_03_bounding_box.py
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

# Path to the image you want to analyse (relative to the repo root).
IMAGE_PATH = "data/raw/tram_1762_v2_f0037.jpg"

# Where to save the annotated image (with boxes drawn on it).
OUTPUT_PATH = "results/tram_1762_v2_f0037_annotated.jpg"

# The instruction sent to the model. Feel free to experiment with the wording.
PROMPT = """Detect ALL anomalies inside this tram interior: graffiti, vandalism,
and forgotten/left-behind objects (bags, backpacks, bottles, phones, wallets,
packages).

Scan the whole image zone by zone - left-hand seats, right-hand seats, the floor
and aisle, and the far end of the car. Report EVERY object separately, including
small dark ones on a seat edge or on the floor; do NOT stop after the first one.

Return ONLY a JSON array. Each detected item is one element:
{"label": "graffiti|vandalism|forgotten_object",
 "bbox": [x0, y0, x1, y1],
 "severity": <1-5>}

Rules:
- ONE array element PER object (there are often several).
- Coordinates are NORMALIZED between 0 and 1 (top-left origin):
  x0,y0 = top-left corner, x1,y1 = bottom-right corner.
- severity: 1 = minor, 5 = serious.
- If nothing is found, return an empty array: []
- Do NOT write any text outside the JSON array.
"""

# =============================================================================
#  MODEL / RUNTIME  ---  MODEL_NAME is only the DEFAULT: override with --model
#  (the ctx/predict/temperature values are tuned for 8-9B models on the target
#  RTX 3080 Ti and are shared by every model in the benchmark grid)
# =============================================================================
MODEL_NAME  = "qwen3-vl:8b-instruct"
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


def severity_color(severity):
    """Map a 1-5 severity score to a colour from green (low) to red (high)."""
    palette = {
        1: (0, 200, 0),      # green
        2: (160, 200, 0),    # yellow-green
        3: (255, 200, 0),    # amber
        4: (255, 120, 0),    # orange
        5: (220, 0, 0),      # red
    }
    try:
        return palette.get(int(severity), (255, 255, 255))
    except (TypeError, ValueError):
        return (255, 255, 255)


def parse_json(raw: str):
    """Extract a JSON array from the model output.

    The model sometimes wraps the JSON in Markdown code fences (```json ... ```)
    or adds stray text, so we strip the fences and slice to the array bounds.
    """
    text = raw.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("WARNING: could not parse the model output as JSON.")
        print("Raw output was:\n", raw)
        return []


def main() -> None:
    check_model(MODEL_NAME)

    image_path = _p(IMAGE_PATH)
    if not Path(image_path).exists():
        print(f"ERROR: image not found: {image_path}")
        print("Upload your images into  data/raw/  and edit IMAGE_PATH.")
        sys.exit(1)

    print(f"Analysing : {image_path}")
    print(f"Model     : {MODEL_NAME}\n")

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": PROMPT,
            "images": [image_path],
        }],
        think=False,  # skip chain-of-thought; write the answer straight into `content`
        options={
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "temperature": TEMPERATURE,
        },
    )

    raw = response["message"]["content"]
    detections = parse_json(raw)

    print(f"Detections: {len(detections)}")
    for d in detections:
        print(f"  - {d.get('label')}  severity={d.get('severity')}  bbox={d.get('bbox')}")

    # --- Draw the boxes ------------------------------------------------------
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    # Figure out the coordinate scale the model actually used. Despite the prompt
    # asking for 0-1 normalized values, the Qwen family returns boxes on a 0-1000
    # scale (or in absolute pixels). We detect it from the largest coordinate so
    # the boxes land on the image instead of ~500x off-screen.
    all_coords = [c for d in detections for c in d.get("bbox", [])
                  if isinstance(c, (int, float))]
    max_coord = max(all_coords) if all_coords else 0.0
    if max_coord <= 1.0:
        scale_x, scale_y = float(width), float(height)   # 0-1 normalized
    elif max_coord <= 1000.0:
        scale_x, scale_y = width / 1000.0, height / 1000.0  # 0-1000 (Qwen default)
    else:
        scale_x = scale_y = 1.0                          # already absolute pixels
    print(f"(bbox scale detected: max coord = {max_coord:g} -> "
          f"{'0-1' if max_coord <= 1 else '0-1000' if max_coord <= 1000 else 'pixels'})")

    for d in detections:
        bbox = d.get("bbox", [])
        if len(bbox) != 4:
            continue
        x0 = int(bbox[0] * scale_x)
        y0 = int(bbox[1] * scale_y)
        x1 = int(bbox[2] * scale_x)
        y1 = int(bbox[3] * scale_y)

        severity = d.get("severity", 1)
        colour = severity_color(severity)
        label = f"{d.get('label', '?')} ({severity})"

        draw.rectangle([x0, y0, x1, y1], outline=colour, width=3)
        draw.text((x0 + 3, max(0, y0 - 20)), label, fill=colour, font=font)

    output_path = _p(OUTPUT_PATH)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"\nAnnotated image saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Bounding-box tram anomaly detection via a local VLM. "
                    "Defaults come from the USER CONFIG block in this file.")
    ap.add_argument("--model", default=MODEL_NAME,
                    help="Ollama model name (default: %(default)s)")
    ap.add_argument("--image", default=IMAGE_PATH,
                    help="image to analyse (default: %(default)s)")
    ap.add_argument("--output", default=OUTPUT_PATH,
                    help="annotated image output path (default: %(default)s)")
    ap.add_argument("--prompt-file", default=None,
                    help="read the prompt from this text file instead of PROMPT")
    args = ap.parse_args()
    MODEL_NAME = args.model
    IMAGE_PATH = args.image
    OUTPUT_PATH = args.output
    if args.prompt_file:
        PROMPT = Path(args.prompt_file).read_text(encoding="utf-8")
    main()
