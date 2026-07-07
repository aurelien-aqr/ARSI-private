#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - vlm_02_reference_compare.py
#  Reference-image comparison of a tram interior.
#
#  Sends TWO images to the model:
#     1) a CLEAN reference image of the empty tram interior,
#     2) the INSPECTION image to be checked (windows masked black).
#  The model reports only the DIFFERENCES (new graffiti, vandalism, objects).
#
#  Hardware target : x86 Ubuntu + NVIDIA RTX 3080 Ti (12 GB VRAM)
#  Model           : qwen2.5vl:7b  (served locally by Ollama)
#
#  Run from the repository root:   python vlm_02_reference_compare.py
# =============================================================================

import sys
from pathlib import Path
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
REFERENCE_PATH = "data/reference/tram_1762_v1_f0227_masked_reference.jpg"

# 2) Inspection image to be checked (windows already masked black).
INSPECTION_PATH = "data/masked/tram_1762_v2_f0032_masked.jpg"

# The instruction sent to the model. Feel free to experiment with the wording.
PROMPT = """You are given two images of the SAME tram interior.

- The FIRST image is the CLEAN reference (empty, undamaged).
- The SECOND image is the CURRENT inspection image. Its windows are masked
  black on purpose - ignore the black areas completely.

Compare them ZONE BY ZONE (left-hand seats, right-hand seats, floor and aisle,
far end of the car) and report EVERYTHING that is NEW or DIFFERENT in the second
image: graffiti, vandalism, or forgotten/left-behind objects. A small dark
object appearing on a seat edge or on the floor counts - look for these on
purpose. There may be SEVERAL new objects; do NOT stop after the first one.

Answer ONLY in the following structured format:

GRAFFITI: <yes/no> - short note
VANDALISM: <yes/no> - short note
FORGOTTEN OBJECT: <yes/no>
  If yes, list EVERY new object, one per line, as:
  - <what it is> (<which seat / zone>)
DESCRIPTION: <one or two sentences describing the differences>
SEVERITY: <1-5, where 1 = nothing wrong, 5 = serious problem>
"""

# =============================================================================
#  HARDWARE-LOCK  ---  DO NOT CHANGE
# =============================================================================
MODEL_NAME  = "qwen3.5:9b"   # locked to this machine's GPU (RTX 3080 Ti)
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

    if model_name not in names:
        print(f"ERROR: model '{model_name}' is not installed.")
        print(f"Install it with:  ollama pull {model_name}")
        sys.exit(1)


def main() -> None:
    check_model(MODEL_NAME)

    reference_path = _p(REFERENCE_PATH)
    inspection_path = _p(INSPECTION_PATH)
    for label, path in (("reference", reference_path), ("inspection", inspection_path)):
        if not Path(path).exists():
            print(f"ERROR: {label} image not found: {path}")
            sys.exit(1)

    print(f"Reference  : {reference_path}")
    print(f"Inspection : {inspection_path}")
    print(f"Model      : {MODEL_NAME}\n")

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": PROMPT,
            # Order matters: reference first, inspection second.
            "images": [reference_path, inspection_path],
        }],
        think=False,  # skip chain-of-thought; write the answer straight into `content`
        options={
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "temperature": TEMPERATURE,
        },
    )

    print("----- MODEL OUTPUT -------------------------------------------")
    print(response["message"]["content"].strip())
    print("-------------------------------------------------------------")


if __name__ == "__main__":
    main()
