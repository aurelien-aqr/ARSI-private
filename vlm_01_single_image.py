#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - vlm_01_single_image.py
#  Standalone single-image anomaly analysis of a tram interior.
#
#  Sends ONE image to a local vision-language model (VLM) via Ollama and asks
#  it to report graffiti, vandalism and forgotten objects in a fixed format.
#
#  Hardware target : x86 Ubuntu + NVIDIA RTX 3080 Ti (12 GB VRAM)
#  Model           : qwen2.5vl:7b  (served locally by Ollama)
#
#  Run from the repository root:   python vlm_01_single_image.py
# =============================================================================

import sys
from pathlib import Path
import ollama

# Resolve every path relative to the repository root (= this file's folder),
# so the script works no matter which folder you launch it from. [not USER CONFIG]
REPO_ROOT = Path(__file__).resolve().parent
def _p(path):
    p = Path(path)
    return str(p if p.is_absolute() else REPO_ROOT / p)

# =============================================================================
#  USER CONFIG  ---  the ONLY part you are meant to edit
# =============================================================================

# Path to the image you want to analyse (relative to the repo root).
# IMAGE_PATH = "data/raw/tram_1762_v1_f0001.jpg" # nothing
IMAGE_PATH = "data/raw/tram_1762_v2_f0037.jpg" # phone, wallet and bag

# The instruction sent to the model. Feel free to experiment with the wording.
PROMPT = """You are inspecting the interior of a tram for problems: graffiti,
vandalism, and forgotten/left-behind objects (bags, backpacks, bottles, phones,
wallets, packages, etc.).

Inspect the image METHODICALLY, one zone at a time: the left-hand seats, the
right-hand seats, the floor and the aisle, and the far end of the car. Small
dark objects on a seat edge or on the floor are easy to miss - look for them on
purpose. There may be SEVERAL objects; do NOT stop after the first one.

Answer ONLY in the following structured format:

GRAFFITI: <yes/no> - short note
VANDALISM: <yes/no> - short note
FORGOTTEN OBJECT: <yes/no>
  If yes, list EVERY object found, one per line, as:
  - <what it is> (<which seat / zone>)
DESCRIPTION: <one or two sentences describing what you see>
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

    message = response.get("message", {})
    content = message.get("content", "")
    thinking = message.get("thinking", "")

    output = content.strip() or thinking.strip()

    print("----- MODEL OUTPUT -------------------------------------------")
    print(output if output else "[EMPTY OUTPUT]")
    print("-------------------------------------------------------------")


if __name__ == "__main__":
    main()
