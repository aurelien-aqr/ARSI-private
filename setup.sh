#!/usr/bin/env bash
# =============================================================================
#  ARSI-VLM - environment setup (run once on a fresh machine)
#  Target: x86 Ubuntu + NVIDIA RTX 3080 Ti (12 GB VRAM)
#  Locked VLM model: qwen2.5vl:7b
#
#      git clone https://github.com/mpyt/ARSI-vlm.git
#      cd ARSI-vlm
#      bash setup.sh
# =============================================================================
set -e

echo "==================================================="
echo "  ARSI-VLM setup  (RTX 3080 Ti, qwen2.5vl:7b)"
echo "==================================================="

# --- 1) Python virtual environment -------------------------------------------
if [ ! -d "venv" ]; then
  echo "[1/4] Creating virtual environment (venv) ..."
  python3 -m venv venv
else
  echo "[1/4] venv already exists - skipping."
fi

# --- 2) Python libraries -----------------------------------------------------
echo "[2/4] Installing Python libraries ..."
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# --- 3) Ollama server --------------------------------------------------------
echo "[3/4] Checking the Ollama server ..."
if ! command -v ollama >/dev/null 2>&1; then
  echo "      Ollama not found - installing the server ..."
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "      Ollama already installed: $(ollama --version 2>/dev/null || echo present)"
fi

# --- 4) Vision-language model ------------------------------------------------
echo "[4/4] Pulling the vision-language model (qwen2.5vl:7b) ..."
echo "      (~6 GB download on first run)"
ollama pull qwen2.5vl:7b

echo
echo "==================================================="
echo "  Done."
echo "==================================================="
echo "Activate the environment in every new terminal:"
echo "    source venv/bin/activate"
echo
echo "Put your images into:"
echo "    data/reference/   (clean reference image)"
echo "    data/raw/         (frames to inspect)"
echo "    data/masked/      (masked frames, *_masked.jpg)"
echo
echo "Then run, e.g.:"
echo "    python vlm_01_single_image.py"
echo "==================================================="
