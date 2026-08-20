#!/usr/bin/env bash
# =============================================================================
#  ARSI-VLM - environment setup (run once on a fresh machine)
#  Target: x86 Ubuntu + NVIDIA RTX 3080 Ti (12 GB VRAM)
#  Judge model: haervwe/GLM-4.6V-Flash-9B (measured winner, docs/DECISIONS.md)
#
#      git clone git@github.com:aurelien-aqr/ARSI-private.git ARSI-vlm
#      cd ARSI-vlm
#      bash setup.sh
# =============================================================================
set -e

echo "==================================================="
echo "  ARSI-VLM setup  (RTX 3080 Ti, GLM-4.6V-Flash-9B)"
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
echo "[4/4] Pulling the judge model (haervwe/GLM-4.6V-Flash-9B) ..."
echo "      (~8 GB download on first run)"
ollama pull haervwe/GLM-4.6V-Flash-9B

echo
echo "==================================================="
echo "  Done."
echo "==================================================="
echo "Start ARSI Studio from the repository root:"
echo "    venv/bin/python -m uvicorn app.backend.main:app --port 8321"
echo "    then open http://localhost:8321"
echo
echo "The standalone vlm_0x scripts use the same model by default."
echo "==================================================="
