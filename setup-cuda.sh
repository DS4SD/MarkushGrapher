#!/bin/bash
# MarkushGrapher 2.0 — CUDA Setup Script (UV Version)
#
# Usage:
#   bash setup-cuda.sh
#
# Creates TWO virtual environments:
#   - chemicalocr-env  : vllm + stock transformers (fast ChemicalOCR on GPU)
#   - markushgrapher-env: custom transformers fork  (MarkushGrapher model inference)
#
# Why two environments?
#   vllm >= 0.6.x requires tokenizers >= 0.19, but the custom transformers fork
#   (which contains the MarkushGrapher model class) requires tokenizers < 0.14.
#   These ranges do not overlap, so a single environment cannot satisfy both.
#   The two inference stages are already separate processes communicating via
#   HuggingFace dataset files on disk, so using two envs is a clean fit.
#
# After setup, run inference with:
#   bash scripts/inference/inference.sh ./data/images

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export UV_CACHE_DIR="/tmp/${USER:-uv}-cache"
export UV_LINK_MODE=copy
mkdir -p "$UV_CACHE_DIR"

echo "=== MarkushGrapher 2.0 CUDA Setup (UV) ==="
echo ""

# ---------------------------------------------------------------------------
# Ensure uv is available
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv is not installed. Please install it first." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Environment A: chemicalocr-env
# vllm + stock transformers — no custom fork needed here
# ---------------------------------------------------------------------------
echo "[1/6] Setting up chemicalocr-env (vllm + stock transformers)..."
if [ ! -d "chemicalocr-env" ]; then
    uv venv --python 3.10 chemicalocr-env
fi

UV_PIP_A="uv pip install --python chemicalocr-env/bin/python"

# Install the markushgrapher package for Chemical_OCR and markushgenerator,
# but ignore its torch==2.2.0 pin — vllm needs torch >= 2.4
$UV_PIP_A -e . --no-deps -q
$UV_PIP_A \
    "torch>=2.4" torchvision torchaudio \
    "transformers>=4.46" "tokenizers>=0.19" \
    "datasets" "pillow" "tqdm" "huggingface_hub" \
    "markushgenerator @ git+https://git@github.com/DS4SD/MarkushGenerator.git" \
    "aiohttp" "s3fs" "fsspec>=2024.6.1" \
    "numpy<2" \
    -q

# Install vllm — requires CUDA to be available at install time
$UV_PIP_A vllm -q

echo "       chemicalocr-env ready."
echo ""

# ---------------------------------------------------------------------------
# Environment B: markushgrapher-env
# Custom transformers fork — same as setup.sh (no vllm)
# ---------------------------------------------------------------------------
echo "[2/6] Setting up markushgrapher-env (custom transformers fork)..."
if [ ! -d "markushgrapher-env" ]; then
    uv venv --python 3.10 markushgrapher-env
fi

UV_PIP_B="uv pip install --python markushgrapher-env/bin/python"

$UV_PIP_B -e . -q

echo "[3/6] Installing transformers fork and MolScribe..."
if [ ! -d "external/transformers" ]; then
    git clone --quiet https://github.com/lucas-morin/transformers.git ./external/transformers
fi
if [ ! -d "external/MolScribe" ]; then
    git clone --quiet https://github.com/lucas-morin/MolScribe.git ./external/MolScribe
fi
$UV_PIP_B -e ./external/MolScribe --no-deps -q

echo "[4/6] Pinning numpy<2, pyonmttok==1.37.1, OpenNMT-py==2.2.0..."
$UV_PIP_B "numpy<2" "pyonmttok==1.37.1" "OpenNMT-py==2.2.0" -q
# Reinstall the transformers fork last so it wins over anything pulled above
$UV_PIP_B -e ./external/transformers -q

echo "       markushgrapher-env ready."
echo ""

# ---------------------------------------------------------------------------
# Download model weights (shared)
# ---------------------------------------------------------------------------
echo "[5/6] Downloading model weights..."
source markushgrapher-env/bin/activate

if [ ! -d "models/markushgrapher-2" ]; then
    huggingface-cli download docling-project/MarkushGrapher-2 --local-dir ./models/markushgrapher-2
else
    echo "       MarkushGrapher-2 weights already downloaded."
fi

if [ ! -d "models/chemicalocr" ]; then
    huggingface-cli download docling-project/ChemicalOCR --local-dir ./models/chemicalocr
else
    echo "       ChemicalOCR weights already downloaded."
fi

if [ ! -f "external/MolScribe/ckpts/swin_base_char_aux_1m680k.pth" ]; then
    mkdir -p external/MolScribe/ckpts
    wget -q https://huggingface.co/yujieq/MolScribe/resolve/main/swin_base_char_aux_1m680k.pth \
        -P ./external/MolScribe/ckpts/
else
    echo "       MolScribe weights already downloaded."
fi

deactivate

echo ""
echo "[6/6] Done."
echo ""
echo "=== CUDA Setup complete ==="
echo ""
echo "To run inference:"
echo "  bash scripts/inference/inference.sh ./data/images"
echo ""
echo "The inference script automatically uses:"
echo "  chemicalocr-env  for ChemicalOCR (vllm on GPU)"
echo "  markushgrapher-env for MarkushGrapher inference (GPU)"
