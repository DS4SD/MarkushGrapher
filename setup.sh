#!/bin/bash
# MarkushGrapher 2.0 — Full Setup Script
#
# Usage:
#   bash setup.sh
#
# This script will:
#   1. Create a virtual environment (markushgrapher-env)
#   2. Install MarkushGrapher and dependencies
#   3. Clone and install required forks (transformers, MolScribe)
#   4. Install mlx-vlm on Apple Silicon (optional, for fast ChemicalOCR)
#   5. Download model weights and MolScribe checkpoint

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== MarkushGrapher 2.0 Setup ==="
echo ""

# Step 1: Virtual environment
if [ ! -d "markushgrapher-env" ]; then
    echo "[1/5] Creating virtual environment..."
    python3.10 -m venv markushgrapher-env
else
    echo "[1/5] Virtual environment already exists, skipping."
fi
source markushgrapher-env/bin/activate

# Step 2: Install MarkushGrapher
echo "[2/5] Installing MarkushGrapher..."
PIP_USE_PEP517=0 pip install -e . -q

# Step 3: Clone and install forks
echo "[3/5] Installing transformers fork and MolScribe..."
if [ ! -d "external/transformers" ]; then
    git clone --quiet https://github.com/lucas-morin/transformers.git ./external/transformers
fi
pip install -e ./external/transformers -q

if [ ! -d "external/MolScribe" ]; then
    git clone --quiet https://github.com/lucas-morin/MolScribe.git ./external/MolScribe
fi
pip install -e ./external/MolScribe --no-deps -q

# Step 4: Apple Silicon — install mlx-vlm
echo "[4/5] Checking for Apple Silicon..."
if python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    echo "       Apple Silicon detected, installing mlx-vlm..."
    pip install mlx-vlm -q
else
    echo "       Not Apple Silicon (or MPS unavailable), skipping mlx-vlm."
fi

# Step 5: Download model weights
echo "[5/5] Downloading model weights..."
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
    wget -q https://huggingface.co/yujieq/MolScribe/resolve/main/swin_base_char_aux_1m680k.pth -P ./external/MolScribe/ckpts/
else
    echo "       MolScribe weights already downloaded."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the environment:"
echo "  source markushgrapher-env/bin/activate"
echo ""
echo "To run inference:"
echo "  bash scripts/inference/inference.sh ./data/images"
