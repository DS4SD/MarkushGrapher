#!/bin/bash
# End-to-end inference: Images → ChemicalOCR → MarkushGrapher → Predictions
#
# Usage: bash scripts/inference/inference.sh <IMAGE_DIR>
#
# Example:
#   bash scripts/inference/inference.sh ./data/images
#
# The two inference stages use separate Python interpreters so that each can
# have its own dependency set:
#
#   Stage 1 — ChemicalOCR:
#     Default: markushgrapher-env (mlx on Apple Silicon, transformers fallback on CPU/GPU)
#     CUDA fast: chemicalocr-env  (vllm on GPU — set up via setup-cuda.sh)
#     Override:  CHEMICALOCR_PYTHON=/path/to/python
#
#   Stage 2 — MarkushGrapher:
#     Always:    markushgrapher-env (custom transformers fork with the model class)
#     Override:  MARKUSHGRAPHER_PYTHON=/path/to/python

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------------------------------------------------------------------------
# Resolve Python interpreters
# ---------------------------------------------------------------------------

# Stage 2 always uses the markushgrapher-env (needs the custom transformers fork)
if [ -n "$MARKUSHGRAPHER_PYTHON" ]; then
    MG_PYTHON="$MARKUSHGRAPHER_PYTHON"
elif [ -f "$PROJECT_DIR/markushgrapher-env/bin/python" ]; then
    MG_PYTHON="$PROJECT_DIR/markushgrapher-env/bin/python"
else
    MG_PYTHON="python3"
fi

# Stage 1 uses chemicalocr-env when available (enables vllm on CUDA),
# falls back to markushgrapher-env (mlx / transformers backends)
if [ -n "$CHEMICALOCR_PYTHON" ]; then
    OCR_PYTHON="$CHEMICALOCR_PYTHON"
elif [ -f "$PROJECT_DIR/chemicalocr-env/bin/python" ]; then
    OCR_PYTHON="$PROJECT_DIR/chemicalocr-env/bin/python"
else
    OCR_PYTHON="$MG_PYTHON"
fi

IMAGE_DIR="${1:-$PROJECT_DIR/data/images}"
OCR_MODEL_PATH="${OCR_MODEL_PATH:-$PROJECT_DIR/models/chemicalocr}"
HF_DATASET_DIR="$PROJECT_DIR/data/hf/sample-images"

echo "=== MarkushGrapher 2.0 End-to-End Inference ==="
echo "Image directory:  $IMAGE_DIR"
echo "OCR model:        $OCR_MODEL_PATH"
echo "OCR python:       $OCR_PYTHON"
echo "Model python:     $MG_PYTHON"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Convert images to HF dataset and apply ChemicalOCR
# ---------------------------------------------------------------------------
echo "[1/2] Converting images and running ChemicalOCR..."
PYTHONPATH="$PROJECT_DIR" "$OCR_PYTHON" \
    "$PROJECT_DIR/scripts/dataset/image_dir_to_hf_dataset.py" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$HF_DATASET_DIR" \
    --apply_ocr \
    --ocr_model_path "$OCR_MODEL_PATH"

# ---------------------------------------------------------------------------
# Step 2: Run MarkushGrapher inference
# ---------------------------------------------------------------------------
echo ""
echo "[2/2] Running MarkushGrapher inference..."
PYTHONPATH="$PROJECT_DIR" "$MG_PYTHON" \
    -m markushgrapher.eval "$PROJECT_DIR/config/predict.yaml"

echo ""
echo "=== Done ==="
echo "Visualizations saved to: $PROJECT_DIR/data/visualization/prediction/"
