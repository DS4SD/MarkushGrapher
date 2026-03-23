#!/bin/bash
# End-to-end inference: Images → ChemicalOCR → MarkushGrapher → Predictions
#
# Usage: bash scripts/inference/inference.sh <IMAGE_DIR>
#
# Example:
#   bash scripts/inference/inference.sh ./data/images
#
# The script will:
#   1. Convert images to HuggingFace dataset format
#   2. Run ChemicalOCR to extract text and bounding boxes
#   3. Run MarkushGrapher to predict CXSMILES and substituent tables
#   4. Save visualizations to data/visualization/prediction/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

IMAGE_DIR="${1:-$PROJECT_DIR/data/images}"
OCR_MODEL_PATH="${OCR_MODEL_PATH:-$PROJECT_DIR/models/chemicalocr}"
HF_DATASET_DIR="$PROJECT_DIR/data/hf/sample-images"

echo "=== MarkushGrapher 2.0 End-to-End Inference ==="
echo "Image directory:  $IMAGE_DIR"
echo "OCR model:        $OCR_MODEL_PATH"
echo ""

# Step 1: Convert images to HF dataset and apply ChemicalOCR
echo "[1/2] Converting images and running ChemicalOCR..."
python3 "$PROJECT_DIR/scripts/dataset/image_dir_to_hf_dataset.py" \
  --image_dir "$IMAGE_DIR" \
  --output_dir "$HF_DATASET_DIR" \
  --apply_ocr \
  --ocr_model_path "$OCR_MODEL_PATH"

# Step 2: Run MarkushGrapher inference
echo ""
echo "[2/2] Running MarkushGrapher inference..."
python3 -m markushgrapher.eval "$PROJECT_DIR/config/predict.yaml"

echo ""
echo "=== Done ==="
echo "Visualizations saved to: $PROJECT_DIR/data/visualization/prediction/"
