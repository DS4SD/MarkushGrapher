#!/bin/bash
# Run ChemicalOCR on a dataset directory.
#
# Usage: bash apply_ocr.sh /path/to/dataset_dir

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/dataset_dir"
  exit 1
fi

DATASET_DIR="$1"

echo "Running ChemicalOCR inference on: $DATASET_DIR"
python3 "$SCRIPT_DIR/apply_ocr.py" --dataset_dir "$DATASET_DIR"