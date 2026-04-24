#!/bin/bash
# End-to-end inference: Images/HF dataset → ChemicalOCR → MarkushGrapher → Predictions
#
# Usage:
#   bash scripts/inference/inference.sh <IMAGE_DIR>
#   bash scripts/inference/inference.sh --hf_dataset <HF_DATASET_OR_LOCAL_PATH> [--hf_config <CONFIG>] [--split <SPLIT>]
#
# Example:
#   bash scripts/inference/inference.sh ./data/images
#   bash scripts/inference/inference.sh --hf_dataset docling-project/MarkushGrapher-2-Datasets --hf_config ip5-markush --split test
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

usage() {
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
}

INPUT_MODE="images"
IMAGE_DIR="$PROJECT_DIR/data/images"
HF_DATASET=""
HF_CONFIG=""
HF_SPLIT="test"
OUTPUT_DATASET_DIR=""
TRAINING_DATASET_NAME=""
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-1000}"

while [ $# -gt 0 ]; do
    case "$1" in
        --image_dir)
            INPUT_MODE="images"
            IMAGE_DIR="$2"
            shift 2
            ;;
        --hf_dataset)
            INPUT_MODE="hf"
            HF_DATASET="$2"
            shift 2
            ;;
        --hf_config)
            HF_CONFIG="$2"
            shift 2
            ;;
        --split)
            HF_SPLIT="$2"
            shift 2
            ;;
        --output_dataset_dir)
            OUTPUT_DATASET_DIR="$2"
            shift 2
            ;;
        --training_dataset_name)
            TRAINING_DATASET_NAME="$2"
            shift 2
            ;;
        --max_eval_samples)
            MAX_EVAL_SAMPLES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            INPUT_MODE="images"
            IMAGE_DIR="$1"
            shift
            ;;
    esac
done

if [ "$INPUT_MODE" = "hf" ] && [ -z "$HF_DATASET" ]; then
    echo "ERROR: --hf_dataset is required for HF dataset inference." >&2
    exit 1
fi

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

OCR_MODEL_PATH="${OCR_MODEL_PATH:-$PROJECT_DIR/models/chemicalocr}"

RUN_STEM="sample-images"
if [ "$INPUT_MODE" = "hf" ]; then
    RUN_STEM="$(printf '%s-%s-%s' "$HF_DATASET" "$HF_CONFIG" "$HF_SPLIT" | tr -c 'A-Za-z0-9_.-' '-')"
fi
RUN_DIR="$PROJECT_DIR/data/hf/inference/$RUN_STEM-$(date +%Y%m%d-%H%M%S)"
RUN_DATASET_CONFIG="$RUN_DIR/datasets_predict.yaml"
RUN_EVAL_CONFIG="$RUN_DIR/predict.yaml"

echo "=== MarkushGrapher 2.0 End-to-End Inference ==="
echo "Input mode:       $INPUT_MODE"
if [ "$INPUT_MODE" = "hf" ]; then
    echo "HF dataset:       $HF_DATASET"
    echo "HF config:        ${HF_CONFIG:-<default>}"
    echo "HF split:         $HF_SPLIT"
else
    echo "Image directory:  $IMAGE_DIR"
fi
echo "OCR model:        $OCR_MODEL_PATH"
echo "OCR python:       $OCR_PYTHON"
echo "Model python:     $MG_PYTHON"
echo "Run directory:    $RUN_DIR"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Convert images to HF dataset and apply ChemicalOCR
# ---------------------------------------------------------------------------
mkdir -p "$RUN_DIR"

if [ "$INPUT_MODE" = "hf" ]; then
    RAW_HF_DATASET_DIR="$RUN_DIR/raw"
    HF_DATASET_DIR="${OUTPUT_DATASET_DIR:-$RUN_DIR/ocr}"
    TRAINING_DATASET_NAME="${TRAINING_DATASET_NAME:-mdu}"

    echo "[1/2] Loading HF dataset and running ChemicalOCR..."
    PREPARE_ARGS=(
        "$PROJECT_DIR/scripts/dataset/prepare_hf_dataset.py"
        --dataset "$HF_DATASET"
        --split "$HF_SPLIT"
        --output_dir "$RAW_HF_DATASET_DIR"
        --target_split test
    )
    if [ -n "$HF_CONFIG" ]; then
        PREPARE_ARGS+=(--config "$HF_CONFIG")
    fi
    PYTHONPATH="$PROJECT_DIR" "$MG_PYTHON" "${PREPARE_ARGS[@]}"

    PYTHONPATH="$PROJECT_DIR" "$OCR_PYTHON" \
        "$PROJECT_DIR/scripts/ocr/apply_ocr.py" \
        --dataset_dir "$RAW_HF_DATASET_DIR" \
        --model_path "$OCR_MODEL_PATH" \
        --output_dir "$HF_DATASET_DIR" \
        --split test
else
    HF_DATASET_DIR="${OUTPUT_DATASET_DIR:-$PROJECT_DIR/data/hf/sample-images}"
    TRAINING_DATASET_NAME="${TRAINING_DATASET_NAME:-mdu_3008_aug}"

    echo "[1/2] Converting images and running ChemicalOCR..."
    PYTHONPATH="$PROJECT_DIR" "$OCR_PYTHON" \
        "$PROJECT_DIR/scripts/dataset/image_dir_to_hf_dataset.py" \
        --image_dir "$IMAGE_DIR" \
        --output_dir "$HF_DATASET_DIR" \
        --apply_ocr \
        --ocr_model_path "$OCR_MODEL_PATH"
fi

cat > "$RUN_DATASET_CONFIG" <<EOF
mdu_dataset:
  apply_ocr: false
  augment_test: false
  class_name: MDU_Dataset
  condense_labels: true
  dataset_path: $HF_DATASET_DIR
  encode_definition_group: false
  encode_index: true
  encode_position: false
  grounded_smiles: false
  load_from_cache: false
  mask_ratio: 1
  module_name: mdu_dataset
  name: mdu
  normalize_bbox: true
  splits:
  - test
  stream: false
  task: Question Answering
  training_dataset_name: $TRAINING_DATASET_NAME
  type: supervised
  udop_tokenizer_only: false
EOF

cat > "$RUN_EVAL_CONFIG" <<EOF
{
    model_name_or_path: ./models/markushgrapher-2,
    tokenizer_path: auto,
    output_dir: $RUN_DIR/evaluation,
    datasets_config: $RUN_DATASET_CONFIG,

    max_seq_length: 512,
    image_size: 512,
    max_seq_length_decoder: 512,
    model_type: UdopUnimodel,
    architecture_variant: me-lf-stack-1,
    beam_search: True,
    normalize_bbox: True,

    use_pretrained_molscribe: True,
    freeze_ocsr_encoder: True,
    freeze_vtl_decoder: False,
    freeze_mlp_projector: False,

    do_train: False,
    do_eval: False,
    do_predict: True,
    dataloader_num_workers: 1,
    log_level: "DEBUG",
    viz_out_dir: $RUN_DIR/visualization,
    prediction_loss_only: True,
    label_names: ["labels"],
    unit: word,
    apply_ocr: False,
    max_eval_samples: $MAX_EVAL_SAMPLES,
}
EOF

# ---------------------------------------------------------------------------
# Step 2: Run MarkushGrapher inference
# ---------------------------------------------------------------------------
echo ""
echo "[2/2] Running MarkushGrapher inference..."
PYTHONPATH="$PROJECT_DIR" "$MG_PYTHON" \
    -m markushgrapher.eval "$RUN_EVAL_CONFIG"

echo ""
echo "=== Done ==="
echo "OCR dataset saved to:       $HF_DATASET_DIR"
echo "Evaluation outputs saved to: $RUN_DIR/evaluation"
echo "Evaluation config used:     $RUN_EVAL_CONFIG"
echo "Dataset config used:        $RUN_DATASET_CONFIG"
echo "Visualizations saved to:    $RUN_DIR/visualization"
