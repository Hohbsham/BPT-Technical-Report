#!/bin/bash
# ============================================================
# BPT Fine-tuning — Cloud Launch Script
# ============================================================
# Usage:
#   bash run_cloud.sh [--max_code_len 4000] [--epochs 50] ...
#
# Prerequisites:
#   1. Upload project to cloud:  rsync -avz bpt/ user@cloud:/workspace/bpt/
#   2. Upload data:              rsync -avz supervised_data/ user@cloud:/workspace/supervised_data/
#   3. Upload cached embeddings: rsync -avz cached_embeddings/ user@cloud:/workspace/cached_embeddings/
#   4. Install deps:             pip install -r requirements_cloud.txt
# ============================================================

set -euo pipefail

# ---- Configuration ----
MODEL_PATH=${MODEL_PATH:-"weights/bpt-8-16-500m.pt"}
DATA_DIR=${DATA_DIR:-"./supervised_data"}
CACHE_DIR=${CACHE_DIR:-"./cached_embeddings"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints"}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-1e-4}
MAX_CODE_LEN=${MAX_CODE_LEN:-4000}
SAVE_EVERY=${SAVE_EVERY:-5}
NUM_WORKERS=${NUM_WORKERS:-4}

# ---- Auto-detect GPU count ----
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "1")
echo "Detected GPUs: $GPU_COUNT"

# ---- Launch ----
echo "=========================================="
echo " BPT Fine-Tuning (Cloud)"
echo "=========================================="
echo " Model:       $MODEL_PATH"
echo " Data:        $DATA_DIR"
echo " Cache:       $CACHE_DIR"
echo " Output:      $OUTPUT_DIR"
echo " Epochs:      $EPOCHS"
echo " Batch size:  $BATCH_SIZE x $GRAD_ACCUM grad accum"
echo " Max tokens:  $MAX_CODE_LEN"
echo " LR:          $LR"
echo " GPUs:        $GPU_COUNT"
echo "=========================================="

python train_cloud.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum_steps "$GRAD_ACCUM" \
    --lr "$LR" \
    --max_code_len "$MAX_CODE_LEN" \
    --save_every "$SAVE_EVERY" \
    --num_workers "$NUM_WORKERS" \
    "$@"

echo "Done."
