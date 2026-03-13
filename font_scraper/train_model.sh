#!/bin/bash
# train_model.sh — Build Docker image (if needed) and train the stroke model.
# Runs entirely in the background. Monitor with: ./monitor_training.sh
#
# Usage:
#   ./train_model.sh              # Build + train
#   ./train_model.sh --skip-build # Train only (image must exist)
#   ./train_model.sh --epochs 50  # Override epochs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/docker/stroke_model/logs"
CKPT_DIR="$SCRIPT_DIR/docker/stroke_model/checkpoints"
CACHE_DIR="$SCRIPT_DIR/docker/stroke_model/glyph_cache"
BUILD_LOG="$LOG_DIR/build.log"
TRAIN_LOG="$LOG_DIR/train.log"
STATUS_FILE="$LOG_DIR/status.json"

mkdir -p "$LOG_DIR" "$CKPT_DIR" "$CACHE_DIR"

# Defaults
SKIP_BUILD=false
EPOCHS=100
BATCH_SIZE=32
LR="1e-4"
RENDER_EVERY=2
NUM_WORKERS=8

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build) SKIP_BUILD=true; shift ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --render-every) RENDER_EVERY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

update_status() {
    cat > "$STATUS_FILE" <<EOJSON
{
    "stage": "$1",
    "message": "$2",
    "updated_at": "$(date -Iseconds)",
    "epochs": $EPOCHS,
    "batch_size": $BATCH_SIZE
}
EOJSON
}

# --- Build ---
if [ "$SKIP_BUILD" = false ]; then
    update_status "building" "Docker image build started"
    echo "[$(date)] Building stroke-model:latest ..."
    if docker build -t stroke-model:latest "$SCRIPT_DIR/docker/stroke_model/" > "$BUILD_LOG" 2>&1; then
        update_status "build_done" "Docker image built successfully"
        echo "[$(date)] Build complete."
    else
        update_status "build_failed" "Docker build failed — see $BUILD_LOG"
        echo "[$(date)] BUILD FAILED. See $BUILD_LOG"
        exit 1
    fi
fi

# --- Train ---
update_status "training" "Training started: $EPOCHS epochs, batch=$BATCH_SIZE"
echo "[$(date)] Starting training: $EPOCHS epochs, batch=$BATCH_SIZE, lr=$LR"

docker run --rm --gpus all \
    --name stroke-model-train \
    --shm-size=2g \
    -v "$SCRIPT_DIR/fonts:/fonts:ro" \
    -v "$SCRIPT_DIR/fonts.db:/data/fonts.db:ro" \
    -v "$CKPT_DIR:/app/checkpoints" \
    -v "$CACHE_DIR:/app/cache" \
    stroke-model:latest \
    python3 /app/train.py \
        --db /data/fonts.db \
        --font-dir / \
        --output-dir /app/checkpoints \
        --cache-dir /app/cache \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --augment \
        --num-workers "$NUM_WORKERS" \
        --log-every 50 \
        --save-every 5 \
        --render-every "$RENDER_EVERY" \
    > "$TRAIN_LOG" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    update_status "complete" "Training finished successfully"
    echo "[$(date)] Training complete."
else
    update_status "failed" "Training failed (exit $EXIT_CODE) — see $TRAIN_LOG"
    echo "[$(date)] TRAINING FAILED (exit $EXIT_CODE). See $TRAIN_LOG"
    exit $EXIT_CODE
fi
