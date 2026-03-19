#!/bin/bash
# restart_training.sh — Stop training, wipe all state, and start fresh from epoch 0.
#
# Usage:
#   ./restart_training.sh
#   ./restart_training.sh --dry-run   # Show what would be deleted without doing it

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STROKE_DIR="$SCRIPT_DIR/docker/stroke_model"
CKPT_DIR="$STROKE_DIR/checkpoints"
CONTAINER_NAME="stroke-model-train"

DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
    echo "[dry-run] Would perform the following:"
fi

run() {
    if $DRY_RUN; then
        echo "  $*"
    else
        eval "$@"
    fi
}

# 1. Sync current experiment before destroying anything
SNAPSHOT_SCRIPT="$SCRIPT_DIR/snapshot_experiment.sh"
if [ -x "$SNAPSHOT_SCRIPT" ] && docker ps --filter "name=$CONTAINER_NAME" --format '{{.ID}}' 2>/dev/null | grep -q .; then
    echo "Syncing current experiment before restart..."
    "$SNAPSHOT_SCRIPT" --sync 2>/dev/null || true
fi

# 2. Stop and remove container
if docker ps -a --filter "name=$CONTAINER_NAME" --format '{{.ID}}' 2>/dev/null | grep -q .; then
    echo "Stopping container $CONTAINER_NAME..."
    run "docker stop $CONTAINER_NAME 2>/dev/null || true"
    run "docker rm $CONTAINER_NAME 2>/dev/null || true"
else
    echo "No container to stop."
fi

# 3. Clean checkpoints, tracking, tensorboard (owned by root from Docker)
echo "Cleaning checkpoints and tracking data..."
if docker image inspect stroke-model:latest &>/dev/null; then
    run "docker run --rm -v '$CKPT_DIR:/ckpt' stroke-model:latest bash -c 'rm -rf /ckpt/*.pt /ckpt/tracking /ckpt/runs /ckpt/samples'"
else
    echo "WARNING: stroke-model image not found, trying direct rm..."
    run "rm -rf '$CKPT_DIR'/*.pt '$CKPT_DIR/tracking' '$CKPT_DIR/runs' '$CKPT_DIR/samples'"
fi

# 4. Snapshot new experiment before launching
echo "Creating experiment snapshot..."
if [ -x "$SNAPSHOT_SCRIPT" ]; then
    EXP_DIR=$("$SNAPSHOT_SCRIPT" --init "${NOTES:-}" 2>/dev/null || echo "")
    if [ -n "$EXP_DIR" ]; then
        echo "Experiment: $EXP_DIR"
    fi
fi

# 5. Launch fresh training
echo "Launching fresh training from epoch 0..."
run "docker run -d --gpus all --name $CONTAINER_NAME \
  --shm-size=2g \
  -v '$SCRIPT_DIR/fonts.db:/data/fonts.db:ro' \
  -v '$SCRIPT_DIR:/fontdata:ro' \
  -v '$CKPT_DIR:/app/checkpoints' \
  -v '$STROKE_DIR/glyph_cache:/app/cache' \
  -v '$STROKE_DIR/model.py:/app/model.py:ro' \
  -v '$STROKE_DIR/losses.py:/app/losses.py:ro' \
  -v '$STROKE_DIR/dataset.py:/app/dataset.py:ro' \
  -v '$STROKE_DIR/train.py:/app/train.py:ro' \
  -v '$STROKE_DIR/triton_render.py:/app/triton_render.py:ro' \
  stroke-model:latest \
  python3 -u /app/train.py \
    --db /data/fonts.db \
    --font-dir /fontdata \
    --output-dir /app/checkpoints \
    --cache-dir /app/cache \
    --epochs 100 \
    --batch-size 96 \
    --lr 1e-4 \
    --num-workers 4 \
    --log-every 50 \
    --save-every 5 \
    --render-every 1 \
    --augment \
    --loss-weights '{\"canvas_mse\": 1.0, \"stroke_length\": 0.1, \"exist_decay\": 0.0}'"

if ! $DRY_RUN; then
    echo ""
    echo "Waiting for training to start..."
    sleep 10
    if docker ps --filter "name=$CONTAINER_NAME" --format '{{.Status}}' 2>/dev/null | grep -q "Up"; then
        echo "Training started successfully."
        docker logs "$CONTAINER_NAME" 2>&1 | tail -5
    else
        echo "ERROR: Container failed to start. Logs:"
        docker logs "$CONTAINER_NAME" 2>&1 | tail -20
    fi
    echo ""
    echo "Monitor with: watch -n 1 $SCRIPT_DIR/monitor_training.sh"
fi
