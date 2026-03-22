#!/bin/bash
# restart_training.sh — Stop training and start fresh or resume from checkpoint.
#
# Usage:
#   ./restart_training.sh              # Start from saved pretrain checkpoint (default)
#   ./restart_training.sh --pretrain   # Force fresh pretraining from scratch
#   ./restart_training.sh --resume     # Pick a checkpoint to resume from
#   ./restart_training.sh --dry-run    # Show what would happen without doing it

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STROKE_DIR="$SCRIPT_DIR/docker/stroke_model"
CKPT_DIR="$STROKE_DIR/checkpoints"
CONTAINER_NAME="stroke-model-train"
SNAPSHOT_SCRIPT="$SCRIPT_DIR/snapshot_experiment.sh"

DRY_RUN=false
RESUME_MODE=false
RESUME_CKPT=""
FORCE_PRETRAIN=false
SAVED_PRETRAIN="$CKPT_DIR/saved/pretrain_complete.pt"

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --dry-run)    DRY_RUN=true; echo "[dry-run] Would perform the following:" ;;
        --resume)     RESUME_MODE=true ;;
        --pretrain)   FORCE_PRETRAIN=true ;;
    esac
done

run() {
    if $DRY_RUN; then
        echo "  $*"
    else
        eval "$@"
    fi
}

# Handle --resume: list checkpoints and let user pick
if $RESUME_MODE; then
    echo "═══ Available Checkpoints ═══"
    echo ""

    # List all .pt files with details (owned by root, use docker to stat)
    CKPTS=$(find "$CKPT_DIR" -maxdepth 1 -name "*.pt" 2>/dev/null | sort)

    if [ -z "$CKPTS" ]; then
        echo "  No checkpoints found in $CKPT_DIR"
        echo "  Run without --resume to start fresh."
        exit 1
    fi

    # Display numbered list
    i=1
    declare -a CKPT_ARRAY
    while IFS= read -r ckpt; do
        name=$(basename "$ckpt")
        size=$(du -h "$ckpt" 2>/dev/null | cut -f1)
        # Try to extract epoch from checkpoint
        epoch_info=""
        if echo "$name" | grep -qP 'epoch\d+'; then
            epoch_info=" ($(echo "$name" | grep -oP 'epoch\d+'))"
        elif [ "$name" = "best_model.pt" ]; then
            epoch_info=" (best)"
        fi
        printf "  [%d] %s  (%s)%s\n" "$i" "$name" "$size" "$epoch_info"
        CKPT_ARRAY[$i]="$ckpt"
        i=$((i + 1))
    done <<< "$CKPTS"

    echo ""
    read -rp "Select checkpoint [1-$((i-1))]: " choice

    if [ -z "$choice" ] || [ "$choice" -lt 1 ] || [ "$choice" -ge "$i" ] 2>/dev/null; then
        echo "Invalid selection."
        exit 1
    fi

    RESUME_CKPT="${CKPT_ARRAY[$choice]}"
    RESUME_NAME=$(basename "$RESUME_CKPT")
    echo ""
    echo "Resuming from: $RESUME_NAME"

    # Extract epoch number from checkpoint to delete later ones
    RESUME_EPOCH=""
    if echo "$RESUME_NAME" | grep -qP 'epoch(\d+)'; then
        RESUME_EPOCH=$(echo "$RESUME_NAME" | grep -oP 'epoch\K\d+')
    elif [ "$RESUME_NAME" = "best_model.pt" ]; then
        # Read epoch from checkpoint metadata
        RESUME_EPOCH=$(python3 -c "
import torch, sys
try:
    ckpt = torch.load('$RESUME_CKPT', map_location='cpu', weights_only=False)
    print(ckpt.get('epoch', ''))
except:
    print('')
" 2>/dev/null || echo "")
    fi

    # Delete checkpoints AFTER the resume point
    if [ -n "$RESUME_EPOCH" ]; then
        echo "Cleaning checkpoints after epoch $RESUME_EPOCH..."
        for ckpt in "$CKPT_DIR"/checkpoint_epoch*.pt; do
            [ -f "$ckpt" ] || continue
            ckpt_epoch=$(basename "$ckpt" | grep -oP 'epoch\K\d+' || true)
            if [ -n "$ckpt_epoch" ] && [ "$ckpt_epoch" -gt "$RESUME_EPOCH" ]; then
                echo "  Removing: $(basename "$ckpt")"
                run "rm -f '$ckpt'"
            fi
        done

        # Delete tracking images after resume epoch
        for tdir in "$CKPT_DIR"/tracking/epoch_*; do
            [ -d "$tdir" ] || continue
            t_epoch=$(basename "$tdir" | grep -oP '\d+' || true)
            if [ -n "$t_epoch" ] && [ "$t_epoch" -gt "$RESUME_EPOCH" ]; then
                echo "  Removing tracking: $(basename "$tdir")"
                run "rm -rf '$tdir'"
            fi
        done
    fi
fi

# 1. Sync current experiment before destroying anything
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

# 3. If fresh start, clean everything
if ! $RESUME_MODE; then
    echo "Cleaning checkpoints and tracking data..."
    if docker image inspect stroke-model:latest &>/dev/null; then
        run "docker run --rm -v '$CKPT_DIR:/ckpt' stroke-model:latest bash -c 'rm -rf /ckpt/*.pt /ckpt/tracking /ckpt/runs /ckpt/samples'"
    else
        echo "WARNING: stroke-model image not found, trying direct rm..."
        run "rm -rf '$CKPT_DIR'/*.pt '$CKPT_DIR/tracking' '$CKPT_DIR/runs' '$CKPT_DIR/samples'"
    fi
fi

# 4. Snapshot new experiment before launching
echo "Creating experiment snapshot..."
if [ -x "$SNAPSHOT_SCRIPT" ]; then
    EXP_DIR=$("$SNAPSHOT_SCRIPT" --init "${NOTES:-}" 2>/dev/null || echo "")
    if [ -n "$EXP_DIR" ]; then
        echo "Experiment: $EXP_DIR"
    fi
fi

# 5. Build resume flag and pretrain flag
RESUME_FLAG=""
PRETRAIN_FLAG=""

if $RESUME_MODE && [ -n "$RESUME_CKPT" ]; then
    # Resuming from user-selected checkpoint
    CKPT_BASENAME=$(basename "$RESUME_CKPT")
    RESUME_FLAG="--resume /app/checkpoints/$CKPT_BASENAME"
    echo "Resume flag: $RESUME_FLAG"
elif $FORCE_PRETRAIN; then
    # Forced pretrain from scratch
    PRETRAIN_FLAG="--pretrain-epochs 5"
    echo "Running fresh pretraining (5 epochs)..."
elif [ -f "$SAVED_PRETRAIN" ]; then
    # Default: use saved pretrain checkpoint
    echo "Using saved pretrain checkpoint: pretrain_complete.pt"
    # Copy to top-level so Docker can see it
    run "cp '$SAVED_PRETRAIN' '$CKPT_DIR/pretrain_resume.pt'"
    RESUME_FLAG="--resume /app/checkpoints/pretrain_resume.pt"
else
    # No saved pretrain, run it
    echo "No saved pretrain found. Running pretraining (5 epochs)..."
    PRETRAIN_FLAG="--pretrain-epochs 5"
fi

# 6. Launch training
if $RESUME_MODE; then
    echo "Resuming training from $RESUME_NAME..."
elif $FORCE_PRETRAIN; then
    echo "Launching pretraining + training from scratch..."
else
    echo "Launching training from pretrain checkpoint..."
fi

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
    --batch-size 144 \
    --lr 1e-4 \
    --num-workers 4 \
    --log-every 50 \
    --save-every 5 \
    --render-every 1 \
    --augment \
    $PRETRAIN_FLAG \
    $RESUME_FLAG \
    --loss-weights '{\"canvas_mse\": 1.0, \"merge\": 1.0, \"stroke_length\": 0.001}'"

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
