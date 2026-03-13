#!/bin/bash
# monitor_training.sh — Live dashboard for stroke model training.
#
# Usage:
#   ./monitor_training.sh          # One-shot status
#   ./monitor_training.sh --watch  # Auto-refresh every 5s
#   ./monitor_training.sh --tail   # Tail the training log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/docker/stroke_model/logs"
CKPT_DIR="$SCRIPT_DIR/docker/stroke_model/checkpoints"
TRAIN_LOG="$LOG_DIR/train.log"
BUILD_LOG="$LOG_DIR/build.log"
STATUS_FILE="$LOG_DIR/status.json"

print_status() {
    echo "═══════════════════════════════════════════════════"
    echo "  STROKE MODEL TRAINING MONITOR"
    echo "═══════════════════════════════════════════════════"
    echo ""

    # Status file
    if [ -f "$STATUS_FILE" ]; then
        stage=$(python3 -c "import json; d=json.load(open('$STATUS_FILE')); print(d.get('stage','?'))" 2>/dev/null || echo "?")
        msg=$(python3 -c "import json; d=json.load(open('$STATUS_FILE')); print(d.get('message','?'))" 2>/dev/null || echo "?")
        updated=$(python3 -c "import json; d=json.load(open('$STATUS_FILE')); print(d.get('updated_at','?'))" 2>/dev/null || echo "?")
        echo "  Stage:   $stage"
        echo "  Status:  $msg"
        echo "  Updated: $updated"
    else
        echo "  No status file found. Training may not have started."
    fi

    echo ""

    # Docker container check
    if docker ps --filter name=stroke-model-train --format '{{.Status}}' 2>/dev/null | grep -q .; then
        container_status=$(docker ps --filter name=stroke-model-train --format '{{.Status}}')
        echo "  Container: RUNNING ($container_status)"
    else
        echo "  Container: not running"
    fi

    # GPU usage
    if command -v nvidia-smi &>/dev/null; then
        gpu_util=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "?,?,?")
        echo "  GPU:       $gpu_util (util%, mem used MB, mem total MB)"
    fi

    echo ""

    # Checkpoints
    if [ -d "$CKPT_DIR" ]; then
        n_ckpt=$(ls "$CKPT_DIR"/*.pt 2>/dev/null | wc -l)
        latest=$(ls -t "$CKPT_DIR"/*.pt 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            latest_name=$(basename "$latest")
            latest_size=$(du -h "$latest" | cut -f1)
            echo "  Checkpoints: $n_ckpt saved (latest: $latest_name, $latest_size)"
        else
            echo "  Checkpoints: none yet"
        fi
    fi

    echo ""

    # Last training lines
    if [ -f "$TRAIN_LOG" ]; then
        lines=$(wc -l < "$TRAIN_LOG")
        echo "  Train log: $lines lines ($TRAIN_LOG)"
        echo "  ─── Last 10 lines ───"
        tail -10 "$TRAIN_LOG" | sed 's/^/  /'
    elif [ -f "$BUILD_LOG" ]; then
        echo "  Build log (training not started yet):"
        echo "  ─── Last 5 lines ───"
        tail -5 "$BUILD_LOG" | sed 's/^/  /'
    fi

    echo ""
    echo "═══════════════════════════════════════════════════"
}

case "${1:-}" in
    --watch)
        while true; do
            clear
            print_status
            sleep 5
        done
        ;;
    --tail)
        if [ -f "$TRAIN_LOG" ]; then
            tail -f "$TRAIN_LOG"
        else
            echo "No train log yet. Watching build log..."
            tail -f "$BUILD_LOG" 2>/dev/null || echo "No logs found."
        fi
        ;;
    *)
        print_status
        ;;
esac
