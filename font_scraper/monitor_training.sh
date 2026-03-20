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
CONTAINER_NAME="stroke-model-train"

get_training_logs() {
    # Try docker logs first (works for running container), fall back to log file
    if docker ps --filter "name=$CONTAINER_NAME" --format '{{.ID}}' 2>/dev/null | grep -q .; then
        docker logs "$CONTAINER_NAME" 2>&1
    elif [ -f "$TRAIN_LOG" ]; then
        cat "$TRAIN_LOG"
    fi
}

parse_progress() {
    # Extract epoch/step progress from latest log line
    local last_line
    last_line=$(get_training_logs | grep "Epoch.*Step" | tail -1)

    if [ -z "$last_line" ]; then
        echo "  Progress: waiting for first step..."
        return
    fi

    # Parse: [Pretrain] Epoch 0 Step 1900/50336 | Loss: 28.17
    local epoch step total_steps loss phase
    epoch=$(echo "$last_line" | grep -oP 'Epoch \K[0-9]+' || echo "")
    step=$(echo "$last_line" | grep -oP 'Step \K[0-9]+' || echo "")
    total_steps=$(echo "$last_line" | grep -oP 'Step [0-9]+/\K[0-9]+' || echo "")
    loss=$(echo "$last_line" | grep -oP 'Loss: \K[0-9.]+' || echo "?")

    # Detect phase (pretrain vs training)
    phase=""
    if echo "$last_line" | grep -q "Pretrain"; then
        phase=" [PRETRAIN]"
    fi

    # Get total epochs from status file or default
    local total_epochs=100
    if [ -f "$STATUS_FILE" ]; then
        total_epochs=$(python3 -c "import json; d=json.load(open('$STATUS_FILE')); print(d.get('epochs', 100))" 2>/dev/null || echo 100)
    fi

    # Find session start epoch (first epoch logged after last "Starting")
    local start_epoch=0
    local start_line
    start_line=$(get_training_logs | grep -E "Starting (training|pretraining)" | tail -1 || true)
    if [ -n "$start_line" ]; then
        local first_step_line
        first_step_line=$(get_training_logs | grep "Epoch.*Step" | head -1 || true)
        if [ -n "$first_step_line" ]; then
            start_epoch=$(echo "$first_step_line" | grep -oP 'Epoch \K[0-9]+' || echo "0")
        fi
    fi

    # Calculate percentages (relative to session)
    local epoch_pct=0 total_pct=0
    if [ -n "$step" ] && [ -n "$total_steps" ] && [ "$total_steps" -gt 0 ]; then
        epoch_pct=$(python3 -c "print(f'{100*$step/$total_steps:.1f}')")
        total_pct=$(python3 -c "
session_done = ($epoch - $start_epoch) * $total_steps + $step
session_total = ($total_epochs - $start_epoch) * $total_steps
print(f'{100*session_done/session_total:.2f}' if session_total > 0 else '0.00')
")
    fi

    # Estimate time remaining (relative to session start, not epoch 0)
    local eta_str=""
    if [ -n "$start_line" ]; then
        local start_ts
        start_ts=$(echo "$start_line" | grep -oP '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
        local last_ts
        last_ts=$(echo "$last_line" | grep -oP '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
        if [ -n "$start_ts" ] && [ -n "$last_ts" ]; then
            eta_str=$(python3 -c "
from datetime import datetime
start = datetime.strptime('$start_ts', '%Y-%m-%d %H:%M:%S')
now = datetime.strptime('$last_ts', '%Y-%m-%d %H:%M:%S')
elapsed = (now - start).total_seconds()
start_epoch = $start_epoch
session_done = ($epoch - start_epoch) * $total_steps + $step
session_total = ($total_epochs - start_epoch) * $total_steps
if session_done > 0:
    remain = elapsed * (session_total - session_done) / session_done
    hours = int(remain // 3600)
    mins = int((remain % 3600) // 60)
    elapsed_h = int(elapsed // 3600)
    elapsed_m = int((elapsed % 3600) // 60)
    print(f'{elapsed_h}h{elapsed_m:02d}m elapsed, ~{hours}h{mins:02d}m remaining')
else:
    print('calculating...')
" 2>/dev/null || echo "")
        fi
    fi

    # Display
    echo "  Epoch:    $epoch / $total_epochs  (step $step / $total_steps)$phase"

    # Progress bar for current epoch
    local bar_width=30
    local filled=$(python3 -c "print(int($epoch_pct / 100 * $bar_width))")
    local empty=$((bar_width - filled))
    local bar=$(printf '█%.0s' $(seq 1 $filled 2>/dev/null) ; printf '░%.0s' $(seq 1 $empty 2>/dev/null))
    echo "  Epoch:    [$bar] ${epoch_pct}%"

    # Progress bar for overall
    filled=$(python3 -c "print(int(float($total_pct) / 100 * $bar_width))")
    empty=$((bar_width - filled))
    bar=$(printf '█%.0s' $(seq 1 $filled 2>/dev/null) ; printf '░%.0s' $(seq 1 $empty 2>/dev/null))
    echo "  Overall:  [$bar] ${total_pct}%"

    echo "  Loss:     $loss"
    if [ -n "$eta_str" ]; then
        echo "  Time:     $eta_str"
    fi
}

print_status() {
    echo "═══════════════════════════════════════════════════════"
    echo "  STROKE MODEL TRAINING MONITOR  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    # Docker container check
    if docker ps --filter "name=$CONTAINER_NAME" --format '{{.Status}}' 2>/dev/null | grep -q .; then
        container_status=$(docker ps --filter "name=$CONTAINER_NAME" --format '{{.Status}}')
        echo "  Container: RUNNING ($container_status)"
    else
        echo "  Container: NOT RUNNING"
        # Check if it exited
        local exit_status
        exit_status=$(docker ps -a --filter "name=$CONTAINER_NAME" --format '{{.Status}}' 2>/dev/null | head -1)
        if [ -n "$exit_status" ]; then
            echo "  Last status: $exit_status"
        fi
    fi

    # GPU usage
    if command -v nvidia-smi &>/dev/null; then
        gpu_util=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "?,?,?,?")
        IFS=',' read -r util mem_used mem_total temp <<< "$gpu_util"
        echo "  GPU:       ${util}% util | ${mem_used}/${mem_total} MB | ${temp}°C"
    fi

    echo ""

    # Training progress
    parse_progress

    echo ""

    # Checkpoints
    if [ -d "$CKPT_DIR" ]; then
        n_ckpt=$(find "$CKPT_DIR" -maxdepth 1 -name "*.pt" 2>/dev/null | wc -l)
        latest=$(find "$CKPT_DIR" -maxdepth 1 -name "*.pt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
        if [ -n "$latest" ]; then
            latest_name=$(basename "$latest")
            latest_size=$(du -h "$latest" | cut -f1)
            echo "  Checkpoints: $n_ckpt saved (latest: $latest_name, $latest_size)"
        else
            echo "  Checkpoints: none yet"
        fi
    fi

    echo ""

    # Loss breakdown from last log line
    local last_line
    last_line=$(get_training_logs | grep "Epoch.*Step" | tail -1)
    if [ -n "$last_line" ]; then
        echo "  ─── Loss Breakdown ───"
        set +eo pipefail
        echo "$last_line" | sed 's/.*Loss: [0-9.]* | //' | tr ' ' '\n' | grep '=' | while IFS='=' read -r key val; do
            printf "    %-14s %s\n" "$key:" "$val"
        done
        set -eo pipefail
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════"
}

case "${1:-}" in
    --watch)
        while true; do
            clear
            print_status
            sleep 1
        done
        ;;
    --tail)
        if docker ps --filter "name=$CONTAINER_NAME" --format '{{.ID}}' 2>/dev/null | grep -q .; then
            docker logs -f "$CONTAINER_NAME" 2>&1
        elif [ -f "$TRAIN_LOG" ]; then
            tail -f "$TRAIN_LOG"
        else
            echo "No training logs found."
        fi
        ;;
    *)
        print_status
        ;;
esac
