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

    # Parse: Epoch 0 Step 1900/50336 | Loss: 28.17
    local epoch step total_steps loss
    epoch=$(echo "$last_line" | grep -oP 'Epoch \K[0-9]+')
    step=$(echo "$last_line" | grep -oP 'Step \K[0-9]+')
    total_steps=$(echo "$last_line" | grep -oP 'Step [0-9]+/\K[0-9]+')
    loss=$(echo "$last_line" | grep -oP 'Loss: \K[0-9.]+')

    # Get total epochs from status file or default
    local total_epochs=100
    if [ -f "$STATUS_FILE" ]; then
        total_epochs=$(python3 -c "import json; d=json.load(open('$STATUS_FILE')); print(d.get('epochs', 100))" 2>/dev/null || echo 100)
    fi

    # Calculate percentages
    local epoch_pct=0 total_pct=0
    if [ -n "$step" ] && [ -n "$total_steps" ] && [ "$total_steps" -gt 0 ]; then
        epoch_pct=$(python3 -c "print(f'{100*$step/$total_steps:.1f}')")
        total_pct=$(python3 -c "print(f'{100*($epoch*$total_steps+$step)/($total_epochs*$total_steps):.2f}')")
    fi

    # Estimate time remaining
    local start_line eta_str=""
    start_line=$(get_training_logs | grep "Starting training" | tail -1)
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
pct = ($epoch * $total_steps + $step) / ($total_epochs * $total_steps)
if pct > 0:
    total_est = elapsed / pct
    remain = total_est - elapsed
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
    echo "  Epoch:    $epoch / $total_epochs  (step $step / $total_steps)"

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
    echo "  STROKE MODEL TRAINING MONITOR"
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

    # Loss breakdown from last log line
    local last_line
    last_line=$(get_training_logs | grep "Epoch.*Step" | tail -1)
    if [ -n "$last_line" ]; then
        echo "  ─── Loss Breakdown ───"
        echo "$last_line" | grep -oP '(coverage|outside|overlap|smoothness|existence)=[0-9.]+' | while read -r component; do
            printf "    %-14s %s\n" "$(echo "$component" | cut -d= -f1):" "$(echo "$component" | cut -d= -f2)"
        done
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════"
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
