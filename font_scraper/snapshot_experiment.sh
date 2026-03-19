#!/bin/bash
# snapshot_experiment.sh — Capture the current training run's config, code state,
# tracking images, and loss progression into an experiments/ folder.
#
# Usage:
#   ./snapshot_experiment.sh              # Snapshot current run
#   ./snapshot_experiment.sh --init       # Initialize a new experiment at launch time
#   ./snapshot_experiment.sh --sync       # Copy latest tracking images + losses into current experiment
#   ./snapshot_experiment.sh --notes "increased overlap weight"   # Add notes to current experiment
#
# Designed to be called from restart_training.sh (--init) and manually (--sync/default).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STROKE_DIR="$SCRIPT_DIR/docker/stroke_model"
CKPT_DIR="$STROKE_DIR/checkpoints"
EXPERIMENTS_DIR="$STROKE_DIR/experiments"
CONTAINER_NAME="stroke-model-train"
CURRENT_LINK="$EXPERIMENTS_DIR/current"

mkdir -p "$EXPERIMENTS_DIR"

get_training_logs() {
    if docker ps --filter "name=$CONTAINER_NAME" --format '{{.ID}}' 2>/dev/null | grep -q .; then
        docker logs "$CONTAINER_NAME" 2>&1
    fi
}

# Generate experiment directory name
make_experiment_dir() {
    local timestamp git_hash
    timestamp=$(date '+%Y-%m-%d_%H%M')
    git_hash=$(cd "$SCRIPT_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "nogit")
    echo "$EXPERIMENTS_DIR/${timestamp}_${git_hash}"
}

# --init: Create new experiment folder, snapshot config + code state
init_experiment() {
    local exp_dir notes
    exp_dir=$(make_experiment_dir)
    notes="${1:-}"

    # Avoid duplicate if called twice in same minute
    if [ -d "$exp_dir" ]; then
        exp_dir="${exp_dir}_$(date '+%S')"
    fi

    mkdir -p "$exp_dir"

    # 1. Capture git state
    (cd "$SCRIPT_DIR" && git rev-parse HEAD > "$exp_dir/git_commit.txt" 2>/dev/null || true)
    (cd "$SCRIPT_DIR" && git diff > "$exp_dir/git_diff.patch" 2>/dev/null || true)
    (cd "$SCRIPT_DIR" && git diff --cached >> "$exp_dir/git_diff.patch" 2>/dev/null || true)

    # 2. Capture training config from restart script args
    # Extract the docker run command from restart_training.sh
    local config_json
    config_json=$(python3 -c "
import re, json, sys

# Parse the restart script for training args
with open('$SCRIPT_DIR/restart_training.sh') as f:
    content = f.read()

config = {}

# Extract key training params
for param in ['batch-size', 'lr', 'epochs', 'render-every', 'save-every', 'feature-dim', 'num-workers']:
    m = re.search(r'--' + param + r'\s+(\S+)', content)
    if m:
        key = param.replace('-', '_')
        val = m.group(1).rstrip(' \\\\')
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        config[key] = val

# Extract loss weights
m = re.search(r'--loss-weights\s+.(.+?).\"?\s*$', content, re.MULTILINE)
if m:
    raw = m.group(1).replace('\\\\\"', '\"').replace('\\\"', '\"')
    try:
        config['loss_weights'] = json.loads(raw)
    except json.JSONDecodeError:
        config['loss_weights_raw'] = raw

# Check for augment flag
if '--augment' in content:
    config['augment'] = True

# Check width floor from model.py
try:
    with open('$STROKE_DIR/model.py') as f:
        model_src = f.read()
    m = re.search(r'softplus.*\+\s*([0-9.]+)', model_src)
    if m:
        config['width_floor'] = float(m.group(1))
except Exception:
    pass

json.dump(config, sys.stdout, indent=2)
" 2>/dev/null || echo '{"error": "failed to parse config"}')

    echo "$config_json" > "$exp_dir/config.json"

    # 3. Notes
    if [ -n "$notes" ]; then
        echo "$notes" > "$exp_dir/notes.txt"
    fi

    # 4. Update 'current' symlink
    rm -f "$CURRENT_LINK"
    ln -s "$exp_dir" "$CURRENT_LINK"

    echo "Experiment initialized: $exp_dir"
    echo "$exp_dir"
}

# --sync: Copy tracking images and loss data from running/completed training into current experiment
sync_experiment() {
    if [ ! -L "$CURRENT_LINK" ]; then
        echo "ERROR: No current experiment. Run with --init first."
        exit 1
    fi

    local exp_dir
    exp_dir=$(readlink -f "$CURRENT_LINK")

    if [ ! -d "$exp_dir" ]; then
        echo "ERROR: Current experiment dir missing: $exp_dir"
        exit 1
    fi

    # 1. Copy tracking images
    local tracking_src="$CKPT_DIR/tracking"
    if [ -d "$tracking_src" ]; then
        local count=0
        for epoch_dir in "$tracking_src"/epoch_*; do
            if [ -d "$epoch_dir" ]; then
                local epoch_name
                epoch_name=$(basename "$epoch_dir")
                local dest="$exp_dir/$epoch_name"
                if [ ! -d "$dest" ] || [ "$(ls "$epoch_dir" | wc -l)" -gt "$(ls "$dest" 2>/dev/null | wc -l)" ]; then
                    cp -r "$epoch_dir" "$exp_dir/"
                    count=$((count + 1))
                fi
            fi
        done
        echo "Synced $count epoch(s) of tracking images"
    else
        echo "No tracking images yet"
    fi

    # 2. Capture loss progression
    local logs
    logs=$(get_training_logs)
    if [ -n "$logs" ]; then
        # Epoch summaries
        echo "$logs" | grep "complete.*Loss:" > "$exp_dir/epoch_losses.txt" 2>/dev/null || true

        # Step-level for the last epoch (for debugging)
        echo "$logs" | grep "Epoch.*Step" | tail -50 > "$exp_dir/recent_steps.txt" 2>/dev/null || true

        # Best model saves
        echo "$logs" | grep "Saved checkpoint.*best" > "$exp_dir/best_saves.txt" 2>/dev/null || true

        echo "Synced loss data"
    fi

    echo "Experiment synced: $exp_dir"
}

# --notes: Add/append notes to current experiment
add_notes() {
    if [ ! -L "$CURRENT_LINK" ]; then
        echo "ERROR: No current experiment."
        exit 1
    fi
    local exp_dir
    exp_dir=$(readlink -f "$CURRENT_LINK")
    echo "" >> "$exp_dir/notes.txt"
    echo "[$(date '+%Y-%m-%d %H:%M')] $1" >> "$exp_dir/notes.txt"
    echo "Note added to $exp_dir/notes.txt"
}

# Default: full snapshot (init if needed, then sync)
full_snapshot() {
    if [ ! -L "$CURRENT_LINK" ]; then
        init_experiment ""
    fi
    sync_experiment
}

# List all experiments
list_experiments() {
    echo "═══ Experiments ═══"
    for exp in "$EXPERIMENTS_DIR"/20*; do
        [ -d "$exp" ] || continue
        local name config_summary notes_summary
        name=$(basename "$exp")

        # Read key config values
        config_summary=""
        if [ -f "$exp/config.json" ]; then
            config_summary=$(python3 -c "
import json
with open('$exp/config.json') as f:
    c = json.load(f)
parts = []
if 'width_floor' in c: parts.append(f\"w={c['width_floor']}\")
if 'batch_size' in c: parts.append(f\"bs={c['batch_size']}\")
if 'loss_weights' in c:
    lw = c['loss_weights']
    for k in ['overlap', 'smoothness', 'reversal']:
        if k in lw: parts.append(f\"{k[:4]}={lw[k]}\")
print(' | '.join(parts))
" 2>/dev/null || echo "?")
        fi

        notes_summary=""
        if [ -f "$exp/notes.txt" ]; then
            notes_summary=$(head -1 "$exp/notes.txt")
        fi

        # Count epochs
        local n_epochs
        n_epochs=$(find "$exp" -maxdepth 1 -name "epoch_*" -type d 2>/dev/null | wc -l)

        local is_current=""
        if [ -L "$CURRENT_LINK" ] && [ "$(readlink -f "$CURRENT_LINK")" = "$(readlink -f "$exp")" ]; then
            is_current=" <-- CURRENT"
        fi

        echo "  $name  [$config_summary]  epochs: $n_epochs  $notes_summary$is_current"
    done
}

case "${1:-}" in
    --init)
        init_experiment "${2:-}"
        ;;
    --sync)
        sync_experiment
        ;;
    --notes)
        if [ -z "${2:-}" ]; then
            echo "Usage: $0 --notes \"your note here\""
            exit 1
        fi
        add_notes "$2"
        ;;
    --list)
        list_experiments
        ;;
    *)
        full_snapshot
        ;;
esac
