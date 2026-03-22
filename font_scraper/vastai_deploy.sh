#!/bin/bash
# vastai_deploy.sh — One-click deployment of stroke model training to Vast.ai
#
# Usage:
#   ./vastai_deploy.sh              # Find GPU, create instance, upload, train
#   ./vastai_deploy.sh --status     # Check training status on remote
#   ./vastai_deploy.sh --sync       # Download latest checkpoints and tracking
#   ./vastai_deploy.sh --stop       # Stop and destroy the instance
#   ./vastai_deploy.sh --resume     # Resume training on existing instance
#   ./vastai_deploy.sh --tail       # Follow remote training log
#
# Prerequisites:
#   pipx install vastai
#   vastai set api-key YOUR_API_KEY
#   SSH key added at https://cloud.vast.ai/account/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STROKE_DIR="$SCRIPT_DIR/docker/stroke_model"
STATE_FILE="$SCRIPT_DIR/.vastai_instance"
REMOTE_DIR="/workspace/stroke_model"

# Training config
EPOCHS=100
BATCH_SIZE=512
LR="4e-4"
RENDER_EVERY=1   # Triton renderer is fast enough for every step
NUM_WORKERS=8
SAVE_EVERY=5

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[vastai]${NC} $*"; }
warn() { echo -e "${YELLOW}[vastai]${NC} $*"; }
err() { echo -e "${RED}[vastai]${NC} $*" >&2; }
info() { echo -e "${CYAN}[vastai]${NC} $*"; }

dump_and_destroy() {
    # Dump all available logs before destroying an instance
    local iid="$1"
    warn "═══ Dumping logs for instance $iid before destroy ═══"

    # Instance status
    warn "── Instance Status ──"
    vastai show instance "$iid" --raw 2>/dev/null | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    for k in ['id','actual_status','intended_status','status_msg','public_ipaddr','ssh_host','ssh_port','ports','cur_state','image_uuid']:
        if k in d:
            print(f'  {k}: {d[k]}')
except: pass
" 2>/dev/null || echo "  (could not read instance status)"

    # Container logs from Vast.ai
    warn "── Container Logs (last 50 lines) ──"
    vastai logs "$iid" 2>/dev/null | tail -50 || echo "  (no logs available)"

    warn "═══ End logs ═══"
    echo ""

    vastai destroy instance "$iid" 2>/dev/null || true
}

check_deps() {
    if ! command -v vastai &>/dev/null; then
        err "vastai CLI not found. Install: pipx install vastai"
        exit 1
    fi
    if ! vastai show user 2>/dev/null | grep -q "id"; then
        err "vastai API key not set. Run: vastai set api-key YOUR_KEY"
        exit 1
    fi
    log "API key verified."
}

# Get SSH connection details from state file
_load_ssh() {
    if [ ! -f "$STATE_FILE" ]; then
        err "No active instance. Run without flags to deploy."
        exit 1
    fi
    INSTANCE_ID=$(head -1 "$STATE_FILE")
    # Read SSH URL from state file (line 2), saved during create_instance
    SSH_URL=$(sed -n '2p' "$STATE_FILE" 2>/dev/null || true)
    if [ -z "$SSH_URL" ]; then
        # Fall back to vastai ssh-url
        SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null || true)
    fi
    if [ -z "$SSH_URL" ]; then
        err "Instance $INSTANCE_ID not reachable."
        exit 1
    fi
    SSH_HOST=$(echo "$SSH_URL" | sed 's|ssh://||' | sed 's|root@||' | cut -d: -f1)
    SSH_PORT=$(echo "$SSH_URL" | sed 's|ssh://||' | cut -d: -f2)
    SSH_CMD="ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST"
    RSYNC_CMD="rsync -avz --progress -e 'ssh -o StrictHostKeyChecking=no -p $SSH_PORT'"
}

create_instance() {
    # Common filters for fast deployment:
    #   inet_down>=500   — fast package installs (apt/pip)
    #   inet_up>=200     — fast checkpoint sync back
    #   disk_bw>=500     — fast dataset I/O
    #   direct_port_count>=1 — SSH works without relay (faster connect)
    #   reliability>=0.98 — fewer flaky hosts
    #   datacenter=true  — dedicated infra, faster boot times
    COMMON_FILTERS='num_gpus=1 disk_space>=80 inet_down>=500 inet_up>=200 disk_bw>=500 direct_port_count>=1 reliability>=0.98 static_ip=true'

    log "Searching for RTX 5090 instances..."
    OFFER_JSON=$(vastai search offers \
        "gpu_name=RTX_5090 $COMMON_FILTERS" \
        -o 'dph+' \
        --raw 2>/dev/null) || true

    OFFER_ID=""
    if [ -n "$OFFER_JSON" ]; then
        OFFER_ID=$(echo "$OFFER_JSON" | python3 -c "
import json, sys
offers = json.load(sys.stdin)
if not offers:
    sys.exit(1)
o = offers[0]
print(o['id'])
" 2>/dev/null) || true
        if [ -n "$OFFER_ID" ]; then
            echo "$OFFER_JSON" | python3 -c "
import json, sys
o = json.load(sys.stdin)[0]
print(f'  Best: \${o[\"dph_total\"]:.3f}/hr, {o.get(\"gpu_name\",\"?\")}, {o.get(\"inet_down\",0):.0f}/{o.get(\"inet_up\",0):.0f} Mbps, disk_bw={o.get(\"disk_bw\",0):.0f} MB/s, reliability={o.get(\"reliability\",0):.2f}')
" 2>/dev/null || true
        fi
    fi

    if [ -z "$OFFER_ID" ]; then
        warn "No RTX 5090 found. Trying any GPU with >=24GB VRAM..."
        OFFER_JSON=$(vastai search offers \
            "gpu_ram>=24 $COMMON_FILTERS" \
            -o 'dph+' \
            --raw 2>/dev/null) || true
        OFFER_ID=$(echo "$OFFER_JSON" | python3 -c "
import json, sys
offers = json.load(sys.stdin)
if not offers: sys.exit(1)
o = offers[0]
print(o['id'])
" 2>/dev/null) || true

        if [ -z "$OFFER_ID" ]; then
            warn "No offers with strict filters. Relaxing network requirements..."
            OFFER_ID=$(vastai search offers \
                'gpu_ram>=24 num_gpus=1 disk_space>=80 inet_down>=200 reliability>=0.95' \
                -o 'dph+' \
                --raw 2>/dev/null | python3 -c "
import json, sys
offers = json.load(sys.stdin)
if not offers: sys.exit(1)
print(offers[0]['id'])
" 2>/dev/null) || { err "No suitable GPU found."; exit 1; }
            warn "Using relaxed filters — deployment may be slower."
        fi
    fi

    log "Creating instance from offer $OFFER_ID..."
    INSTANCE_ID=$(vastai create instance "$OFFER_ID" \
        --image "justinmvail/stroke-model:latest" \
        --disk 80 \
        --ssh --direct \
        --raw 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('new_contract', data.get('id', '')))
")

    if [ -z "$INSTANCE_ID" ]; then
        err "Failed to create instance."
        exit 1
    fi

    echo "$INSTANCE_ID" > "$STATE_FILE"
    log "Instance $INSTANCE_ID created."

    # Wait for instance to be running, tailing logs as they appear
    info "Waiting for instance to start (tailing logs)..."
    LAST_LOG_COUNT=0
    for i in $(seq 1 120); do
        STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | \
            python3 -c "import json,sys; print(json.load(sys.stdin).get('actual_status',''))" 2>/dev/null || true)
        if [ "$STATUS" = "running" ]; then
            echo ""
            log "Instance running."
            break
        fi
        # Stream new log lines every 10 seconds
        if [ $((i % 2)) -eq 0 ]; then
            ALL_LOGS=$(vastai logs "$INSTANCE_ID" 2>/dev/null || true)
            if [ -n "$ALL_LOGS" ]; then
                CURRENT_COUNT=$(echo "$ALL_LOGS" | wc -l)
                if [ "$CURRENT_COUNT" -gt "$LAST_LOG_COUNT" ]; then
                    echo "$ALL_LOGS" | tail -n +$((LAST_LOG_COUNT + 1))
                    LAST_LOG_COUNT=$CURRENT_COUNT
                fi
            fi
        fi
        sleep 5
    done

    if [ "$STATUS" != "running" ]; then
        err "Instance not running after 10 minutes."
        dump_and_destroy "$INSTANCE_ID"
        rm -f "$STATE_FILE"
        CREATE_RETRIES=$((${CREATE_RETRIES:-0} + 1))
        if [ "$CREATE_RETRIES" -ge 3 ]; then
            err "Failed on 3 instances. Giving up."
            exit 1
        fi
        export CREATE_RETRIES
        create_instance
        return
    fi

    # Get direct SSH connection info (bypass relay)
    SSH_HOST=""
    SSH_PORT=""
    for i in $(seq 1 12); do
        INSTANCE_JSON=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null || true)
        if [ -n "$INSTANCE_JSON" ]; then
            SSH_HOST=$(echo "$INSTANCE_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
# Try direct IP + mapped port first
ip = d.get('public_ipaddr', '')
ports = d.get('ports', {})
ssh_port = ''
if '22/tcp' in ports:
    ssh_port = str(ports['22/tcp'][0].get('HostPort', ''))
if ip and ssh_port:
    print(ip)
else:
    # Fall back to ssh-url style
    print(d.get('ssh_host', ''))
" 2>/dev/null || true)
            SSH_PORT=$(echo "$INSTANCE_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
ports = d.get('ports', {})
if '22/tcp' in ports:
    print(str(ports['22/tcp'][0].get('HostPort', '')))
else:
    print(str(d.get('ssh_port', '')))
" 2>/dev/null || true)
        fi
        if [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ]; then
            break
        fi
        sleep 5
    done

    if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
        # Fall back to vastai ssh-url
        warn "Direct connection info not available, trying ssh-url..."
        SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null || true)
        if [ -n "$SSH_URL" ]; then
            SSH_HOST=$(echo "$SSH_URL" | sed 's|ssh://||' | cut -d: -f1)
            SSH_PORT=$(echo "$SSH_URL" | sed 's|ssh://||' | cut -d: -f2)
        fi
    fi

    if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
        err "Could not get SSH connection info."
        exit 1
    fi

    echo "ssh://root@${SSH_HOST}:${SSH_PORT}" >> "$STATE_FILE"
    info "SSH target: $SSH_HOST:$SSH_PORT"

    # Wait for SSH to accept connections
    info "Waiting for SSH connection (5 min timeout)..."
    for i in $(seq 1 60); do
        if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p "$SSH_PORT" "root@$SSH_HOST" "echo ok" 2>/dev/null; then
            echo ""
            log "SSH connected!"
            return 0
        fi
        printf "."
        sleep 5
    done
    echo ""
    CREATE_RETRIES=$((${CREATE_RETRIES:-0} + 1))
    if [ "$CREATE_RETRIES" -ge 3 ]; then
        err "SSH failed on 3 instances. Giving up."
        dump_and_destroy "$INSTANCE_ID"
        rm -f "$STATE_FILE"
        exit 1
    fi
    err "SSH not available after 5 minutes. Destroying instance and retrying ($CREATE_RETRIES/3)..."
    dump_and_destroy "$INSTANCE_ID"
    rm -f "$STATE_FILE"
    export CREATE_RETRIES
    create_instance
}

setup_remote() {
    _load_ssh
    log "Verifying remote environment..."
    $SSH_CMD "python3 -c \"import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')\"" 2>/dev/null
    $SSH_CMD "mkdir -p /workspace/stroke_model/checkpoints /workspace/stroke_model/cache" 2>/dev/null
    log "Environment ready."
}

upload_data() {
    _load_ssh
    log "Uploading training data..."

    # Upload code (including triton_render.py)
    info "Uploading model code..."
    eval $RSYNC_CMD "$STROKE_DIR/"*.py "${SSH_HOST}:${REMOTE_DIR}/"

    # Upload fonts database
    info "Uploading fonts.db (50MB)..."
    eval $RSYNC_CMD "$SCRIPT_DIR/fonts.db" "${SSH_HOST}:${REMOTE_DIR}/"

    # Generate list of fonts actually used in training (6.5K out of 87K)
    info "Generating training font list..."
    python3 -c "
import sqlite3
conn = sqlite3.connect('$SCRIPT_DIR/fonts.db')
rows = conn.execute('''
    SELECT f.file_path FROM fonts f
    LEFT JOIN font_checks fc ON f.id = fc.font_id
    LEFT JOIN font_removals fr ON f.id = fr.font_id
    WHERE fr.font_id IS NULL
    AND (fc.prefilter_passed = 1 OR fc.prefilter_passed IS NULL)
    AND (fc.is_cursive = 0 OR fc.is_cursive IS NULL)
''').fetchall()
conn.close()
for r in rows:
    print(r[0])
" > /tmp/training_fonts.txt
    FONT_COUNT=$(wc -l < /tmp/training_fonts.txt)
    info "Uploading $FONT_COUNT training fonts (~572MB, not all 87K)..."
    eval $RSYNC_CMD --files-from=/tmp/training_fonts.txt "$SCRIPT_DIR/" "${SSH_HOST}:${REMOTE_DIR}/"

    # Always start fresh — never upload local checkpoints

    log "Upload complete."
}

start_training() {
    _load_ssh
    log "Starting training..."

    RESUME_FLAG=""
    if $SSH_CMD "test -f ${REMOTE_DIR}/checkpoints/best_model.pt" 2>/dev/null; then
        RESUME_FLAG="--resume ${REMOTE_DIR}/checkpoints/best_model.pt"
        info "Resuming from best_model.pt"
    fi

    $SSH_CMD -T "cd ${REMOTE_DIR} && nohup python3 -u train.py \
        --db ${REMOTE_DIR}/fonts.db \
        --font-dir ${REMOTE_DIR} \
        --output-dir ${REMOTE_DIR}/checkpoints \
        --cache-dir ${REMOTE_DIR}/cache \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --augment \
        --num-workers $NUM_WORKERS \
        --log-every 50 \
        --save-every $SAVE_EVERY \
        --render-every $RENDER_EVERY \
        --pretrain-epochs 5 \
        --loss-weights '{\"canvas_mse\": 1.0, \"merge\": 1.0, \"stroke_length\": 0.001}' \
        $RESUME_FLAG \
        > ${REMOTE_DIR}/train.log 2>&1 &
    echo \$! > ${REMOTE_DIR}/train.pid
    echo 'Training started (PID: '\$(cat ${REMOTE_DIR}/train.pid)')'
    "
    log "Training launched. Use --tail to follow logs."
}

show_status() {
    _load_ssh
    $SSH_CMD -T "
        echo '═══════════════════════════════════════════════════════'
        echo '  VAST.AI TRAINING STATUS  '\$(date '+%Y-%m-%d %H:%M:%S')
        echo '═══════════════════════════════════════════════════════'
        echo ''
        if [ -f ${REMOTE_DIR}/train.pid ] && kill -0 \$(cat ${REMOTE_DIR}/train.pid) 2>/dev/null; then
            echo '  Training: RUNNING (PID '\$(cat ${REMOTE_DIR}/train.pid)')'
        else
            echo '  Training: STOPPED'
        fi
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read util mem_used mem_total temp; do
            echo \"  GPU:      \${util}% util | \${mem_used}/\${mem_total} MB | \${temp}°C\"
        done
        echo ''
        echo '  ─── Latest Log ───'
        tail -5 ${REMOTE_DIR}/train.log 2>/dev/null || echo '  No log found.'
        echo ''
        echo '  ─── Checkpoints ───'
        ls -lh ${REMOTE_DIR}/checkpoints/*.pt 2>/dev/null | awk '{print \"  \" \$NF \" (\" \$5 \", \" \$6 \" \" \$7 \")\"}' || echo '  None yet.'
        echo ''
        echo '  ─── Tracking Epochs ───'
        ls -d ${REMOTE_DIR}/checkpoints/tracking/epoch_* 2>/dev/null | wc -l | xargs -I{} echo '  {} epochs tracked'
        echo '═══════════════════════════════════════════════════════'
    " 2>/dev/null
}

tail_log() {
    _load_ssh
    log "Following remote training log..."
    log "(Will auto-sync and shut down when training completes)"
    log "(Ctrl+C to detach — instance keeps running)"
    echo ""

    trap 'echo ""; warn "Detached. Instance still running."; warn "Use: $0 --status | --sync | --stop"; exit 0' INT

    while true; do
        # Check if training is still running
        RUNNING=$($SSH_CMD -T "
            if [ -f ${REMOTE_DIR}/train.pid ] && kill -0 \$(cat ${REMOTE_DIR}/train.pid) 2>/dev/null; then
                echo 'yes'
            else
                echo 'no'
            fi
        " 2>/dev/null || echo "unknown")

        if [ "$RUNNING" = "no" ]; then
            echo ""
            FINAL_LINE=$($SSH_CMD -T "tail -1 ${REMOTE_DIR}/train.log 2>/dev/null" || true)
            if echo "$FINAL_LINE" | grep -q "Training complete"; then
                log "Training completed successfully!"
            else
                warn "Training stopped. Last line: $FINAL_LINE"
            fi
            log "Downloading results..."
            sync_results
            log "Destroying instance..."
            INSTANCE_ID=$(head -1 "$STATE_FILE")
            vastai destroy instance "$INSTANCE_ID" 2>/dev/null
            rm -f "$STATE_FILE"
            log "Done! Checkpoints at: $STROKE_DIR/checkpoints/"
            return 0
        fi

        # Show latest log line
        $SSH_CMD -T "tail -1 ${REMOTE_DIR}/train.log 2>/dev/null" 2>/dev/null || true
        sleep 30
    done
}

sync_results() {
    _load_ssh
    log "Syncing checkpoints and tracking images..."

    mkdir -p "$STROKE_DIR/checkpoints/tracking"

    eval $RSYNC_CMD "${SSH_HOST}:${REMOTE_DIR}/checkpoints/*.pt" "$STROKE_DIR/checkpoints/" 2>/dev/null || true
    eval $RSYNC_CMD "${SSH_HOST}:${REMOTE_DIR}/checkpoints/tracking/" "$STROKE_DIR/checkpoints/tracking/" 2>/dev/null || true
    eval $RSYNC_CMD "${SSH_HOST}:${REMOTE_DIR}/train.log" "$STROKE_DIR/logs/train_vastai.log" 2>/dev/null || true

    log "Sync complete."
}

monitor_until_done() {
    _load_ssh
    LAST_EPOCH=""
    SYNC_INTERVAL=300  # sync every 5 minutes
    LAST_SYNC=$(date +%s)

    trap 'echo ""; warn "Detached. Instance still running."; warn "Use: $0 --status | --sync | --stop"; exit 0' INT

    while true; do
        # Check if training is still running
        RUNNING=$($SSH_CMD -T "
            if [ -f ${REMOTE_DIR}/train.pid ] && kill -0 \$(cat ${REMOTE_DIR}/train.pid) 2>/dev/null; then
                echo 'yes'
            else
                echo 'no'
            fi
        " 2>/dev/null || echo "unknown")

        if [ "$RUNNING" = "no" ]; then
            echo ""
            log "Training finished!"
            # Check if it completed or crashed
            FINAL_LINE=$($SSH_CMD -T "tail -1 ${REMOTE_DIR}/train.log 2>/dev/null" || true)
            if echo "$FINAL_LINE" | grep -q "Training complete"; then
                log "Training completed successfully."
            else
                warn "Training may have crashed. Last log line:"
                warn "  $FINAL_LINE"
            fi
            echo ""
            log "Downloading final results..."
            sync_results
            echo ""
            log "Destroying instance..."
            INSTANCE_ID=$(head -1 "$STATE_FILE")
            vastai destroy instance "$INSTANCE_ID" 2>/dev/null
            rm -f "$STATE_FILE"
            log "Done! Checkpoints saved to: $STROKE_DIR/checkpoints/"
            return 0
        fi

        # Show latest log line
        LATEST=$($SSH_CMD -T "tail -1 ${REMOTE_DIR}/train.log 2>/dev/null" 2>/dev/null || true)
        if [ -n "$LATEST" ]; then
            # Extract epoch info
            EPOCH=$(echo "$LATEST" | grep -oP 'Epoch \K[0-9]+' || true)
            STEP=$(echo "$LATEST" | grep -oP 'Step \K[0-9]+/[0-9]+' || true)
            LOSS=$(echo "$LATEST" | grep -oP 'Loss: \K[0-9.]+' || true)

            if [ -n "$EPOCH" ]; then
                printf "\r  Epoch %s Step %s | Loss: %s    " "$EPOCH" "$STEP" "$LOSS"

                # Sync when epoch changes
                if [ "$EPOCH" != "$LAST_EPOCH" ] && [ -n "$LAST_EPOCH" ]; then
                    echo ""
                    info "Epoch $EPOCH started. Syncing checkpoints..."
                    sync_results 2>/dev/null
                    info "Sync done."
                fi
                LAST_EPOCH="$EPOCH"
            fi
        fi

        # Periodic sync
        NOW=$(date +%s)
        if [ $((NOW - LAST_SYNC)) -ge $SYNC_INTERVAL ]; then
            echo ""
            info "Periodic sync..."
            sync_results 2>/dev/null
            LAST_SYNC=$NOW
        fi

        sleep 10
    done
}

stop_instance() {
    if [ ! -f "$STATE_FILE" ]; then
        err "No active instance."
        exit 1
    fi
    INSTANCE_ID=$(head -1 "$STATE_FILE")

    warn "Syncing results before stopping..."
    sync_results || true

    warn "Destroying instance $INSTANCE_ID..."
    vastai destroy instance "$INSTANCE_ID"
    rm -f "$STATE_FILE"
    log "Instance destroyed."
}

# Main
case "${1:-}" in
    --status)  check_deps; show_status ;;
    --sync)    check_deps; sync_results ;;
    --stop)    check_deps; stop_instance ;;
    --resume)  check_deps; start_training ;;
    --tail)    check_deps; tail_log ;;
    --help)
        echo "Usage: $0 [--status|--sync|--stop|--resume|--tail|--help]"
        echo ""
        echo "  (no args)  Full deploy: find GPU, create instance, upload, train"
        echo "  --status   Check training status on remote instance"
        echo "  --tail     Follow remote training log live"
        echo "  --sync     Download checkpoints and tracking images"
        echo "  --stop     Sync results, then destroy the instance"
        echo "  --resume   Restart training on existing instance"
        ;;
    *)
        check_deps
        log "Starting full deployment..."
        echo ""
        # Reuse existing instance if one is running
        if [ -f "$STATE_FILE" ]; then
            INSTANCE_ID=$(head -1 "$STATE_FILE")
            STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('actual_status',''))" 2>/dev/null || true)
            if [ "$STATUS" = "running" ]; then
                log "Reusing existing instance $INSTANCE_ID (running)"
            else
                log "Instance $INSTANCE_ID is $STATUS. Creating new one..."
                rm -f "$STATE_FILE"
                create_instance
            fi
        else
            create_instance
        fi
        echo ""
        setup_remote
        echo ""
        upload_data
        echo ""
        start_training
        echo ""
        xdg-open "https://cloud.vast.ai/instances/" 2>/dev/null &
        echo ""
        tail_log
        ;;
esac
