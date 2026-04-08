#!/bin/bash
# vastai_deploy.sh — One-click deployment of stroke model training to Vast.ai
#
# Usage:
#   ./vastai_deploy.sh              # Find GPU, create instance, start training
#   ./vastai_deploy.sh --status     # Check training status on remote
#   ./vastai_deploy.sh --sync       # Download latest checkpoints and tracking
#   ./vastai_deploy.sh --stop       # Stop and destroy the instance
#   ./vastai_deploy.sh --resume     # Resume training on existing instance
#   ./vastai_deploy.sh --tail       # Follow remote training log
#
# The Docker image (built by docker/stroke_model/build_vastai.sh) contains
# everything: code, fonts, and fonts.db. No upload step needed.
#
# Prerequisites:
#   pipx install vastai
#   vastai set api-key YOUR_API_KEY
#   SSH key added at https://cloud.vast.ai/account/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STROKE_DIR="$SCRIPT_DIR/docker/stroke_model"
STATE_FILE="$STROKE_DIR/.vastai_instance"
TAG_FILE="$STROKE_DIR/.last_build_tag"
BLACKLIST_FILE="$STROKE_DIR/.host_blacklist"
REMOTE_DIR="/workspace"  # Outputs go here (not baked into image)
DATA_DIR="/data"         # Fonts + db downloaded here at boot
GDRIVE_FILE_ID="1MWPIwbt5aFwSGpnX0VaH-1KgCUZ5R8o0"  # training_data.tar.gz on Google Drive

# Training config
EPOCHS=100
BATCH_SIZE=512
LR="4e-4"
RENDER_EVERY=2   # render_every=1 causes collapse without overlap annealing
NUM_WORKERS=8
SAVE_EVERY=5
FLAT_CAPS=false   # round caps (flat caps destabilize via merge penalty interaction)
SERIFS=false      # serif head disabled until stroke quality stabilizes
LOSS_WEIGHTS='{"canvas_mse": 1.0, "merge": 2.0, "sinuosity": 0.01, "smoothness": 0.001, "width_smooth": 0.01, "hires_mse": 1.0, "overlap": 0.3, "parallel": 1.0, "exist_reward": 0.3, "exist_decay": 0.05}'

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

_blacklist_host() {
    local machine_id="$1"
    local reason="$2"
    if [ -n "$machine_id" ] && ! grep -q "^${machine_id}$" "$BLACKLIST_FILE" 2>/dev/null; then
        echo "$machine_id" >> "$BLACKLIST_FILE"
        warn "Blacklisted machine $machine_id ($reason)"
    fi
}

_filter_blacklisted() {
    # Reads JSON offer list from stdin, removes blacklisted machine_ids, outputs filtered JSON
    if [ ! -f "$BLACKLIST_FILE" ]; then
        cat  # no blacklist, pass through
        return
    fi
    python3 -c "
import json, sys
bl = set(open('$BLACKLIST_FILE').read().split())
offers = json.load(sys.stdin)
filtered = [o for o in offers if str(o.get('machine_id','')) not in bl]
json.dump(filtered, sys.stdout)
" 2>/dev/null
}

dump_and_destroy() {
    local iid="$1"
    local reason="${2:-unknown}"
    # Blacklist the machine if we have its ID
    local mid=$(sed -n '3p' "$STATE_FILE" 2>/dev/null || true)
    if [ -n "$mid" ]; then
        _blacklist_host "$mid" "$reason"
    fi
    warn "═══ Dumping logs for instance $iid before destroy ═══"

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

    warn "── Container Logs (last 50 lines) ──"
    vastai logs "$iid" 2>/dev/null | tail -50 || echo "  (no logs available)"

    warn "═══ End logs ═══"
    echo ""

    vastai destroy instance "$iid" 2>/dev/null || true

    # Verify instance is actually dead before returning — Vast.ai destroy is async
    for _try in $(seq 1 12); do
        sleep 5
        _status=$(vastai show instance "$iid" --raw 2>/dev/null | \
            python3 -c "import json,sys; print(json.load(sys.stdin).get('actual_status',''))" 2>/dev/null || echo "gone")
        _status=${_status:-gone}
        if [ "$_status" = "gone" ] || [ "$_status" = "exited" ] || [ "$_status" = "" ]; then
            log "Instance $iid confirmed destroyed."
            return
        fi
        # Retry destroy if still alive
        vastai destroy instance "$iid" 2>/dev/null || true
        printf "."
    done
    warn "Instance $iid may still be lingering after 60s — check manually."
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

_load_ssh() {
    if [ ! -f "$STATE_FILE" ]; then
        err "No active instance. Run without flags to deploy."
        exit 1
    fi
    INSTANCE_ID=$(head -1 "$STATE_FILE")
    SSH_URL=$(sed -n '2p' "$STATE_FILE" 2>/dev/null || true)
    if [ -z "$SSH_URL" ]; then
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

_get_image_tag() {
    if [ ! -f "$TAG_FILE" ]; then
        err "No image tag found. Run build_vastai.sh first:"
        err "  $STROKE_DIR/build_vastai.sh"
        exit 1
    fi
    IMAGE_TAG=$(cat "$TAG_FILE")
    log "Using image: $IMAGE_TAG"
}

_build_onstart_cmd() {
    # Downloads training data from Google Drive, then starts training
    # Outputs go to /workspace/ which is not baked into the image
    cat <<ONSTART
echo "=== Fixing SSH permissions ===" && \
sed -i 's/^#*StrictModes.*/StrictModes no/' /etc/ssh/sshd_config 2>/dev/null; \
echo "StrictModes no" >> /etc/ssh/sshd_config 2>/dev/null; \
chmod 600 /root/.ssh/authorized_keys 2>/dev/null; \
echo "=== Downloading training data from Google Drive ===" && \
mkdir -p ${DATA_DIR} ${REMOTE_DIR}/checkpoints ${REMOTE_DIR}/cache && \
if [ ! -f ${DATA_DIR}/fonts.db ]; then
    gdown ${GDRIVE_FILE_ID} -O /tmp/training_data.tar.gz && \
    tar xzf /tmp/training_data.tar.gz -C ${DATA_DIR} && \
    rm /tmp/training_data.tar.gz && \
    echo "=== Data downloaded: \$(find ${DATA_DIR}/fonts -type f | wc -l) fonts ==="
else
    echo "=== Data already present, skipping download ==="
fi && \
cd /app && \
RESUME_FLAG="" && \
if [ -f ${REMOTE_DIR}/checkpoints/best_model.pt ]; then RESUME_FLAG="--resume ${REMOTE_DIR}/checkpoints/best_model.pt"; fi && \
EXTRA_FLAGS="" && \
if [ "$FLAT_CAPS" = "true" ]; then EXTRA_FLAGS="\$EXTRA_FLAGS --flat-caps"; fi && \
if [ "$SERIFS" = "true" ]; then EXTRA_FLAGS="\$EXTRA_FLAGS --serifs"; fi && \
nohup python3 -u train.py \
    --db /data/fonts.db \
    --font-dir /data \
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
    --loss-weights '$(echo "$LOSS_WEIGHTS" | sed "s/'/\\\\'/g")' \
    \$RESUME_FLAG \$EXTRA_FLAGS \
    > ${REMOTE_DIR}/train.log 2>&1 &
echo \$! > ${REMOTE_DIR}/train.pid
ONSTART
}

create_instance() {
    _get_image_tag

    COMMON_FILTERS='num_gpus=1 disk_space>=100 inet_down>=500 inet_up>=200 disk_bw>=500 direct_port_count>=1 reliability>=0.98 static_ip=true'

    # Show blacklist if any
    if [ -f "$BLACKLIST_FILE" ] && [ -s "$BLACKLIST_FILE" ]; then
        info "Blacklisted machines: $(tr '\n' ' ' < "$BLACKLIST_FILE")"
    fi

    log "Searching for RTX 5090 instances..."
    OFFER_JSON=$(vastai search offers \
        "gpu_name=RTX_5090 $COMMON_FILTERS" \
        -o 'dph+' \
        --raw 2>/dev/null | _filter_blacklisted) || true

    OFFER_ID=""
    MACHINE_ID=""
    if [ -n "$OFFER_JSON" ]; then
        read -r OFFER_ID MACHINE_ID < <(echo "$OFFER_JSON" | python3 -c "
import json, sys
offers = json.load(sys.stdin)
if not offers: sys.exit(1)
o = offers[0]
print(o['id'], o.get('machine_id',''))
" 2>/dev/null) || true
        if [ -n "$OFFER_ID" ]; then
            echo "$OFFER_JSON" | python3 -c "
import json, sys
o = json.load(sys.stdin)[0]
print(f'  Best: \${o[\"dph_total\"]:.3f}/hr, {o.get(\"gpu_name\",\"?\")}, {o.get(\"inet_down\",0):.0f}/{o.get(\"inet_up\",0):.0f} Mbps, disk_bw={o.get(\"disk_bw\",0):.0f} MB/s, reliability={o.get(\"reliability\",0):.2f}, machine={o.get(\"machine_id\",\"?\")}')
" 2>/dev/null || true
        fi
    fi

    if [ -z "$OFFER_ID" ]; then
        warn "No RTX 5090 found. Trying any GPU with >=24GB VRAM..."
        OFFER_JSON=$(vastai search offers \
            "gpu_ram>=24 $COMMON_FILTERS" \
            -o 'dph+' \
            --raw 2>/dev/null | _filter_blacklisted) || true
        read -r OFFER_ID MACHINE_ID < <(echo "$OFFER_JSON" | python3 -c "
import json, sys
offers = json.load(sys.stdin)
if not offers: sys.exit(1)
o = offers[0]
print(o['id'], o.get('machine_id',''))
" 2>/dev/null) || true

        if [ -z "$OFFER_ID" ]; then
            warn "No offers with strict filters. Relaxing network requirements..."
            read -r OFFER_ID MACHINE_ID < <(vastai search offers \
                'gpu_ram>=24 num_gpus=1 disk_space>=100 inet_down>=200 reliability>=0.95' \
                -o 'dph+' \
                --raw 2>/dev/null | _filter_blacklisted | python3 -c "
import json, sys
offers = json.load(sys.stdin)
if not offers: sys.exit(1)
o = offers[0]
print(o['id'], o.get('machine_id',''))
" 2>/dev/null) || { err "No suitable GPU found."; exit 1; }
            warn "Using relaxed filters — deployment may be slower."
        fi
    fi

    ONSTART_CMD=$(_build_onstart_cmd)

    log "Creating instance from offer $OFFER_ID..."
    INSTANCE_ID=$(vastai create instance "$OFFER_ID" \
        --image "$IMAGE_TAG" \
        --disk 100 \
        --ssh --direct \
        --onstart-cmd "$ONSTART_CMD" \
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
    echo "" >> "$STATE_FILE"  # placeholder for SSH URL (line 2)
    echo "$MACHINE_ID" >> "$STATE_FILE"  # machine_id (line 3)
    log "Instance $INSTANCE_ID created (machine $MACHINE_ID)."

    # Wait for instance to be running
    info "Waiting for instance to start (20 min timeout, tailing logs)..."
    LAST_LOG_COUNT=0
    for i in $(seq 1 240); do
        INSTANCE_RAW=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null || true)
        STATUS=$(echo "$INSTANCE_RAW" | python3 -c "import json,sys; print(json.load(sys.stdin).get('actual_status',''))" 2>/dev/null || true)
        STATUS_MSG=$(echo "$INSTANCE_RAW" | python3 -c "import json,sys; print(json.load(sys.stdin).get('status_msg',''))" 2>/dev/null || true)

        if [ "$STATUS" = "running" ]; then
            echo ""
            log "Instance running."
            break
        fi

        if echo "$STATUS_MSG" | grep -qi "deadline exceeded\|unauthorized\|manifest unknown\|denied\|error"; then
            echo ""
            err "Docker pull failed: $STATUS_MSG"
            dump_and_destroy "$INSTANCE_ID" "docker pull failed"
            rm -f "$STATE_FILE"
            CREATE_RETRIES=$((${CREATE_RETRIES:-0} + 1))
            if [ "$CREATE_RETRIES" -ge 5 ]; then
                err "Failed on 5 instances. Giving up."
                exit 1
            fi
            warn "Retrying on a different host ($CREATE_RETRIES/5)..."
            export CREATE_RETRIES
            create_instance
            return
        fi

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
        if [ "$STATUS" = "loading" ]; then
            warn "Still loading after 20 min (image pull in progress). Max 15 more min..."
            LOAD_POLLS=0
            LOAD_MAX=90  # 90 × 10s = 15 min
            while [ "$LOAD_POLLS" -lt "$LOAD_MAX" ]; do
                STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | \
                    python3 -c "import json,sys; print(json.load(sys.stdin).get('actual_status',''))" 2>/dev/null || true)
                if [ "$STATUS" = "running" ]; then
                    log "Instance running!"
                    break
                elif [ "$STATUS" != "loading" ]; then
                    err "Instance status changed to: $STATUS"
                    dump_and_destroy "$INSTANCE_ID" "status changed to $STATUS during pull"
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
                printf "."
                LOAD_POLLS=$((LOAD_POLLS + 1))
                sleep 10
            done
            if [ "$STATUS" != "running" ]; then
                err "Image pull stuck after 35 min total. Host is broken."
                dump_and_destroy "$INSTANCE_ID" "image pull timeout"
                rm -f "$STATE_FILE"
                CREATE_RETRIES=$((${CREATE_RETRIES:-0} + 1))
                if [ "$CREATE_RETRIES" -ge 5 ]; then
                    err "Failed on 5 instances. Giving up."
                    exit 1
                fi
                warn "Retrying on a different host ($CREATE_RETRIES/5)..."
                export CREATE_RETRIES
                create_instance
                return
            fi
        else
            err "Instance status: $STATUS (not loading or running)."
            dump_and_destroy "$INSTANCE_ID" "bad status: $STATUS"
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
    fi

    # Get direct SSH connection info
    SSH_HOST=""
    SSH_PORT=""
    for i in $(seq 1 12); do
        INSTANCE_JSON=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null || true)
        if [ -n "$INSTANCE_JSON" ]; then
            SSH_HOST=$(echo "$INSTANCE_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
ip = d.get('public_ipaddr', '')
ports = d.get('ports', {})
ssh_port = ''
if '22/tcp' in ports:
    ssh_port = str(ports['22/tcp'][0].get('HostPort', ''))
if ip and ssh_port:
    print(ip)
else:
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

    sed -i "2s|.*|ssh://root@${SSH_HOST}:${SSH_PORT}|" "$STATE_FILE"
    info "SSH target: $SSH_HOST:$SSH_PORT"

    # Wait for SSH — try direct port first, fall back to proxy
    info "Waiting for SSH connection (5 min timeout)..."

    # Also extract proxy SSH info as fallback
    PROXY_HOST=$(echo "$INSTANCE_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ssh_host',''))" 2>/dev/null || true)
    PROXY_PORT=$(echo "$INSTANCE_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ssh_port',''))" 2>/dev/null || true)

    SSH_CONNECTED=false
    for i in $(seq 1 60); do
        if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p "$SSH_PORT" "root@$SSH_HOST" "echo ok" 2>/dev/null; then
            echo ""
            log "SSH connected (direct)!"
            SSH_CONNECTED=true
            break
        fi
        # Try proxy fallback every 10 attempts
        if [ $((i % 10)) -eq 0 ] && [ -n "$PROXY_HOST" ] && [ -n "$PROXY_PORT" ]; then
            if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p "$PROXY_PORT" "root@$PROXY_HOST" "echo ok" 2>/dev/null; then
                echo ""
                log "SSH connected (proxy: $PROXY_HOST:$PROXY_PORT)!"
                SSH_HOST="$PROXY_HOST"
                SSH_PORT="$PROXY_PORT"
                # Update state file with working SSH info
                sed -i "2s|.*|ssh://root@${SSH_HOST}:${SSH_PORT}|" "$STATE_FILE"
                SSH_CONNECTED=true
                break
            fi
        fi
        printf "."
        sleep 5
    done

    if ! $SSH_CONNECTED; then
        CREATE_RETRIES=$((${CREATE_RETRIES:-0} + 1))
        if [ "$CREATE_RETRIES" -ge 3 ]; then
            err "SSH failed on 3 instances. Giving up."
            dump_and_destroy "$INSTANCE_ID" "SSH unreachable"
            rm -f "$STATE_FILE"
            exit 1
        fi
        err "SSH not available after 5 minutes. Retrying ($CREATE_RETRIES/3)..."
        dump_and_destroy "$INSTANCE_ID" "SSH unreachable"
        rm -f "$STATE_FILE"
        export CREATE_RETRIES
        create_instance
        return
    fi
}

verify_training() {
    _load_ssh
    info "Verifying training started (waiting up to 3 min)..."

    # Poll every 15s for up to 3 minutes — onstart needs time to download data + start
    for i in $(seq 1 12); do
        sleep 15
        RESULT=$($SSH_CMD -T "
            # Check 1: is a python3 train.py process running?
            if pgrep -f 'python3.*train.py' >/dev/null 2>&1; then
                echo 'running'
            # Check 2: does train.log exist and have recent content?
            elif [ -f ${REMOTE_DIR}/train.log ] && [ \$(wc -l < ${REMOTE_DIR}/train.log) -gt 2 ]; then
                echo 'log_exists'
            # Check 3: is data still downloading?
            elif pgrep -f 'gdown' >/dev/null 2>&1; then
                echo 'downloading'
            else
                echo 'not_started'
            fi
        " 2>/dev/null || echo "ssh_error")

        case "$RESULT" in
            running)
                log "Training verified running!"
                $SSH_CMD -T "tail -1 ${REMOTE_DIR}/train.log 2>/dev/null" 2>/dev/null || true
                return 0
                ;;
            log_exists)
                log "Training log found, process may have just started."
                $SSH_CMD -T "tail -3 ${REMOTE_DIR}/train.log 2>/dev/null" 2>/dev/null || true
                return 0
                ;;
            downloading)
                info "  Still downloading training data..."
                ;;
            not_started)
                info "  Waiting for onstart to complete..."
                ;;
            ssh_error)
                warn "  SSH check failed, retrying..."
                ;;
        esac
    done

    # Final check — don't destroy if there's any sign of life
    FINAL=$($SSH_CMD -T "
        if [ -f ${REMOTE_DIR}/train.log ]; then
            echo 'has_log'
            tail -5 ${REMOTE_DIR}/train.log
        else
            echo 'no_log'
        fi
    " 2>/dev/null || echo "ssh_failed")

    if echo "$FINAL" | grep -q "has_log"; then
        warn "Training process not detected but log exists. NOT destroying — check manually:"
        warn "  $0 --status"
        return 0
    fi

    err "Training failed to start after 3 minutes. No log file found."
    err "Destroying instance..."
    INSTANCE_ID=$(head -1 "$STATE_FILE")
    dump_and_destroy "$INSTANCE_ID" "training failed to start"
    rm -f "$STATE_FILE"
    exit 1
}

start_training() {
    # For --resume: manually start training on an existing instance
    _load_ssh
    log "Starting training..."

    ONSTART_CMD=$(_build_onstart_cmd)
    $SSH_CMD -T "$ONSTART_CMD" 2>/dev/null
    sleep 5
    $SSH_CMD -T "
        if [ -f ${REMOTE_DIR}/train.pid ] && kill -0 \$(cat ${REMOTE_DIR}/train.pid) 2>/dev/null; then
            echo 'Training started (PID '\$(cat ${REMOTE_DIR}/train.pid)')'
        else
            echo 'WARNING: Training may not have started. Check --status'
        fi
    " 2>/dev/null
    log "Training launched. Use --tail to follow logs."
}

show_status() {
    _load_ssh
    $SSH_CMD -T "
        echo '═══════════════════════════════════════════════════════'
        echo '  VAST.AI TRAINING STATUS  '\$(date '+%Y-%m-%d %H:%M:%S')
        echo '═══════════════════════════════════════════════════════'
        echo ''
        TRAIN_PID=\$(pgrep -f 'python3.*train.py' 2>/dev/null || true)
        if [ -n \"\$TRAIN_PID\" ]; then
            echo \"  Training: RUNNING (PID \$TRAIN_PID)\"
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
    log "(Ctrl+C to detach — background monitor keeps running)"
    echo ""

    # Start background monitor that survives terminal disconnects
    _start_bg_monitor

    trap 'echo ""; warn "Detached. Background monitor still watching."; warn "Use: $0 --status | --sync | --stop"; exit 0' INT

    while true; do
        RUNNING=$($SSH_CMD -T "
            if pgrep -f 'python3.*train.py' >/dev/null 2>&1; then
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
            _cleanup_on_complete
            return 0
        fi

        LAST_LINE=$($SSH_CMD -T "tail -1 ${REMOTE_DIR}/train.log 2>/dev/null" 2>/dev/null || true)
        echo "$LAST_LINE"
        # Check for disk full error
        if echo "$LAST_LINE" | grep -qi "No space left on device"; then
            echo ""
            err "DISK FULL detected! Syncing what we have and destroying instance..."
            _cleanup_on_complete
            return 0
        fi
        sleep 30
    done
}

_cleanup_on_complete() {
    # Sync results and destroy instance
    log "Downloading results..."
    sync_results
    log "Destroying instance..."
    INSTANCE_ID=$(head -1 "$STATE_FILE")
    vastai destroy instance "$INSTANCE_ID" 2>/dev/null
    rm -f "$STATE_FILE"
    _kill_bg_monitor
    log "Done! Results at: $STROKE_DIR/history/latest/"
}

_start_bg_monitor() {
    # Background process that polls every 5 min and auto-syncs/destroys on completion.
    # Survives terminal disconnects. Writes PID to file for cleanup.
    local MONITOR_PID_FILE="$STROKE_DIR/.monitor.pid"
    local SCRIPT_PATH="$(readlink -f "$0")"

    # Kill any existing monitor
    _kill_bg_monitor

    (
        while true; do
            sleep 300  # Check every 5 minutes

            # Reload SSH config in case state file changed
            if [ ! -f "$STATE_FILE" ]; then
                exit 0  # Instance already destroyed
            fi

            INSTANCE_ID=$(head -1 "$STATE_FILE")
            STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get('actual_status', 'unknown'))
except: print('unknown')
" 2>/dev/null || echo "unknown")

            # If instance is gone, exit
            if [ "$STATUS" = "unknown" ] || [ "$STATUS" = "exited" ]; then
                echo "[monitor] Instance gone. Running final sync..."
                "$SCRIPT_PATH" --sync 2>/dev/null || true
                rm -f "$STATE_FILE"
                rm -f "$MONITOR_PID_FILE"
                exit 0
            fi

            # Check if training finished or disk full via Vast.ai API (no SSH needed)
            LOG_TAIL=$(vastai logs "$INSTANCE_ID" --tail 5 2>/dev/null || true)
            if echo "$LOG_TAIL" | grep -q "Training complete"; then
                echo "[monitor] Training complete! Syncing and destroying..."
                "$SCRIPT_PATH" --sync 2>/dev/null || true
                vastai destroy instance "$INSTANCE_ID" 2>/dev/null
                rm -f "$STATE_FILE"
                rm -f "$MONITOR_PID_FILE"
                exit 0
            fi
            if echo "$LOG_TAIL" | grep -qi "No space left on device"; then
                echo "[monitor] DISK FULL! Syncing what we have and destroying..."
                "$SCRIPT_PATH" --sync 2>/dev/null || true
                vastai destroy instance "$INSTANCE_ID" 2>/dev/null
                rm -f "$STATE_FILE"
                rm -f "$MONITOR_PID_FILE"
                exit 0
            fi
        done
    ) &
    echo $! > "$MONITOR_PID_FILE"
    log "Background monitor started (PID $!, checks every 5 min)"
}

_kill_bg_monitor() {
    local MONITOR_PID_FILE="$STROKE_DIR/.monitor.pid"
    if [ -f "$MONITOR_PID_FILE" ]; then
        local PID=$(cat "$MONITOR_PID_FILE")
        kill "$PID" 2>/dev/null || true
        rm -f "$MONITOR_PID_FILE"
    fi
}

sync_results() {
    _load_ssh
    INSTANCE_ID=$(head -1 "$STATE_FILE")

    # Each run syncs to a unique folder: history/<date>_<instance>_<commit>/
    COMMIT=$(git -C "$SCRIPT_DIR/.." log --oneline -1 2>/dev/null | cut -c1-7 || echo "unknown")
    RUN_DIR="$STROKE_DIR/history/$(date +%Y%m%d)_i${INSTANCE_ID}_${COMMIT}"
    mkdir -p "$RUN_DIR/tracking"

    # Update "latest" symlink
    ln -sfn "$RUN_DIR" "$STROKE_DIR/history/latest"

    log "Syncing to $RUN_DIR ..."

    eval $RSYNC_CMD "root@${SSH_HOST}:${REMOTE_DIR}/checkpoints/tracking/" "$RUN_DIR/tracking/" 2>/dev/null || true
    eval $RSYNC_CMD "root@${SSH_HOST}:${REMOTE_DIR}/train.log" "$RUN_DIR/train.log" 2>/dev/null || true
    eval $RSYNC_CMD "root@${SSH_HOST}:${REMOTE_DIR}/checkpoints/*.pt" "$RUN_DIR/" 2>/dev/null || true

    log "Sync complete. Results at: $RUN_DIR/"
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
    _kill_bg_monitor
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
        echo "  (no args)  Full deploy: find GPU, create instance, start training"
        echo "  --status   Check training status on remote instance"
        echo "  --tail     Follow remote training log live"
        echo "  --sync     Download checkpoints and tracking images"
        echo "  --stop     Sync results, then destroy the instance"
        echo "  --resume   Restart training on existing instance"
        echo ""
        echo "Before deploying, build the image:"
        echo "  $STROKE_DIR/build_vastai.sh"
        ;;
    *)
        check_deps
        log "Starting full deployment..."
        echo ""
        # Reuse existing instance if one is running, clean up stale state
        if [ -f "$STATE_FILE" ]; then
            INSTANCE_ID=$(head -1 "$STATE_FILE")
            STATUS=$(timeout 10 vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('actual_status',''))" 2>/dev/null || echo "dead")
            STATUS=${STATUS:-dead}  # empty = dead
            if [ "$STATUS" = "running" ] || [ "$STATUS" = "loading" ]; then
                log "Reusing existing instance $INSTANCE_ID ($STATUS)"
            else
                log "Instance $INSTANCE_ID is gone ($STATUS). Cleaning up..."
                rm -f "$STATE_FILE"
                _kill_bg_monitor
                create_instance
            fi
        else
            create_instance
        fi
        echo ""
        verify_training
        echo ""
        xdg-open "https://cloud.vast.ai/instances/" 2>/dev/null &
        echo ""
        tail_log
        ;;
esac
