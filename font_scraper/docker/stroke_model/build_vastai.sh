#!/bin/bash
# build_vastai.sh — Build, tag, and push the Vast.ai training image
#
# Usage:
#   ./build_vastai.sh           # Build, tag with timestamp, push
#   ./build_vastai.sh --local   # Build only, no push
#
# The image includes: base pytorch + pip deps + code only
# Fonts/db are downloaded from Google Drive at boot (see vastai_deploy.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKER_REPO="justinmvail/stroke-model"
TAG_FILE="$SCRIPT_DIR/.last_build_tag"
LOCAL_ONLY=false

if [ "${1:-}" = "--local" ]; then
    LOCAL_ONLY=true
fi

log() { echo -e "\033[0;32m[build]\033[0m $*"; }
err() { echo -e "\033[0;31m[build]\033[0m $*" >&2; }

VERSION_TAG="$(date +%Y%m%d-%H%M%S)"
FULL_TAG="$DOCKER_REPO:$VERSION_TAG"
LATEST_TAG="$DOCKER_REPO:vastai"

log "Building image: $FULL_TAG"
docker build \
    --memory=4g \
    -f "$SCRIPT_DIR/Dockerfile.vastai" \
    -t "$FULL_TAG" \
    -t "$LATEST_TAG" \
    "$SCRIPT_DIR"

log "Build complete."

if $LOCAL_ONLY; then
    log "Local build only, skipping push."
else
    log "Pushing $FULL_TAG ..."
    docker push "$FULL_TAG"
    log "Pushing $LATEST_TAG ..."
    docker push "$LATEST_TAG"
    log "Push complete."
fi

echo "$FULL_TAG" > "$TAG_FILE"
log "Image tag saved to $TAG_FILE"
log "Done: $FULL_TAG"
