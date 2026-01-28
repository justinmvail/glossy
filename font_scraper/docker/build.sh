#!/bin/bash
# Build Docker images for InkSight and TrOCR

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building InkSight container (TensorFlow 2.17 + GPU)..."
docker build -t inksight:latest ./inksight

echo ""
echo "Building TrOCR container (PyTorch + GPU)..."
docker build -t trocr:latest ./trocr

echo ""
echo "Done! Test with:"
echo "  docker run --rm --gpus all inksight python -c \"import tensorflow as tf; print('TF:', tf.__version__, 'GPU:', tf.config.list_physical_devices('GPU'))\""
echo "  docker run --rm --gpus all trocr python -c \"import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())\""
