# InkSight Container - TensorFlow 2.17 with GPU
# For font vectorization using Google's InkSight model

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TF_CPP_MIN_LOG_LEVEL=2

# Install Python 3.10 and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install TensorFlow 2.17 (working version for InkSight)
RUN pip3 install --no-cache-dir \
    tensorflow[and-cuda]==2.17.0 \
    tensorflow-text==2.17.0 \
    numpy==1.26.4 \
    pillow \
    scipy \
    matplotlib

# Create working directory
WORKDIR /app

# Vectorizer code and fonts will be mounted as volumes at runtime
# Model path: /app/model (mounted volume)

# Default command - Python REPL for testing
CMD ["python3"]
