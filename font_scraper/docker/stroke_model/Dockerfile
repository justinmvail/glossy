# Stroke Model Container - CNN+Transformer stroke prediction with GPU
# Trains a neural network to predict centerline strokes from glyph rasters.
# Uses pydiffvg for differentiable rendering loss.

FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10, build tools, and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    cmake \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install PyTorch (same version as DiffVG container for compatibility)
RUN pip3 install --no-cache-dir \
    torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Install DiffVG dependencies (numpy<2 required — DiffVG compiled against numpy 1.x)
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    scipy \
    pillow \
    scikit-image

# Clone and build DiffVG from source (for differentiable rendering loss)
# Set CUDA env vars so cmake finds nvcc and enables GPU support
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
WORKDIR /opt
# Cache-bust: force CUDA build v2
RUN git clone https://github.com/BachiLi/diffvg.git && \
    cd diffvg && \
    git submodule update --init --recursive && \
    DIFFVG_CUDA=1 python3 setup.py install

# Install additional ML dependencies
RUN pip3 install --no-cache-dir \
    timm>=0.9.0 \
    tensorboard>=2.14.0 \
    cssutils \
    matplotlib

WORKDIR /app

# Copy all model code
COPY model.py /app/model.py
COPY losses.py /app/losses.py
COPY dataset.py /app/dataset.py
COPY train.py /app/train.py
COPY predict.py /app/predict.py
COPY render_utils.py /app/render_utils.py

CMD ["python3"]
