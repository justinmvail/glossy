# DiffVG Container - Differentiable Vector Graphics with GPU
# For gradient-based stroke optimization against font glyph masks

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

# Install PyTorch (conservative version for DiffVG compatibility)
RUN pip3 install --no-cache-dir \
    torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Install DiffVG dependencies (numpy<2 required — DiffVG compiled against numpy 1.x)
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    scipy \
    pillow \
    scikit-image \
    matplotlib \
    svgpathtools \
    svgwrite \
    cssutils

# Clone and build DiffVG from source
WORKDIR /opt
RUN git clone https://github.com/BachiLi/diffvg.git && \
    cd diffvg && \
    git submodule update --init --recursive && \
    python3 setup.py install

WORKDIR /app

# Copy optimization script
COPY optimize_strokes.py /app/optimize_strokes.py

CMD ["python3"]
