FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    wget \
    curl \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create user for Hugging Face
RUN useradd -m -u 1000 user

WORKDIR /app

# Upgrade pip and install setuptools
RUN pip install --upgrade pip
RUN pip install setuptools==69.5.1 wheel

# Install PyTorch and dependencies
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Force specific numpy/pillow versions
RUN pip install numpy==1.24.3 pillow==10.0.0

# Install COCO API and OpenCV
RUN pip install pycocotools opencv-python

# Install detectron2 - FIXED VERSION
RUN git clone https://github.com/facebookresearch/detectron2.git /tmp/detectron2 && \
    cd /tmp/detectron2 && \
    git checkout v0.6 && \
    pip install --no-build-isolation --no-deps . && \
    cd / && \
    rm -rf /tmp/detectron2

# Install detectron2 dependencies manually (since v0.6 has no requirements.txt)
RUN pip install \
    fvcore \
    iopath \
    omegaconf \
    hydra-core \
    black \
    pyyaml \
    matplotlib \
    tqdm \
    termcolor \
    yacs \
    tabulate \
    cloudpickle \
    Pillow \
    scipy

# Install additional dependencies
RUN pip install \
    gradio \
    huggingface_hub \
    scikit-learn \
    scikit-image

# Switch to user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

# Copy application files
COPY --chown=user:user . /app

# Install any remaining user requirements
COPY --chown=user:user requirements.txt /app/
RUN pip install --user --no-deps -r requirements.txt || true

# Environment variables
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CUDA="0"
ENV TORCH_CUDA_ARCH_LIST=""

EXPOSE 7860
CMD ["python", "app.py"]
