FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.9 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Upgrade pip
RUN python -m pip install --upgrade pip==23.0.1 setuptools==65.5.0 wheel

# Create user
RUN useradd -m -u 1000 user
WORKDIR /app

# Install PyTorch CPU
RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install core dependencies with compatible versions
RUN pip install \
    numpy==1.21.6 \
    Pillow==9.3.0 \
    opencv-python==4.7.0.72 \
    cython==0.29.35

# Install detectron2 CPU
RUN pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.13/index.html

# Install compatible versions - Fixed for httpx/gradio compatibility
RUN pip install \
    gradio==3.50.2 \
    httpx==0.24.1 \
    httpcore==0.17.3 \
    huggingface_hub==0.16.4 \
    scipy==1.9.3 \
    scikit-image==0.19.3 \
    scikit-learn==1.2.2 \
    timm==0.6.13 \
    einops==0.6.1 \
    tqdm==4.65.0 \
    imutils==0.5.4 \
    shapely==2.0.1 \
    h5py==3.8.0 \
    regex==2023.3.23 \
    ftfy==6.1.1 \
    inflect==6.0.4 \
    gdown==4.7.1 \
    wget==3.2

# Optional dependencies
RUN pip install submitit==1.4.5 || echo "submitit failed"
RUN pip install pytorch_lightning==1.9.5 || echo "pytorch_lightning failed"
RUN pip install wandb==0.15.0 || echo "wandb failed"

# Try NATTEN
RUN pip install natten==0.14.6 -f https://shi-labs.com/natten/wheels/cpu/torch1.13/index.html || \
    echo "NATTEN installation failed"

# Switch to user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

# Copy application
COPY --chown=user:user . /app

# CPU environment
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CUDA="0"
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["python", "gradio_test.py"]
