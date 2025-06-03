FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
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

# Set python3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip
RUN python -m pip install --upgrade pip==22.3.1 setuptools==59.5.0 wheel

# Create user
RUN useradd -m -u 1000 user
WORKDIR /app

# Install PyTorch CPU first
RUN pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install core dependencies with compatible versions
RUN pip install \
    numpy==1.19.5 \
    Pillow==8.3.2 \
    opencv-python==4.5.3.56 \
    cython==0.29.24

# Install detectron2 CPU
RUN pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html

# Install compatible versions for Python 3.8
RUN pip install \
    gradio==3.1.7 \
    huggingface_hub==0.8.1 \
    scipy==1.7.1 \
    scikit-image==0.18.3 \
    scikit-learn==1.0.2 \
    timm==0.4.12 \
    einops==0.3.2 \
    tqdm==4.62.3 \
    imutils==0.5.4 \
    shapely==1.8.0 \
    h5py==3.1.0 \
    regex==2021.11.10 \
    ftfy==6.0.3 \
    inflect==5.3.0 \
    gdown==4.2.0 \
    wget==3.2

# Optional dependencies (skip if causing issues)
RUN pip install submitit==1.4.1 || echo "submitit failed"
RUN pip install pytorch_lightning==1.5.10 || echo "pytorch_lightning failed"
RUN pip install wandb==0.12.9 || echo "wandb failed"
RUN pip install icecream==2.1.1 || echo "icecream failed"

# Try NATTEN
RUN pip install natten==0.14.6 -f https://shi-labs.com/natten/wheels/cpu/torch1.9/index.html || \
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
