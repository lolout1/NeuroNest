FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
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
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set python3.9 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Upgrade pip and install build tools
RUN python -m pip install --upgrade pip==23.3.1 setuptools==68.0.0 wheel cython

# Create user for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Install PyTorch 1.10.1 CPU (OneFormer's official supported version)
RUN pip install --user torch==1.10.1+cpu torchvision==0.11.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install numpy first (required for other packages) - compatible with torch 1.10
RUN pip install --user "numpy>=1.21.0,<1.24.0"

# Install core dependencies compatible with Python 3.9 and torch 1.10
RUN pip install --user \
    "Pillow>=9.0.0,<10.0.0" \
    "opencv-python>=4.5.0,<5.0.0"

# Install scientific computing dependencies
RUN pip install --user \
    "scipy>=1.7.0,<2.0.0" \
    "scikit-image>=0.19.0,<1.0.0" \
    "scikit-learn>=1.0.0,<2.0.0"

# Install detectron2 v0.6 with REAL prebuilt wheels for PyTorch 1.10 CPU
RUN pip install --user detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html

# Install pycocotools separately to avoid compilation issues
RUN pip install --user pycocotools --no-build-isolation

# Install ML dependencies
RUN pip install --user \
    "timm>=0.9.0,<1.0.0" \
    "einops>=0.7.0,<1.0.0" \
    "h5py>=3.8.0,<4.0.0" \
    "shapely>=2.0.0,<3.0.0" \
    "tqdm>=4.65.0,<5.0.0" \
    "imutils>=0.5.4,<1.0.0"

# Install web framework dependencies with compatible versions
# Using compatible versions for Gradio 4.44.1
RUN pip install --user \
    "httpx>=0.24.0" \
    "anyio>=3.7.0" \
    "fastapi>=0.100.0" \
    "uvicorn[standard]>=0.23.0"

# Install Gradio and HuggingFace Hub with compatible versions
# Gradio 4.44.1 requires huggingface_hub>=0.20.0
RUN pip install --user \
    "huggingface_hub>=0.20.0,<1.0.0" \
    "gradio==4.44.1"

# Try to install NATTEN for PyTorch 1.10 (optional - may not have wheels)
RUN pip install --user natten -f https://shi-labs.com/natten/wheels/cpu/torch1.10/index.html || \
    echo "NATTEN installation failed (expected, not critical) - continuing without it"

# Install remaining dependencies
RUN pip install --user \
    "PyYAML>=6.0" \
    "matplotlib>=3.7.0,<4.0.0" \
    "regex>=2023.0.0" \
    "ftfy>=6.1.0" \
    "wandb" \
    "diffdist" \
    "inflect>=6.0.0" \
    "gdown>=4.6.0" \
    "wget>=3.2"

# Copy application files
COPY --chown=user:user . /app

# Set environment variables for CPU-only operation
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CUDA="0"
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV PYTHONUNBUFFERED=1

# Expose port for Gradio
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
