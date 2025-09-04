FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
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

# Set python3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Install pip for Python 3.8 specifically
RUN curl https://bootstrap.pypa.io/pip/3.8/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Upgrade pip and install build tools
RUN python -m pip install --upgrade pip==23.0.1 setuptools==59.5.0 wheel cython

# Create user for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Install PyTorch 1.9 CPU first
RUN pip install --user torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install numpy first (required for other packages)
RUN pip install --user numpy==1.21.6

# Install core dependencies in order
RUN pip install --user \
    Pillow==8.3.2 \
    opencv-python==4.5.5.64

# Install scientific computing dependencies
RUN pip install --user \
    scipy==1.7.3 \
    scikit-image==0.19.3 \
    scikit-learn==1.0.2

# Install detectron2 for PyTorch 1.9 CPU
RUN pip install --user detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html

# Install pycocotools separately to avoid compilation issues
RUN pip install --user pycocotools --no-build-isolation

# Install ML dependencies
RUN pip install --user \
    timm==0.4.12 \
    einops==0.6.1 \
    h5py==3.7.0 \
    shapely==1.8.5 \
    tqdm==4.64.1 \
    imutils==0.5.4

# Install web framework dependencies with compatible versions
RUN pip install --user \
    httpx==0.23.0 \
    httpcore==0.15.0 \
    anyio==3.6.1 \
    starlette==0.19.1 \
    fastapi==0.78.0 \
    uvicorn==0.18.2

# Install Gradio and HuggingFace Hub with compatible versions
# These versions are compatible with each other and the rest of the stack
RUN pip install --user \
    huggingface_hub==0.19.3 \
    gradio==4.44.1

# Try to install NATTEN (optional)
RUN pip install --user natten==0.14.6 -f https://shi-labs.com/natten/wheels/cpu/torch1.9/index.html || \
    echo "NATTEN installation failed - continuing without it"

# Install remaining dependencies
RUN pip install --user \
    PyYAML==5.4.1 \
    matplotlib==3.5.3 \
    regex==2022.10.31 \
    ftfy==6.1.1 \
    wandb \
    diffdist \
    inflect==6.0.4 \
    gdown==4.5.4 \
    wget==3.2

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
