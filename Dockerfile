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
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set python3.9 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Upgrade pip
RUN python -m pip install --upgrade pip==23.0.1 setuptools==65.5.0 wheel

# Create user
RUN useradd -m -u 1000 user
WORKDIR /app

# Install PyTorch 1.9 CPU - FIXED VERSION
RUN pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install core dependencies compatible with torch 1.9
RUN pip install \
    numpy==1.19.5 \
    Pillow==8.3.2 \
    opencv-python==4.5.3.56 \
    cython==0.29.24

# Install detectron2 for PyTorch 1.9 CPU - CORRECT URL
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html

# Install Gradio with compatible dependencies
RUN pip install \
    gradio==3.1.7 \
    httpx==0.23.0 \
    httpcore==0.15.0 \
    anyio==3.6.1 \
    starlette==0.19.1 \
    fastapi==0.78.0 \
    uvicorn==0.18.2

# Install other dependencies
RUN pip install \
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
    wget==3.2 \
    PyYAML==5.4.1 \
    pycocotools==2.0.4 \
    matplotlib==3.5.1

# Optional dependencies
RUN pip install submitit==1.4.1 || echo "submitit failed"
RUN pip install pytorch_lightning==1.5.10 || echo "pytorch_lightning failed"
RUN pip install wandb==0.12.9 || echo "wandb failed"
RUN pip install icecream==2.1.1 || echo "icecream failed"

# Try NATTEN for torch 1.9
RUN pip install natten==0.14.6 -f https://shi-labs.com/natten/wheels/cpu/torch1.9/index.html || \
    echo "NATTEN installation failed"

# OneFormer specific dependencies
RUN pip install \
    omegaconf==2.1.1 \
    hydra-core==1.1.1 \
    termcolor==1.1.0 \
    tabulate==0.8.9 \
    yacs==0.1.8 \
    cloudpickle==2.0.0 \
    packaging==21.3

# Install additional tools
RUN pip install \
    iopath==0.1.9 \
    fvcore==0.1.5.post20220512 \
    pydot==1.4.2 \
    portalocker==2.5.1

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
