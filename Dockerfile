FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget curl \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Stage 1: Install PyTorch first
RUN pip install --upgrade pip==22.3.1 && \
    pip install --no-cache-dir \
    torch==1.9.0+cpu \
    torchvision==0.10.0+cpu \
    torchaudio==0.9.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Stage 2: Install core dependencies
RUN pip install --no-cache-dir \
    numpy==1.21.6 \
    Pillow==8.3.2 \
    cython==0.29.35 \
    setuptools==59.5.0

# Stage 3: Install detectron2
RUN pip install --no-cache-dir \
    detectron2==0.6 \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html

# Stage 4: Try to install NATTEN (may fail, that's OK)
RUN pip install --no-cache-dir \
    natten==0.14.6 \
    -f https://shi-labs.com/natten/wheels/cpu/torch1.9/index.html || \
    echo "NATTEN installation failed, continuing without it"

# Stage 5: Install remaining requirements
RUN pip install --no-cache-dir \
    opencv-python==4.5.5.64 \
    imutils==0.5.4 \
    scipy==1.7.3 \
    shapely==1.8.5 \
    h5py==3.7.0 \
    scikit-image==0.19.3 \
    scikit-learn \
    timm==0.4.12 \
    einops==0.6.1 \
    icecream==2.1.3 \
    wandb==0.13.5 \
    tqdm==4.64.1 \
    ftfy==6.1.1 \
    regex==2022.10.31 \
    inflect==6.0.4 \
    gdown==4.5.4 \
    wget==3.2 \
    huggingfac_hub==0.15.0 \
    pytorch_lightning==1.8.6 \
    gradio==3.35.2 \
    diffdist \
    submitit==1.4.5

# Switch to user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

# Copy application
COPY --chown=user:user . /app

# Create directories

# Download example images

# CPU environment
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CUDA="0"
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

EXPOSE 7860
CMD ["python", "gradio_test.py"]
