FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3.8-distutils \
    build-essential \
    cmake \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-7-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create user
RUN useradd -m -u 1000 user
WORKDIR /app

# Install pip for python3.8
RUN python -m pip install --upgrade pip==22.3.1 setuptools wheel

# Install PyTorch CPU with specific versions that work with detectron2
RUN pip install --no-cache-dir \
    torch==1.9.0+cpu \
    torchvision==0.10.0+cpu \
    torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install numpy and other core dependencies
RUN pip install --no-cache-dir \
    numpy==1.21.6 \
    Pillow==8.3.2 \
    cython==0.29.35 \
    pycocotools

# Install detectron2 - using the exact URL that works
RUN pip install --no-cache-dir \
    detectron2==0.6 \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html

# Install OpenCV and other dependencies
RUN pip install --no-cache-dir \
    opencv-python==4.5.5.64 \
    scipy==1.7.3 \
    scikit-image==0.19.3 \
    scikit-learn==1.0.2 \
    matplotlib==3.5.3

# Install gradio and other packages
RUN pip install --no-cache-dir \
    gradio==3.35.2 \
    timm==0.4.12 \
    einops==0.6.1 \
    tqdm==4.64.1 \
    huggingface_hub==0.14.8 \
    imutils==0.5.4 \
    shapely==1.8.5 \
    h5py==3.7.0

# Install remaining packages (skip problematic ones)
RUN pip install --no-cache-dir \
    icecream==2.1.3 \
    ftfy==6.1.1 \
    regex==2022.10.31 \
    inflect==6.0.4 \
    gdown==4.5.4 \
    wget==3.2 \
    submitit==1.4.5 || true

# Switch to user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

# Copy application
COPY --chown=user:user . /app

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CUDA="0"
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["python", "gradio_test.py"]
