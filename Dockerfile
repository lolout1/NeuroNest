FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Create user
RUN useradd -m -u 1000 user
WORKDIR /app

# Upgrade pip
RUN python -m pip install --upgrade pip==21.3.1 setuptools==59.5.0 wheel

# Install PyTorch CPU (specific version that works with detectron2)
RUN pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install numpy first (dependency for many packages)
RUN pip install numpy==1.21.6

# Build and install detectron2 from source for compatibility
RUN pip install cython pybind11
RUN git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    git checkout v0.6 && \
    pip install -e . && \
    cd .. && \
    rm -rf detectron2/.git

# Install core dependencies with compatible versions
RUN pip install \
    Pillow==8.3.2 \
    opencv-python==4.5.3.56 \
    scipy==1.7.1 \
    scikit-image==0.18.3 \
    scikit-learn==1.0.2 \
    h5py==3.1.0 \
    shapely==1.7.1 \
    imutils==0.5.4

# Install ML dependencies with compatible versions
RUN pip install \
    timm==0.4.12 \
    einops==0.3.2 \
    tqdm==4.62.3 \
    pycocotools

# Install compatible gradio and huggingface_hub
RUN pip install \
    gradio==3.1.7 \
    huggingface_hub==0.10.1

# Install other dependencies
RUN pip install \
    gdown==4.5.1 \
    wget==3.2 \
    ftfy==6.0.3 \
    regex==2021.11.10 \
    inflect==5.3.0 \
    submitit==1.4.1

# Optional: pytorch_lightning (skip if not needed for faster builds)
# RUN pip install pytorch_lightning==1.5.10

# Switch to user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

# Copy application
COPY --chown=user:user . /app

# CPU environment variables
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CUDA="0"
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["python", "gradio_test.py"]
