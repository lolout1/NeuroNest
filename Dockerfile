FROM python:3.8-slim

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
    gcc-8 \
    g++-8 \
    && rm -rf /var/lib/apt/lists/*

# Set gcc-8 as default (better compatibility with detectron2)
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8

# Create user
RUN useradd -m -u 1000 user

WORKDIR /app

# Upgrade pip to a compatible version
RUN pip install --upgrade "pip<24.1" setuptools wheel

# Install PyTorch for Python 3.8
RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install compatible numpy and pillow
RUN pip install numpy==1.21.6 pillow==9.5.0

# Install detectron2 dependencies
RUN pip install \
    cython \
    pycocotools \
    opencv-python==4.5.5.64 \
    scipy==1.7.3 \
    matplotlib==3.5.3 \
    cloudpickle \
    tabulate \
    tensorboard \
    yacs \
    termcolor \
    future \
    fvcore \
    iopath==0.1.9 \
    omegaconf==2.1.2 \
    hydra-core==1.1.2

# Install detectron2 from source with Python 3.8 compatibility
RUN git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    git checkout v0.6 && \
    python -m pip install -e . && \
    cd ..

# Install additional dependencies
RUN pip install \
    scikit-learn==1.0.2 \
    scikit-image==0.19.3 \
    gradio==3.35.2 \
    huggingface_hub==0.15.1 \
    tqdm

# Switch to user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

# Copy application
COPY --chown=user:user . /app

# Install additional requirements
COPY --chown=user:user requirements.txt /app/requirements.txt
RUN pip install --user --no-deps -r /app/requirements.txt || true

# Environment variables
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CUDA="0"
ENV TORCH_CUDA_ARCH_LIST=""
ENV PYTHONUNBUFFERED=1

# Create necessary directories
RUN mkdir -p /home/user/.cache

EXPOSE 7860
CMD ["python", "app.py"]
