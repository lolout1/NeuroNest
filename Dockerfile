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

# Create user
RUN useradd -m -u 1000 user

WORKDIR /app

# Install Python packages
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch FIRST
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Install numpy and pillow with specific versions
RUN pip install numpy==1.24.3 pillow==9.5.0

# Install detectron2 dependencies
RUN pip install \
    pycocotools \
    opencv-python==4.8.1.78 \
    scipy==1.10.1 \
    matplotlib==3.7.2 \
    iopath==0.1.9 \
    omegaconf \
    hydra-core \
    cloudpickle \
    tabulate \
    tensorboard \
    yacs \
    termcolor \
    future \
    fvcore

# Clone and install detectron2 with --no-build-isolation
RUN git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    git checkout v0.6 && \
    pip install --no-build-isolation -e . && \
    cd ..

# Install other dependencies
RUN pip install \
    scikit-learn==1.3.0 \
    scikit-image==0.21.0 \
    gradio==3.50.2 \
    huggingface_hub==0.19.4 \
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

EXPOSE 7860
CMD ["python", "app.py"]
