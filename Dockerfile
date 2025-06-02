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

# CRITICAL: Downgrade setuptools to avoid the deprecated warnings
RUN pip install --upgrade pip
RUN pip install setuptools==69.5.1 wheel

# Install PyTorch and core dependencies
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install numpy==1.24.3 pillow==10.0.0

# Install COCO API first (required by detectron2)
RUN pip install pycocotools opencv-python

# Method 1: Install detectron2 without editable mode and without build isolation
RUN git clone https://github.com/facebookresearch/detectron2.git /tmp/detectron2 && \
    cd /tmp/detectron2 && \
    git checkout v0.6 && \
    pip install --no-build-isolation --no-deps . && \
    pip install -r requirements.txt

# Alternative Method 2: Build wheel first
# RUN git clone https://github.com/facebookresearch/detectron2.git /tmp/detectron2 && \
#     cd /tmp/detectron2 && \
#     git checkout v0.6 && \
#     python setup.py bdist_wheel && \
#     pip install dist/*.whl && \
#     pip install -r requirements.txt

# Install other dependencies
RUN pip install \
    gradio \
    huggingface_hub \
    scipy \
    scikit-learn \
    scikit-image \
    matplotlib \
    tqdm

# Switch to user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

# Copy application
COPY --chown=user:user . /app

# Install remaining requirements as user
COPY --chown=user:user requirements.txt /app/
RUN pip install --user --no-deps -r requirements.txt || true

# Environment variables
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CUDA="0"
ENV TORCH_CUDA_ARCH_LIST=""

EXPOSE 7860
CMD ["python", "app.py"]
