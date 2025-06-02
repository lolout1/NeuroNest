# Stage 1: Build detectron2
FROM python:3.8-slim as builder

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    ninja-build \
    python3-dev \
    gcc-8 \
    g++-8 \
    && rm -rf /var/lib/apt/lists/*

# Set gcc-8 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8

# Install build dependencies
RUN pip install --upgrade "pip<24.1" setuptools wheel
RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install numpy==1.21.6 cython pycocotools

# Build detectron2 wheel
WORKDIR /tmp
RUN git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    git checkout v0.6 && \
    python setup.py bdist_wheel && \
    cp dist/*.whl /wheels/ || mkdir -p /wheels && \
    pip wheel . --wheel-dir=/wheels

# Stage 2: Runtime image
FROM python:3.8-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -u 1000 user

WORKDIR /app

# Install Python packages
RUN pip install --upgrade "pip<24.1" setuptools wheel

# Install PyTorch
RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install core dependencies
RUN pip install \
    numpy==1.21.6 \
    pillow==9.5.0 \
    pycocotools \
    opencv-python==4.5.5.64 \
    scipy==1.7.3 \
    matplotlib==3.5.3

# Copy and install detectron2 wheel from builder
COPY --from=builder /wheels/detectron2*.whl /tmp/
RUN pip install /tmp/detectron2*.whl && rm -f /tmp/detectron2*.whl

# Install remaining dependencies
RUN pip install \
    scikit-learn==1.0.2 \
    scikit-image==0.19.3 \
    gradio==3.35.2 \
    huggingface_hub==0.15.1 \
    tqdm \
    iopath==0.1.9 \
    omegaconf==2.1.2 \
    hydra-core==1.1.2 \
    tabulate \
    tensorboard \
    yacs \
    termcolor

# Switch to user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

# Copy application
COPY --chown=user:user . /app

# Install user requirements
COPY --chown=user:user requirements.txt /app/requirements.txt
RUN pip install --user --no-deps -r /app/requirements.txt || true

# Environment variables
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CUDA="0"
ENV TORCH_CUDA_ARCH_LIST=""
ENV PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["python", "app.py"]
