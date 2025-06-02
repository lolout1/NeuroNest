FROM python:3.10-slim

# Install system dependencies including those needed for building detectron2
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

# Create user (required by Hugging Face)
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Install Python dependencies in the correct order
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch dependencies FIRST
RUN pip install sympy filelock jinja2 networkx requests typing-extensions

# Install PyTorch CPU version
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Install compatible versions of numpy and pillow
RUN pip install numpy==1.24.3 pillow==9.5.0

# Install detectron2 dependencies first
RUN pip install \
    pycocotools \
    opencv-python==4.8.1.78 \
    scipy==1.10.1 \
    scikit-learn==1.3.0 \
    scikit-image==0.21.0 \
    matplotlib==3.7.2 \
    tqdm \
    cloudpickle \
    tabulate \
    tensorboard \
    packaging \
    Pillow==9.5.0 \
    pycocotools \
    matplotlib \
    iopath \
    omegaconf \
    hydra-core \
    black

# Build and install detectron2 from source (more reliable)
RUN git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    git checkout v0.6 && \
    pip install -e . && \
    cd .. && \
    rm -rf detectron2/.git

# Install other core dependencies
RUN pip install \
    gradio==3.50.2 \
    huggingface_hub==0.19.4

# Switch to user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

# Copy application files
COPY --chown=user:user requirements.txt /app/requirements.txt

# Install remaining requirements
RUN pip install --user --no-deps -r /app/requirements.txt || true

COPY --chown=user:user . /app

# Set environment for CPU
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CUDA="0"
ENV TORCH_CUDA_ARCH_LIST=""

# Create necessary directories
RUN mkdir -p /app/oneformer && \
    mkdir -p /app/utils && \
    mkdir -p /app/configs && \
    mkdir -p /app/demo

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
