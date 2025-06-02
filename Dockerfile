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
    libgtk2.0-dev \
    wget \
    curl \
    unzip \
    ffmpeg \
    libopencv-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -u 1000 user

# Install Python dependencies in correct order
RUN pip install --upgrade pip setuptools wheel

# Install sympy FIRST (required by torch)
RUN pip install sympy

# Install PyTorch CPU version
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Install detectron2 from pre-built wheel (more reliable)
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html

# Install other dependencies
RUN pip install \
    numpy>=1.23.0,<2.0.0 \
    opencv-python \
    scipy \
    scikit-learn \
    gradio \
    huggingface_hub

# Switch to user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Set environment for CPU
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CUDA="0"
ENV TORCH_CUDA_ARCH_LIST=""

# Copy requirements and install remaining dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy app files
COPY --chown=user:user . .

# Clone and setup OneFormer if needed
RUN if [ ! -d "oneformer" ]; then \
    git clone https://github.com/SHI-Labs/OneFormer.git oneformer && \
    cd oneformer && \
    pip install --user -e . ; \
    fi

EXPOSE 7860
CMD ["python", "gradio_test.py"]
