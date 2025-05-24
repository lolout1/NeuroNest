FROM python:3.10-slim

WORKDIR /home/user/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# CRITICAL: Install PyTorch CPU FIRST (required for detectron2)
RUN pip install --no-cache-dir \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    torchaudio==2.0.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies BEFORE detectron2
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    opencv-python==4.8.1.78 \
    Pillow==10.2.0 \
    scipy==1.10.1 \
    scikit-learn==1.3.2 \
    scikit-image==0.21.0

# Copy application files (including detectron2 directory from LFS)
COPY . .

# CRITICAL: Install detectron2 from local directory using python -m pip
RUN cd detectron2 && \
    python -m pip install -e . && \
    cd .. && \
    echo "Detectron2 installed successfully from local directory"

# Install remaining dependencies
RUN pip install --no-cache-dir \
    gradio==4.44.1 \
    matplotlib==3.7.2 \
    tqdm==4.66.1 \
    omegaconf==2.3.0 \
    timm==0.9.12 \
    einops==0.7.0 \
    fairscale==0.4.13 \
    huggingface-hub==0.20.3 \
    transformers>=4.21.0 \
    threadpoolctl==3.2.0

# Install OneFormer package if available
RUN if [ -f oneformer/setup.py ]; then \
        cd oneformer && python -m pip install -e . --no-deps && cd ..; \
    fi

# Verify detectron2 installation
RUN python -c "import detectron2; from detectron2.config import get_cfg; print('✅ Detectron2 successfully installed and functional')"

# Create user
RUN useradd -m -u 1000 user && chown -R user:user /home/user/app
USER user

# Environment variables
ENV PYTHONPATH=/home/user/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV DEVICE=cpu
ENV TORCH_HOME=/tmp

EXPOSE 7860

CMD ["python", "app.py"]
