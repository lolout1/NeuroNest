FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    cmake \
    build-essential \
    python3-dev \
    libopencv-dev \
    libglib2.0-0 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /app

# Install Python dependencies in stages
RUN pip install --no-cache-dir pip setuptools wheel -U

# Stage 1: Core dependencies
RUN pip install --no-cache-dir \
    torch==1.10.1 \
    torchvision==0.11.2 \
    numpy==1.24.4

# Stage 2: Basic libraries
RUN pip install --no-cache-dir \
    scipy==1.10.1 \
    matplotlib==3.7.5 \
    Pillow==10.2.0 \
    opencv-python==4.11.0.86

# Stage 3: ML libraries that depend on torch
RUN pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/detectron2.git@v0.6"

# Stage 4: OneFormer and related
RUN pip install --no-cache-dir \
    "git+https://github.com/SHI-Labs/OneFormer.git" \
    timm==0.4.12 \
    transformers==4.33.2 \
    huggingface-hub==0.20.3

# Stage 5: Optional NATTEN (may fail, that's ok)
RUN pip install --no-cache-dir \
    "git+https://github.com/SHI-Labs/NATTEN.git" || echo "NATTEN installation failed, continuing..."

# Stage 6: Remaining dependencies
RUN pip install --no-cache-dir \
    scikit-image==0.21.0 \
    scikit-learn==1.3.2 \
    gradio==4.44.1 \
    colorspacious==1.1.2 \
    omegaconf==2.3.0 \
    pyyaml==6.0.1 \
    tqdm==4.66.1 \
    requests==2.31.0 \
    psutil==5.9.5 \
    fairscale==0.4.13 \
    einops==0.7.0 \
    fvcore==0.1.5.post20221221 \
    pycocotools==2.0.7 \
    threadpoolctl==3.2.0 \
    imageio==2.31.5

# Copy application code
COPY . .

# Set environment variables for CPU optimization
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CPU=1
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["python", "app.py"]
