FROM python:3.10 as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget build-essential \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    libgomp1 ffmpeg cmake \
    && rm -rf /var/lib/apt/lists/*

# STAGE 1: Install PyTorch FIRST
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    torchaudio==2.0.2+cpu

# STAGE 2: Install other dependencies
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    scipy==1.10.1 \
    opencv-python==4.8.1.78 \
    Pillow==10.2.0 \
    scikit-learn==1.3.2 \
    scikit-image==0.21.0 \
    gradio==4.19.2 \
    matplotlib==3.7.2 \
    tqdm==4.66.1 \
    requests>=2.28.0 \
    pandas>=1.5.0 \
    omegaconf==2.3.0 \
    timm==0.9.12 \
    einops==0.7.0 \
    fairscale==0.4.13 \
    huggingface-hub==0.20.3 \
    transformers>=4.21.0 \
    threadpoolctl==3.2.0

# STAGE 3: Install Detectron2 (now torch is available)
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# Final stage
FROM python:3.10

WORKDIR /home/user/app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY . .

# Install local OneFormer
RUN cd oneformer && pip install -e .

# Create user
RUN useradd -m -u 1000 user
USER user

ENV PYTHONPATH=/home/user/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
