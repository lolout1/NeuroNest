FROM python:3.10

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
    unzip \
    ffmpeg \
    libopencv-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -u 1000 user

# Install specific PyTorch CPU version FIRST
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu --index-url https://download.pytorch.org/whl/cpu --no-deps
RUN pip install numpy pillow typing-extensions

# Install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 /tmp/detectron2 && \
    cd /tmp/detectron2 && \
    git checkout v0.6 && \
    pip install -e . --no-deps && \
    pip install -r requirements.txt

# Switch to user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Set environment for CPU
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CUDA="0"
ENV TORCH_CUDA_ARCH_LIST=""

# Copy and install requirements
COPY --chown=user:user requirements-no-torch.txt requirements.txt
RUN pip install --user --no-cache-dir -r requirements-no-torch.txt

# Copy app files
COPY --chown=user:user . .

# Install NATTEN last to ensure correct torch version
RUN pip install --user natten==0.14.6+torch200cpu -f https://shi-labs.com/natten/wheels/cpu/torch2.0.0/index.html --no-deps

EXPOSE 7860
CMD ["python", "app.py"]
