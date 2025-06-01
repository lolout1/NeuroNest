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
    && rm -rf /var/lib/apt/lists/*

# Set up user
RUN useradd -m -u 1000 user

# Install PyTorch CPU version as root (system-wide)
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Clone and install detectron2 as root (without -e flag)
RUN git clone https://github.com/facebookresearch/detectron2 /tmp/detectron2 && \
    cd /tmp/detectron2 && \
    pip install .

# Switch to user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Copy and install other requirements
COPY --chown=user:user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy app files
COPY --chown=user:user . .

EXPOSE 7860
CMD ["python", "app.py"]
