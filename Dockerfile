FROM python:3.10

# Set up user to avoid permission issues
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Switch to root to install system packages
USER root

# Install ALL necessary system dependencies for OpenCV and GUI libraries
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    python3-dev \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk2.0-dev \
    # Additional useful libraries
    wget \
    unzip \
    ffmpeg \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Switch back to user
USER user

# Copy requirements first (for better caching)
COPY --chown=user:user requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir --upgrade pip && \
    pip install --user --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --user --no-cache-dir -r requirements.txt

# Install detectron2 separately after torch is installed
RUN git clone https://github.com/facebookresearch/detectron2 $HOME/app/detectron2 && \
    pip install --user -e $HOME/app/detectron2

# Copy app files
COPY --chown=user:user . .

# Expose port for Gradio
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
