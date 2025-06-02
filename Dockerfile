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
    && rm -rf /var/lib/apt/lists/*

# Create user (required by Hugging Face)
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Install Python dependencies in the correct order
# First, upgrade pip and install essential tools
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch dependencies FIRST
RUN pip install sympy filelock jinja2 networkx requests typing-extensions

# Install PyTorch CPU version WITH dependencies
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Install numpy and pillow (specific versions for compatibility)
RUN pip install numpy==1.24.3 pillow==10.0.0

# Install detectron2 from pre-built wheel (much more reliable than building from source)
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html

# Install other core dependencies
RUN pip install \
    opencv-python \
    scipy \
    scikit-learn \
    scikit-image \
    matplotlib \
    gradio \
    huggingface_hub \
    tqdm

# Switch to user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

# Copy application files
COPY --chown=user:user requirements.txt /app/requirements.txt
RUN pip install --user -r /app/requirements.txt

COPY --chown=user:user . /app

# Set environment for CPU
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CUDA="0"
ENV TORCH_CUDA_ARCH_LIST=""

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "gradio_test.py"]
