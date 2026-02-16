FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (OpenCV needs libgl1, detectron2 needs build-essential for compilation)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create user for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Install PyTorch 2.5.1 CPU first (large download, cached layer)
RUN pip install --user --no-cache-dir \
    torch==2.5.1+cpu torchvision==0.20.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install detectron2 from main branch (PyTorch 2.x compatible via PR #5454)
# Only used for MaskRCNN blackspot detection
RUN pip install --user --no-cache-dir \
    'git+https://github.com/facebookresearch/detectron2.git'

# Install remaining Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user:user . /app

# Environment variables for CPU-only operation
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV PYTHONUNBUFFERED=1

# Expose port for Gradio
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
