FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create user (required by HF Spaces)
RUN useradd -m -u 1000 user

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip==22.3.1 && \
    pip install --no-cache-dir -r requirements.txt

# Verify installations
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CPU: {not torch.cuda.is_available()}')"
RUN python -c "import natten; print(f'NATTEN: {natten.__version__}')"
RUN python -c "import detectron2; print(f'Detectron2: {detectron2.__version__}')"

# Switch to user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

# Copy application
COPY --chown=user:user . /app

# CPU optimizations
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CUDA="0"
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

EXPOSE 7860
CMD ["python", "gradio_app.py"]
