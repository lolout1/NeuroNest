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

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies in correct order
RUN pip install --no-cache-dir pip -U
RUN pip install --no-cache-dir torch==1.10.1 torchvision==0.11.2 numpy==1.24.4
RUN pip install --no-cache-dir -r requirements.txt || echo "Some packages failed but continuing..."

# Copy application code
COPY . .

# Set environment variables for CPU fallback
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CPU=1

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
