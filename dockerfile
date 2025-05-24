FROM python:3.8.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    cmake \
    rsync \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender-dev \
    libgomp1 \
    build-essential \
    python3-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Set working directory
WORKDIR /home/user/app

# Copy requirements first for better caching
COPY requirements.txt packages.txt ./

# Install Python dependencies in correct order
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create user
RUN useradd -m -u 1000 user && chown -R user:user /home/user/app
USER user

# Environment variables for CPU-only execution
ENV CUDA_VISIBLE_DEVICES=""
ENV TORCH_CUDA_ARCH_LIST=""
ENV FORCE_CUDA="0"

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
