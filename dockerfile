FROM python:3.8-slim

# Set working directory
WORKDIR /home/user/app

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

# Create user
RUN useradd -m -u 1000 user

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies in correct order
RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir torch==1.10.1 torchvision==0.11.2 numpy==1.24.4
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=1000:1000 . .

# Switch to user
USER user

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
