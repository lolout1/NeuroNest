FROM python:3.8.18-slim

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
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Create user
RUN useradd -m -u 1000 user

# Install Python dependencies in STRICT order
RUN pip install --no-cache-dir -U pip setuptools wheel

# STEP 1: Install torch FIRST (absolutely critical)
RUN pip install --no-cache-dir \
    torch==1.10.1+cpu \
    torchvision==0.11.2+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# STEP 2: Install basic dependencies
RUN pip install --no-cache-dir \
    numpy==1.24.4 \
    opencv-python==4.8.1.78 \
    Pillow==10.2.0 \
    scipy==1.10.1 \
    matplotlib==3.7.5

# STEP 3: Install detectron2 dependencies using git+https format
RUN pip install --no-cache-dir \
    git+https://github.com/cocodataset/panopticapi.git \
    git+https://github.com/mcordts/cityscapesScripts.git

# STEP 4: Install detectron2 using git+https format
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git@v0.6

# STEP 5: Install OneFormer using git+https format
RUN pip install --no-cache-dir git+https://github.com/SHI-Labs/OneFormer.git

# STEP 6: Install remaining ML dependencies
RUN pip install --no-cache-dir \
    timm==0.4.12 \
    transformers==4.33.2 \
    huggingface-hub==0.20.3 \
    fairscale==0.4.13 \
    einops==0.7.0 \
    fvcore==0.1.5.post20221221 \
    scikit-learn==1.3.0 \
    scikit-image==0.21.0

# STEP 7: Install web interface and utilities
RUN pip install --no-cache-dir \
    gradio==4.44.1 \
    omegaconf==2.3.0 \
    pyyaml==6.0.1 \
    tqdm==4.66.1 \
    requests==2.31.0 \
    psutil==5.9.5 \
    packaging==23.1

# STEP 8: Install color science and additional utilities
RUN pip install --no-cache-dir \
    colorspacious==1.1.2 \
    colour-science==0.4.2 \
    imageio==2.31.5 \
    tifffile==2023.7.10 \
    PyWavelets==1.4.1 \
    threadpoolctl==3.2.0 \
    pandas==2.0.3 \
    seaborn==0.12.2 \
    pycocotools==2.0.7 \
    diffdist==0.1

# Copy application code
COPY --chown=1000:1000 . .

# Switch to user
USER user

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
