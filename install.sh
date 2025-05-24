#!/bin/bash
set -e

echo "🔥 Installing dependencies in correct order..."

# Stage 1: Install PyTorch FIRST
echo "📦 Installing PyTorch..."
pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    torchaudio==2.0.2+cpu

# Stage 2: Install core dependencies
echo "📦 Installing core dependencies..."
pip install --no-cache-dir \
    numpy==1.24.3 \
    scipy==1.10.1 \
    opencv-python==4.8.1.78 \
    Pillow==10.2.0 \
    scikit-learn==1.3.2 \
    scikit-image==0.21.0

# Stage 3: Install detectron2 (now torch is available)
echo "📦 Installing Detectron2..."
pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# Stage 4: Install ML utilities
echo "📦 Installing ML utilities..."
pip install --no-cache-dir \
    timm==0.9.12 \
    einops==0.7.0 \
    fairscale==0.4.13 \
    omegaconf==2.3.0 \
    huggingface-hub==0.20.3 \
    'transformers>=4.21.0'

# Stage 5: Install OneFormer
echo "📦 Installing OneFormer..."
pip install --no-cache-dir 'git+https://github.com/SHI-Labs/OneFormer.git'

# Stage 6: Install remaining dependencies
echo "📦 Installing remaining dependencies..."
pip install --no-cache-dir \
    gradio==4.19.2 \
    matplotlib==3.7.2 \
    tqdm==4.66.1 \
    threadpoolctl==3.2.0 \
    'requests>=2.28.0' \
    'pandas>=1.5.0'

echo "✅ All dependencies installed successfully!"
