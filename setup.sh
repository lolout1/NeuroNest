#!/bin/bash
# Setup script for Hugging Face Spaces

# Install system dependencies
apt-get update && apt-get install -y git build-essential

# Downgrade setuptools
pip install setuptools==69.5.1

# Install PyTorch
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Install detectron2 from source
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
git checkout v0.6
pip install --no-build-isolation .
cd ..
rm -rf detectron2

# Install other requirements
pip install -r requirements.txt
