#!/usr/bin/env python3
"""
Pre-installation setup for Hugging Face Spaces
Run this before main app to ensure detectron2 is properly installed
"""

import subprocess
import sys
import os

# Downgrade setuptools first
subprocess.run([sys.executable, "-m", "pip", "install", "setuptools==69.5.1"], check=True)

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['FORCE_CUDA'] = '0'

# Install torch first
subprocess.run([
    sys.executable, "-m", "pip", "install", 
    "torch==2.0.1+cpu", "torchvision==0.15.2+cpu",
    "--index-url", "https://download.pytorch.org/whl/cpu"
], check=True)

# Install detectron2 dependencies
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "numpy==1.24.3", "pillow==10.0.0", "pycocotools", "opencv-python"
], check=True)

# Clone and install detectron2
import tempfile
import shutil

with tempfile.TemporaryDirectory() as tmpdir:
    # Clone repo
    subprocess.run([
        "git", "clone", "https://github.com/facebookresearch/detectron2.git",
        os.path.join(tmpdir, "detectron2")
    ], check=True)
    
    # Checkout stable version
    os.chdir(os.path.join(tmpdir, "detectron2"))
    subprocess.run(["git", "checkout", "v0.6"], check=True)
    
    # Install without build isolation
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--no-build-isolation", "."
    ], check=True)

print("Pre-installation complete!")
