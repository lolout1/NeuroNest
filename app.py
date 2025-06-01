#!/usr/bin/env python3
"""
Minimal OneFormer Demo for HuggingFace Spaces
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import gradio as gr
import torch
import numpy as np
from PIL import Image

# Force CPU
device = torch.device("cpu")

def process_image(image):
    """Simple image processing function"""
    if image is None:
        return None
    
    # For now, just return the image with a message
    # Replace this with actual OneFormer inference
    return image

# Create simple interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="OneFormer Demo",
    description="OneFormer: Universal Image Segmentation (CPU Mode)",
)

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"Running on: CPU")
    iface.launch(server_name="0.0.0.0", server_port=7860)
