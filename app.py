#!/usr/bin/env python3
"""
Hugging Face Spaces entry point for OneFormer application
"""

import os
import sys
import subprocess

# Set up environment variables for HF Spaces
os.environ['CUDA_HOME'] = '/usr/local/cuda' if os.path.exists('/usr/local/cuda') else ''

# Install deformable attention ops if not already installed
def setup_deformable_attention():
    ops_dir = os.path.join(os.path.dirname(__file__), 'oneformer/modeling/pixel_decoder/ops')
    if os.path.exists(ops_dir):
        try:
            subprocess.run(['bash', 'deform_setup.sh'], check=True, cwd=os.path.dirname(__file__))
            print("Deformable attention ops installed successfully")
        except Exception as e:
            print(f"Warning: Could not install deformable attention ops: {e}")
            print("Continuing without custom CUDA kernels...")

# Run setup on first launch
if not os.path.exists('oneformer/modeling/pixel_decoder/ops/build'):
    setup_deformable_attention()

# Import and run the main gradio app
from gradio_test import demo

if __name__ == "__main__":
    # Launch with HF Spaces compatible settings
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Disabled on HF Spaces
        debug=False
    )