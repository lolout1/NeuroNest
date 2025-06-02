#!/usr/bin/env python3
"""
NeuroNest Hugging Face Spaces App with Robust Detectron2 Support
"""

import os
import sys
import subprocess

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['FORCE_CUDA'] = '0'

def ensure_detectron2():
    """Ensure detectron2 is installed with multiple fallback methods"""
    try:
        import detectron2
        return True
    except ImportError:
        print("Detectron2 not found, attempting installation...")
    
    # Method 1: Install from GitHub with no build isolation
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "--no-build-isolation",
            "git+https://github.com/facebookresearch/detectron2.git@v0.6"
        ], check=True)
        import detectron2
        return True
    except:
        pass
    
    # Method 2: Clone and build
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run([
                "git", "clone", "--depth", "1", "--branch", "v0.6",
                "https://github.com/facebookresearch/detectron2.git",
                f"{tmpdir}/detectron2"
            ], check=True)
            
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "--no-build-isolation", f"{tmpdir}/detectron2"
            ], check=True)
        
        import detectron2
        return True
    except:
        return False

def main():
    # Ensure detectron2 is available
    if not ensure_detectron2():
        print("WARNING: Could not install detectron2, running in limited mode")
        
        # Run minimal version without detectron2
        import gradio as gr
        
        def minimal_interface(image):
            return "Detectron2 not available. Please check deployment logs."
        
        demo = gr.Interface(
            fn=minimal_interface,
            inputs=gr.Image(type="filepath"),
            outputs="text",
            title="NeuroNest - Limited Mode"
        )
        demo.launch(server_name="0.0.0.0", server_port=7860)
        return
    
    # Normal operation with detectron2
    try:
        from gradio_test import create_gradio_interface
        
        print("Starting NeuroNest application...")
        interface = create_gradio_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
