#!/usr/bin/env python3
"""
NeuroNest Application Entry Point
Handles initialization and graceful startup for Hugging Face Spaces
"""

import os
import sys
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['FORCE_CUDA'] = '0'

def setup_oneformer_imports():
    """Add OneFormer to Python path if needed"""
    oneformer_path = Path(__file__).parent / "oneformer"
    if oneformer_path.exists() and str(oneformer_path) not in sys.path:
        sys.path.insert(0, str(oneformer_path))
        logger.info(f"Added OneFormer to path: {oneformer_path}")

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        # Verify torch version
        if not torch.__version__.startswith('1.9'):
            logger.warning(f"Expected PyTorch 1.9.x, got {torch.__version__}")
        
        import detectron2
        logger.info(f"Detectron2 version: {detectron2.__version__}")
        
        import gradio as gr
        logger.info(f"Gradio version: {gr.__version__}")
        
        import cv2
        logger.info(f"OpenCV version: {cv2.__version__}")
        
        import PIL
        logger.info(f"Pillow version: {PIL.__version__}")
        
        # Check PIL compatibility
        if hasattr(PIL.Image, 'LINEAR'):
            logger.info("PIL has LINEAR attribute")
        elif hasattr(PIL.Image, 'BILINEAR'):
            logger.info("PIL has BILINEAR attribute (newer version)")
            # Monkey patch for compatibility
            PIL.Image.LINEAR = PIL.Image.BILINEAR
            logger.info("Applied PIL compatibility patch")
        
        # Check numpy version
        import numpy as np
        logger.info(f"NumPy version: {np.__version__}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def main():
    """Main application entry point"""
    print("=" * 50)
    print(f"NeuroNest Application Startup")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Setup paths
    setup_oneformer_imports()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed")
        sys.exit(1)
    
    try:
        # Import and launch the Gradio interface
        from gradio_test import create_gradio_interface
        
        logger.info("Creating Gradio interface...")
        interface = create_gradio_interface()
        
        logger.info("Launching application...")
        interface.queue(max_size=10).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            debug = True
        )
        
    except Exception as e:
        logger.error(f"Error launching app: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
