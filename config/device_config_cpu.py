"""
NeuroNest: Advanced Environment Analysis for Alzheimer's Care
Robust main entry point.
"""

import logging
import sys
import warnings
import os
import time
from pathlib import Path

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_dependencies():
    """Ensure all critical dependencies are available"""
    try:
        import torch
        import cv2
        import gradio as gr
        import numpy as np
        logger.info("✅ All core dependencies available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

def main():
    """Main application entry point with comprehensive error handling"""
    
    logger.info("🚀 Starting NeuroNest Application")
    
    # Check dependencies
    if not ensure_dependencies():
        logger.error("Critical dependencies missing - cannot start")
        return False
    
    # Setup paths
    project_root = Path(__file__).parent.absolute()
    sys.path.insert(0, str(project_root))
    
    # Add local oneformer to path
    oneformer_path = project_root / "oneformer"
    if oneformer_path.exists():
        sys.path.insert(0, str(oneformer_path))
        logger.info(f"✅ Local OneFormer path added: {oneformer_path}")
    
    try:
        # Import configuration
        from config.device_config import DEVICE, TORCH_AVAILABLE
        logger.info(f"✅ Configuration loaded - Device: {DEVICE}")
        
        # Import and create interface
        from interface.gradio_ui import create_gradio_interface
        logger.info("✅ Interface module imported")
        
        # Create interface
        interface = create_gradio_interface()
        logger.info("✅ Interface created successfully")
        
        # Launch with proper configuration
        logger.info("🌐 Launching Gradio interface...")
        
        interface.queue(
            default_concurrency_limit=2,
            max_size=10
        ).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            show_api=False,
            prevent_thread_lock=False,
            quiet=False
        )
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Application failed to start: {e}")
        import traceback
        traceback.print_exc()
        
        # Create minimal fallback interface
        try:
            create_fallback_interface()
        except Exception as fallback_error:
            logger.error(f"❌ Even fallback failed: {fallback_error}")
            return False

def create_fallback_interface():
    """Create minimal interface when main system fails"""
    import gradio as gr
    
    def status_check():
        return "🔧 NeuroNest is initializing. Please refresh in a moment."
    
    with gr.Blocks(title="NeuroNest - Loading") as interface:
        gr.Markdown("# 🧠 NeuroNest - Alzheimer's Environment Analysis")
        gr.Markdown("## System Status: Initializing...")
        
        status_btn = gr.Button("Check Status", variant="primary")
        status_output = gr.Textbox(label="Status", value=status_check())
        
        status_btn.click(status_check, outputs=status_output)
    
    logger.info("🔧 Launching fallback interface...")
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("✅ NeuroNest started successfully")
    else:
        logger.error("❌ NeuroNest failed to start")
        # Keep container running for debugging
        time.sleep(3600)
