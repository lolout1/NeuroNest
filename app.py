"""
NeuroNest: Advanced Environment Analysis for Alzheimer's Care
Main entry point with comprehensive error handling.
"""

import logging
import sys
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_and_install_dependencies():
    """Check if critical dependencies are available"""
    missing_deps = []
    
    try:
        import torch
        logger.info("✓ PyTorch available")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import cv2
        logger.info("✓ OpenCV available")
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import gradio
        logger.info("✓ Gradio available")
    except ImportError:
        missing_deps.append("gradio")
    
    if missing_deps:
        logger.error(f"Missing dependencies: {missing_deps}")
        logger.info("Installing missing dependencies...")
        
        import subprocess
        for dep in missing_deps:
            try:
                if dep == "torch":
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", 
                        "--extra-index-url", "https://download.pytorch.org/whl/cpu",
                        "torch==2.0.1+cpu", "torchvision==0.15.2+cpu"
                    ])
                else:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                logger.info(f"✓ Installed {dep}")
            except Exception as e:
                logger.error(f"Failed to install {dep}: {e}")
                return False
    
    return True

def main():
    """Main application entry point"""
    
    # Ensure dependencies are available
    if not check_and_install_dependencies():
        logger.error("Failed to install required dependencies")
        return
    
    # Setup paths
    project_root = Path(__file__).parent.absolute()
    sys.path.insert(0, str(project_root))
    
    # Import with error handling
    try:
        # Import config with fallback
        try:
            from config import DEVICE, TORCH_AVAILABLE
            logger.info(f"Using device: {DEVICE}")
        except ImportError as e:
            logger.warning(f"Config import failed: {e}, using fallbacks")
            DEVICE = "cpu"
            TORCH_AVAILABLE = False
        
        # Import interface
        from interface import create_gradio_interface
        logger.info("✓ Interface module imported")
        
    except ImportError as e:
        logger.error(f"Critical import failed: {e}")
        # Create a minimal fallback interface
        return create_fallback_interface()
    
    try:
        # Create and launch interface
        interface = create_gradio_interface()
        
        # Launch configuration
        launch_kwargs = {
            "server_name": "0.0.0.0",
            "server_port": int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
            "share": False,
            "show_error": True,
            "show_api": False
        }
        
        logger.info("🚀 Launching NeuroNest application...")
        interface.queue(max_size=10).launch(**launch_kwargs)
        
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        import traceback
        traceback.print_exc()
        # Try fallback interface
        create_fallback_interface()

def create_fallback_interface():
    """Create a minimal interface when full system isn't available"""
    import gradio as gr
    
    def fallback_analysis(image):
        return "⚠️ NeuroNest is initializing. Please try again in a moment.", None
    
    with gr.Blocks(title="NeuroNest - Initializing") as interface:
        gr.Markdown("# 🧠 NeuroNest - Alzheimer's Environment Analysis")
        gr.Markdown("**System is initializing... Please wait.**")
        
        with gr.Row():
            image_input = gr.Image(label="Upload Image", type="filepath")
            result_output = gr.Textbox(label="Status")
        
        analyze_btn = gr.Button("Analyze", variant="primary")
        analyze_btn.click(fallback_analysis, image_input, result_output)
    
    interface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
