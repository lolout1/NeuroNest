"""
NeuroNest: Advanced Environment Analysis for Alzheimer's Care
Robust main entry point with dependency management.
"""

import subprocess
import sys
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_git_dependencies():
    """Install git-based dependencies after torch is available"""
    git_deps = [
        "git+https://github.com/facebookresearch/detectron2.git",
        "git+https://github.com/SHI-Labs/OneFormer.git"
    ]
    
    for dep in git_deps:
        try:
            logger.info(f"Installing {dep}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--no-cache-dir", dep
            ])
            logger.info(f"✓ Successfully installed {dep}")
        except Exception as e:
            logger.error(f"Failed to install {dep}: {e}")
            return False
    return True

def check_torch_available():
    """Check if torch is available"""
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__} available")
        return True
    except ImportError:
        logger.error("✗ PyTorch not available")
        return False

def setup_oneformer_local():
    """Setup local OneFormer if available"""
    project_root = Path(__file__).parent
    oneformer_path = project_root / "oneformer"
    
    if oneformer_path.exists():
        sys.path.insert(0, str(project_root))
        logger.info(f"✓ Using local OneFormer from {oneformer_path}")
        return True
    else:
        logger.warning("Local OneFormer not found")
        return False

def main():
    """Main application entry point"""
    logger.info("🚀 Starting NeuroNest application...")
    
    # Check if torch is available
    if not check_torch_available():
        logger.error("PyTorch is required but not available")
        return
    
    # Install git dependencies if not already installed
    try:
        import detectron2
        logger.info("✓ Detectron2 already available")
    except ImportError:
        logger.info("Installing Detectron2...")
        if not install_git_dependencies():
            logger.error("Failed to install git dependencies")
            return
    
    # Setup local OneFormer
    setup_oneformer_local()
    
    # Setup paths
    project_root = Path(__file__).parent.absolute()
    sys.path.insert(0, str(project_root))
    
    # Import application modules with error handling
    try:
        from config.device_config import DEVICE
        logger.info(f"✓ Using device: {DEVICE}")
    except ImportError as e:
        logger.warning(f"Config import failed: {e}")
        DEVICE = "cpu"
    
    try:
        from interface.gradio_ui import create_gradio_interface
        logger.info("✓ Interface module imported")
    except ImportError as e:
        logger.error(f"Interface import failed: {e}")
        return create_minimal_interface()
    
    # Create and launch interface
    try:
        interface = create_gradio_interface()
        
        # Launch with HuggingFace Spaces configuration
        interface.queue(max_size=10).launch(
            server_name="0.0.0.0",
            server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
            share=False,
            show_error=True,
            show_api=False,
            debug=False
        )
        
    except Exception as e:
        logger.error(f"Failed to launch interface: {e}")
        import traceback
        traceback.print_exc()
        return create_minimal_interface()

def create_minimal_interface():
    """Create minimal interface when full system isn't available"""
    import gradio as gr
    
    def analyze_placeholder(image):
        return "NeuroNest is initializing. Please wait and try again.", None
    
    with gr.Blocks(title="NeuroNest - Initializing") as interface:
        gr.Markdown("# 🧠 NeuroNest - Alzheimer's Environment Analysis")
        gr.Markdown("**System is starting up. Please wait...**")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Room Image", type="filepath")
                analyze_btn = gr.Button("Analyze Environment", variant="primary")
            
            with gr.Column():
                result = gr.Textbox(label="Analysis Status", lines=10)
        
        analyze_btn.click(analyze_placeholder, image_input, result)
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    )

if __name__ == "__main__":
    main()
