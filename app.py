"""
NeuroNest: Advanced Environment Analysis for Alzheimer's Care
Main entry point - minimal launcher
"""

import os
import sys
import warnings
import logging

warnings.filterwarnings("ignore")

# CRITICAL: Setup Python paths BEFORE any local imports
def setup_python_paths():
    """Setup Python paths for detectron2 and oneformer integration"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Add project root
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Add detectron2 if it exists
    detectron2_path = os.path.join(project_root, "detectron2")
    if os.path.exists(detectron2_path) and detectron2_path not in sys.path:
        sys.path.insert(1, detectron2_path)
    
    # Add oneformer paths
    oneformer_path = os.path.join(project_root, "oneformer")
    if os.path.exists(oneformer_path) and oneformer_path not in sys.path:
        sys.path.append(oneformer_path)
    
    oneformer_local_path = os.path.join(project_root, "oneformer_local")
    if os.path.exists(oneformer_local_path) and oneformer_local_path not in sys.path:
        sys.path.append(oneformer_local_path)
    
    return project_root

# Setup paths FIRST
project_root = setup_python_paths()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# NOW import local modules after paths are set
try:
    from interface.gradio_ui import create_interface
    from utils.setup import validate_environment
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)


def main():
    """Main application entry point"""
    try:
        logger.info("🚀 Starting NeuroNest Alzheimer's Environment Analysis System")
        
        # Validate environment
        issues = validate_environment()
        if issues:
            logger.warning(f"Environment issues detected: {issues}")
        
        # Create and launch interface
        interface = create_interface()
        
        if interface is None:
            logger.error("Failed to create interface")
            return False
        
        logger.info("🌐 Launching NeuroNest Interface...")
        interface.queue(
            default_concurrency_limit=2,
            max_size=10
        ).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            prevent_thread_lock=False
        )
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        logger.error("💥 NeuroNest failed to start")
        sys.exit(1)
