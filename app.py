"""
NeuroNest: Advanced Environment Analysis for Alzheimer's Care
Main entry point - minimal launcher
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, '.')

# Import setup utilities
from utils.setup import setup_python_paths, setup_logging, validate_environment
from interface.gradio_ui import create_interface


def main():
    """Main application entry point"""
    try:
        # Setup environment
        project_root = setup_python_paths()
        logger = setup_logging()
        
        logger.info("🚀 Starting NeuroNest Alzheimer's Environment Analysis System")
        
        # Validate environment
        issues = validate_environment()
        if issues:
            logger.warning(f"Environment issues detected: {issues}")
        
        # Create and launch interface
        interface = create_interface()
        
        logger.info("🌐 Launching NeuroNest Interface...")
        interface.queue(
            default_concurrency_limit=2,
            max_size=10
        ).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
        
        return True
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"💥 Fatal error: {e}")
        else:
            print(f"💥 Fatal error: {e}")
        
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
