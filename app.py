"""
NeuroNest: Advanced Environment Analysis for Alzheimer's Care
Main entry point for the application.
"""

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to Python path FIRST (before OneFormer)
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure local modules take precedence
# Remove any OneFormer path from sys.path temporarily
original_path = sys.path.copy()
sys.path = [p for p in sys.path if 'OneFormerMentoria' not in p]

# Import local modules
from config import DEVICE
from interface import create_gradio_interface

# Restore path for OneFormer checking
sys.path = original_path

# Now check OneFormer availability
def check_oneformer_available():
    try:
        import sys
        oneformer_path = "/home/sww35/OneFormerMentoria"
        
        if oneformer_path not in sys.path:
            sys.path.insert(0, oneformer_path)
        
        from oneformer import add_oneformer_config
        result = True
        
        return result
        
    except ImportError as e:
        print(f"✗ OneFormer not available: {e}")
        return False

ONEFORMER_AVAILABLE = check_oneformer_available()

if __name__ == "__main__":
    print(f"🚀 Starting NeuroNest on {DEVICE}")
    print(f"OneFormer available: {ONEFORMER_AVAILABLE}")

    try:
        interface = create_gradio_interface()

        # Launch the interface with public link
        # share=True creates a public link without requiring authentication
        interface.queue(max_size=10).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,  # This creates a public link
            share_server_address=None,  # Use Gradio's default sharing server
            auth=None,  # No authentication required
            show_api=False,  # Don't show API documentation
            show_error=True  # Show errors in the interface
        )
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        import traceback
        traceback.print_exc()
        raise
