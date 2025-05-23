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

# Import local modules BEFORE adding OneFormer path
from config import DEVICE
from interface import create_gradio_interface

# Now check OneFormer availability without permanent path changes
def check_oneformer_available():
    try:
        import sys
        oneformer_path = "/mmfs1/home/sww35/OneFormerMentoria"
        original_path = sys.path.copy()
        
        if oneformer_path not in sys.path:
            sys.path.insert(0, oneformer_path)
        
        from oneformer import add_oneformer_config
        result = True
        
        # Restore original path
        sys.path = original_path
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

        # Launch the interface
        interface.queue(max_size=10).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True
        )
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        raise
