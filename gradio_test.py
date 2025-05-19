"""
NeuroNest: Advanced Environment Analysis for Alzheimer's Care
Main entry point for the application.
"""

import logging
import sys
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to Python path if needed
from pathlib import Path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import application components
from config import DEVICE
from interface import create_gradio_interface

# Check OneFormer availability
try:
    from oneformer import add_oneformer_config
    ONEFORMER_AVAILABLE = True
except ImportError as e:
    print(f"OneFormer not available: {e}")
    ONEFORMER_AVAILABLE = False


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
