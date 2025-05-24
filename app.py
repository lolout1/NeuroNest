"""
NeuroNest: Advanced Environment Analysis for Alzheimer's Care
Main entry point for the application.
"""

import logging
import sys
import warnings
from pathlib import Path
import os

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: Ensure project root is FIRST in path
project_root = Path(__file__).parent.absolute()
sys.path = [p for p in sys.path if not p.endswith('NeuroNest')]
sys.path.insert(0, str(project_root))

# Add local oneformer to path (now in NeuroNest/oneformer)
local_oneformer = project_root / "oneformer"
if local_oneformer.exists():
    sys.path.insert(1, str(project_root))  # This allows "import oneformer" to work
    logger.info(f"Using local OneFormer from: {local_oneformer}")

# Load .env file if it exists
env_path = project_root / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value.strip()
    logger.info("Loaded environment from .env file")

# Import local modules
try:
    from config import DEVICE
    from interface import create_gradio_interface
    logger.info("✅ All modules imported successfully")
except ImportError as e:
    logger.error(f"Failed to import local modules: {e}")
    logger.error(f"Python path: {sys.path}")
    raise

def check_oneformer_available():
    """Check if OneFormer is available"""
    try:
        from oneformer import add_oneformer_config
        logger.info("✓ OneFormer is available")
        return True
    except ImportError as e:
        logger.warning(f"✗ OneFormer not available: {e}")
        return False

def check_natten_available():
    """Check if NATTEN is available"""
    try:
        import natten
        logger.info("✓ NATTEN is available")
        return True
    except ImportError as e:
        logger.warning(f"✗ NATTEN not available: {e}")
        return False

if __name__ == "__main__":
    print(f"🚀 Starting NeuroNest on {DEVICE}")
    print(f"📍 Project root: {project_root}")
    
    # Check dependencies
    ONEFORMER_AVAILABLE = check_oneformer_available()
    NATTEN_AVAILABLE = check_natten_available()
    
    print(f"📦 OneFormer available: {ONEFORMER_AVAILABLE}")
    print(f"📦 NATTEN available: {NATTEN_AVAILABLE}")
    
    if not ONEFORMER_AVAILABLE:
        print("\n⚠️  OneFormer not found!")
        print("Expected location: NeuroNest/oneformer/")

    try:
        # Create and launch interface
        interface = create_gradio_interface()

        # Launch the interface
        interface.queue(max_size=10).launch(
            server_name=os.environ.get('GRADIO_SERVER_NAME', '0.0.0.0'),
            server_port=int(os.environ.get('GRADIO_SERVER_PORT', 7860)),
            share=os.environ.get('GRADIO_SHARE', 'True').lower() == 'true',
            share_server_address=None,
            auth=None,
            show_api=False,
            show_error=True
        )
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        import traceback
        traceback.print_exc()
        raise
