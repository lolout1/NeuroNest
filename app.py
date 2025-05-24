"""
NeuroNest: Advanced Environment Analysis for Alzheimer's Care
Main entry point for the application.
"""

# Install critical dependencies first
import subprocess
import sys
import os

def ensure_dependencies():
    """Ensure critical dependencies are installed in correct order"""
    try:
        import torch
        import detectron2
        print("✓ Core dependencies already available")
        return True
    except ImportError:
        print("Installing core dependencies...")
        try:
            # Install torch first
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch==2.0.1+cpu", "torchvision==0.15.2+cpu",
                "--find-links", "https://download.pytorch.org/whl/cpu/torch_stable.html"
            ])
            
            # Install detectron2 dependencies
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "git+https://github.com/cocodataset/panopticapi.git",
                "git+https://github.com/mcordts/cityscapesScripts.git"
            ])
            
            # Install detectron2
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "git+https://github.com/facebookresearch/detectron2.git@v0.6"
            ])
            
            print("✓ Dependencies installed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False

# Ensure dependencies before importing main modules
if not ensure_dependencies():
    sys.exit(1)

import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rest of your existing app.py code...
project_root = Path(__file__).parent.absolute()
sys.path = [p for p in sys.path if not p.endswith('NeuroNest')]
sys.path.insert(0, str(project_root))

# Your existing imports and code...


def install_package(package_name, import_name=None):
    """Install a package if it's not already installed"""
    if import_name is None:
        import_name = package_name.split('==')[0].split('@')[0].replace('-', '_')
    
    try:
        importlib.import_module(import_name)
        logger.info(f"✓ {import_name} is already installed")
        return True
    except ImportError:
        logger.info(f"Installing {package_name}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--no-cache-dir", package_name
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"✓ Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to install {package_name}: {e}")
            return False

def ensure_dependencies():
    """Ensure all required dependencies are installed"""
    logger.info("Checking and installing dependencies...")
    
    # Critical dependencies that must be installed in order
    critical_deps = [
        ("torch==1.10.1", "torch"),
        ("torchvision==0.11.2", "torchvision"),
        ("git+https://github.com/facebookresearch/detectron2.git@v0.6", "detectron2"),
        ("git+https://github.com/SHI-Labs/OneFormer.git", "oneformer")
    ]
    
    for package, import_name in critical_deps:
        if not install_package(package, import_name):
            logger.error(f"Failed to install critical dependency: {package}")
            return False
    
    # Optional dependencies
    optional_deps = [
        ("git+https://github.com/SHI-Labs/NATTEN.git", "natten")
    ]
    
    for package, import_name in optional_deps:
        install_package(package, import_name)  # Don't fail if these don't work
    
    logger.info("Dependency check completed")
    return True

# Install dependencies at startup
if not ensure_dependencies():
    logger.error("Failed to install critical dependencies")
    sys.exit(1)

# CRITICAL: Ensure project root is FIRST in path
project_root = Path(__file__).parent.absolute()
sys.path = [p for p in sys.path if not p.endswith('NeuroNest')]
sys.path.insert(0, str(project_root))

# Add local oneformer to path
local_oneformer = project_root / "oneformer"
if local_oneformer.exists():
    sys.path.insert(1, str(project_root))
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

# Import local modules (after dependency installation)
try:
    from config import DEVICE
    from interface import create_gradio_interface
except ImportError as e:
    logger.error(f"Failed to import local modules: {e}")
    logger.error(f"Python path: {sys.path}")
    raise

def check_component_availability():
    """Check which components are available"""
    components = {}
    
    try:
        from oneformer import add_oneformer_config
        components['oneformer'] = True
        logger.info("✓ OneFormer is available")
    except ImportError as e:
        components['oneformer'] = False
        logger.warning(f"✗ OneFormer not available: {e}")
    
    try:
        import natten
        components['natten'] = True
        logger.info("✓ NATTEN is available")
    except ImportError as e:
        components['natten'] = False
        logger.warning(f"✗ NATTEN not available: {e}")
    
    try:
        import detectron2
        components['detectron2'] = True
        logger.info("✓ Detectron2 is available")
    except ImportError as e:
        components['detectron2'] = False
        logger.warning(f"✗ Detectron2 not available: {e}")
    
    return components

if __name__ == "__main__":
    print(f"🚀 Starting NeuroNest on {DEVICE}")
    print(f"📍 Project root: {project_root}")
    
    # Check component availability
    components = check_component_availability()
    
    if not components.get('detectron2', False):
        print("\n⚠️  Detectron2 not found!")
        print("Attempting to install at runtime...")
        install_package("git+https://github.com/facebookresearch/detectron2.git@v0.6", "detectron2")
    
    if not components.get('oneformer', False):
        print("\n⚠️  OneFormer not found!")
        print("Attempting to install at runtime...")
        install_package("git+https://github.com/SHI-Labs/OneFormer.git", "oneformer")

    try:
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
        
        # Try to install missing dependencies and retry
        logger.info("Attempting to install missing dependencies...")
        ensure_dependencies()
        
        try:
            interface = create_gradio_interface()
            interface.queue(max_size=10).launch(
                server_name='0.0.0.0',
                server_port=7860,
                share=True
            )
        except Exception as e2:
            logger.error(f"Failed to launch after dependency installation: {e2}")
            raise
