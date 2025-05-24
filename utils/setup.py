"""Setup and initialization utilities for NeuroNest"""

import os
import sys
import torch
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def setup_python_paths():
    """Setup Python paths for detectron2 and oneformer integration"""
    project_root = Path(__file__).parent.parent.absolute()
    
    # Add project root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Add detectron2 if it exists
    detectron2_path = project_root / "detectron2"
    if detectron2_path.exists() and str(detectron2_path) not in sys.path:
        sys.path.insert(1, str(detectron2_path))
    
    # Add oneformer paths
    oneformer_path = project_root / "oneformer"
    if oneformer_path.exists() and str(oneformer_path) not in sys.path:
        sys.path.append(str(oneformer_path))
    
    return project_root


def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def check_detectron2_comprehensive():
    """Comprehensive detectron2 health check"""
    logger.info("🔍 Checking detectron2 availability...")
    
    status = {
        'available': False,
        'version': 'unknown',
        'config_available': False,
        'model_zoo_available': False,
        'engine_available': False,
        'fully_functional': False
    }
    
    try:
        import detectron2
        status['available'] = True
        status['version'] = getattr(detectron2, '__version__', 'local_build')
        
        try:
            from detectron2.config import get_cfg
            status['config_available'] = True
        except:
            pass
        
        try:
            from detectron2 import model_zoo
            status['model_zoo_available'] = True
        except:
            pass
        
        try:
            from detectron2.engine import DefaultPredictor
            status['engine_available'] = True
        except:
            pass
        
        if status['config_available']:
            try:
                cfg = get_cfg()
                status['fully_functional'] = True
                logger.info("✅ Detectron2 fully functional")
            except:
                pass
        
        return status
        
    except ImportError:
        logger.error("❌ Detectron2 not available")
        return status


def setup_device():
    """Setup compute device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"✅ Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info("💻 Using CPU")
    
    return device


def validate_environment():
    """Validate all dependencies and environment"""
    issues = []
    
    # Check critical imports
    try:
        import cv2
        logger.info("✅ OpenCV available")
    except ImportError:
        issues.append("OpenCV (cv2) not installed")
    
    try:
        import numpy
        logger.info("✅ NumPy available")
    except ImportError:
        issues.append("NumPy not installed")
    
    try:
        import gradio
        logger.info("✅ Gradio available")
    except ImportError:
        issues.append("Gradio not installed")
    
    # Check for detectron2
    try:
        import detectron2
        logger.info("✅ Detectron2 available")
    except ImportError:
        logger.warning("⚠️ Detectron2 not available - some features may be limited")
    
    # Check for natten (needed for OneFormer)
    try:
        import natten
        logger.info("✅ NATTEN available")
    except ImportError:
        logger.warning("⚠️ NATTEN not available - OneFormer may not work properly")
        issues.append("NATTEN not installed (required for OneFormer)")
    
    # Check model files
    project_root = Path(__file__).parent.parent
    model_paths = [
        "models/250_16_swin_l_oneformer_ade20k_160k.pth",
        "oneformer/250_16_swin_l_oneformer_ade20k_160k.pth"
    ]
    
    model_found = any((project_root / p).exists() for p in model_paths)
    if not model_found:
        logger.warning("⚠️ OneFormer model weights not found locally")
    
    return issues
