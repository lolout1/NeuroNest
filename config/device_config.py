"""Device configuration for NeuroNest application with robust CPU fallback."""

import torch
import warnings
import logging
import os

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def get_optimal_device():
    """Get the optimal device with robust fallback to CPU."""
    
    # Force CPU if environment variable is set
    if os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes"):
        logger.info("FORCE_CPU is set, using CPU")
        return torch.device("cpu")
    
    # Check for CUDA availability
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"CUDA available: Using GPU {gpu_name} ({gpu_memory:.1f}GB)")
            return device
        else:
            logger.info("CUDA not available or no GPUs found, using CPU")
    except Exception as e:
        logger.warning(f"Error checking CUDA availability: {e}, falling back to CPU")
    
    # Fallback to CPU
    logger.info("Using CPU device")
    return torch.device("cpu")

def configure_torch_for_device(device):
    """Configure PyTorch settings based on device."""
    if device.type == "cpu":
        # Optimize for CPU
        torch.set_num_threads(min(4, torch.get_num_threads()))
        logger.info(f"Configured PyTorch for CPU with {torch.get_num_threads()} threads")
    else:
        # GPU configuration
        torch.backends.cudnn.benchmark = True
        logger.info("Configured PyTorch for GPU with cuDNN optimization")

# Initialize device configuration
DEVICE = get_optimal_device()
CPU_DEVICE = torch.device("cpu")

# Configure PyTorch
configure_torch_for_device(DEVICE)

# Additional CPU-optimized settings
if DEVICE.type == "cpu":
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
