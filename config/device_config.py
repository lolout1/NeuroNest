"""Device configuration for NeuroNest application."""

import warnings
warnings.filterwarnings("ignore")

# Device configuration with fallbacks
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CPU_DEVICE = torch.device("cpu")
    torch.set_num_threads(4)
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch not available - using CPU fallback")
    DEVICE = "cpu"
    CPU_DEVICE = "cpu" 
    TORCH_AVAILABLE = False
