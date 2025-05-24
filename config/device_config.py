"""Device configuration with robust error handling."""

import warnings
warnings.filterwarnings("ignore")

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CPU_DEVICE = torch.device("cpu")
    try:
        torch.set_num_threads(4)
    except:
        pass
    TORCH_AVAILABLE = True
    print(f"✓ PyTorch available - using {DEVICE}")
except ImportError:
    print("⚠️ PyTorch not found - using CPU fallback")
    DEVICE = "cpu"
    CPU_DEVICE = "cpu"
    TORCH_AVAILABLE = False
except Exception as e:
    print(f"⚠️ PyTorch error: {e} - using CPU fallback")
    DEVICE = "cpu"
    CPU_DEVICE = "cpu"
    TORCH_AVAILABLE = False
