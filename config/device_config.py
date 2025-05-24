"""Device configuration for NeuroNest application - CPU optimized for HuggingFace."""

import torch
import warnings
import os

warnings.filterwarnings("ignore")

# Force CPU mode for HuggingFace Spaces
DEVICE = "cpu" 
CPU_DEVICE = torch.device("cpu")

# Check torch availability
try:
    TORCH_AVAILABLE = True
    torch.set_num_threads(4)  # Optimize for CPU
    
    # Memory optimization
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    print(f"✅ PyTorch {torch.__version__} configured for CPU")
except Exception as e:
    TORCH_AVAILABLE = False
    print(f"❌ PyTorch configuration failed: {e}")

# Check detectron2 availability
try:
    import detectron2
    DETECTRON2_AVAILABLE = True
    print(f"✅ Detectron2 {detectron2.__version__} available")
except ImportError:
    DETECTRON2_AVAILABLE = False
    print("⚠️ Detectron2 not available - will use fallback")

# OneFormer availability
try:
    from oneformer_local import OneFormerManager
    ONEFORMER_LOCAL_AVAILABLE = True
    print("✅ OneFormer local manager available")
except ImportError:
    ONEFORMER_LOCAL_AVAILABLE = False
    print("⚠️ OneFormer local manager not available")
