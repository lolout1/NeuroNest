"""Device configuration for NeuroNest application."""

import torch
import warnings
warnings.filterwarnings("ignore")

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU_DEVICE = torch.device("cpu")
torch.set_num_threads(4)
