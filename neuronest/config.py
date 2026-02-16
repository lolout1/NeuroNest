import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU_DEVICE = torch.device("cpu")

ENABLE_QUANTIZATION = os.environ.get("NEURONEST_QUANTIZE", "1") == "1"

FLOOR_CLASSES = {
    "floor": [3, 4, 13],
    "carpet": [28],
    "mat": [78],
}

FLOOR_CLASS_IDS = [3, 4, 13, 28, 78]

BLACKSPOT_MODEL_REPO = "lolout1/txstNeuroNest"
BLACKSPOT_MODEL_FILE = "model_0004999.pth"

DISPLAY_MAX_WIDTH = 1920
DISPLAY_MAX_HEIGHT = 1080

EOMT_MODEL_ID = "tue-mps/ade20k_semantic_eomt_large_512"
