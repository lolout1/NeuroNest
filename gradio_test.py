"""Backward-compatible facade — imports from neuronest package."""

import torch
torch.set_num_threads(4)

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.INFO)

from neuronest.config import (
    DEVICE, CPU_DEVICE, ENABLE_QUANTIZATION, FLOOR_CLASSES,
    BLACKSPOT_MODEL_REPO, BLACKSPOT_MODEL_FILE,
    DISPLAY_MAX_WIDTH, DISPLAY_MAX_HEIGHT,
)
from neuronest.models import EoMTSegmenter, ImprovedBlackspotDetector, quantize_model_int8
from neuronest.utils import resize_for_processing as resize_image_for_processing
from neuronest.utils import resize_mask_to_original, prepare_display_image
from neuronest.pipeline import NeuroNestApp
from neuronest.ui import create_gradio_interface

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    print(f"Starting NeuroNest on {DEVICE}")
    print("EoMT-DINOv3 segmentation engine")
    try:
        interface = create_gradio_interface()
        interface.queue(max_size=10, default_concurrency_limit=1).launch(
            server_name="0.0.0.0", server_port=7860, share=False,
        )
    except Exception as e:
        logger.error(f"Failed to launch: {e}")
        raise
