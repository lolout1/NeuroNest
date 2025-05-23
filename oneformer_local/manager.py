"""OneFormer model management for semantic segmentation."""

import torch
import numpy as np
import cv2
import logging
from typing import Tuple
from huggingface_hub import hf_hub_download

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

# OneFormer imports - handle path carefully to avoid conflicts
ONEFORMER_AVAILABLE = False
try:
    import sys
    # Only add OneFormer path temporarily for this import
    oneformer_path = "/mmfs1/home/sww35/OneFormerMentoria"
    if oneformer_path not in sys.path:
        sys.path.insert(0, oneformer_path)
    
    from oneformer import (
        add_oneformer_config,
        add_common_config,
        add_swin_config,
        add_dinat_config,
    )
    
    # Try demo predictor first, fallback to detectron2
    try:
        from demo.defaults import DefaultPredictor as OneFormerPredictor
    except ImportError:
        from detectron2.engine import DefaultPredictor as OneFormerPredictor
        print("Using detectron2 DefaultPredictor instead of demo predictor")
    
    ONEFORMER_AVAILABLE = True
    
    # Remove OneFormer path to avoid conflicts with local imports
    if oneformer_path in sys.path:
        sys.path.remove(oneformer_path)
        
except ImportError as e:
    print(f"OneFormer not available: {e}")
    ONEFORMER_AVAILABLE = False

from config import DEVICE, ONEFORMER_CONFIG, FLOOR_CLASSES

logger = logging.getLogger(__name__)


class OneFormerManager:
    """Manages OneFormer model loading and inference"""

    def __init__(self):
        self.predictor = None
        self.metadata = None
        self.initialized = False

    def initialize(self, backbone: str = "swin"):
        """Initialize OneFormer model"""
        if not ONEFORMER_AVAILABLE:
            logger.error("OneFormer not available")
            return False

        try:
            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_common_config(cfg)
            add_swin_config(cfg)
            add_oneformer_config(cfg)
            add_dinat_config(cfg)

            config = ONEFORMER_CONFIG["ADE20K"]
            cfg.merge_from_file(config["swin_cfg"])
            cfg.MODEL.DEVICE = DEVICE

            # Download model if not exists
            model_path = hf_hub_download(
                repo_id=config["swin_model"],
                filename=config["swin_file"]
            )
            cfg.MODEL.WEIGHTS = model_path
            cfg.freeze()

            self.predictor = OneFormerPredictor(cfg)
            self.metadata = MetadataCatalog.get(
                cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
            )
            self.initialized = True
            logger.info("OneFormer initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize OneFormer: {e}")
            return False

    def semantic_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform semantic segmentation"""
        if not self.initialized:
            raise RuntimeError("OneFormer not initialized")

        # Resize image to expected width
        width = ONEFORMER_CONFIG["ADE20K"]["width"]
        h, w = image.shape[:2]
        if w != width:
            scale = width / w
            new_h = int(h * scale)
            image_resized = cv2.resize(image, (width, new_h))
        else:
            image_resized = image

        # Run prediction
        predictions = self.predictor(image_resized, "semantic")
        seg_mask = predictions["sem_seg"].argmax(dim=0).cpu().numpy()

        # Create visualization
        visualizer = Visualizer(
            image_resized[:, :, ::-1],
            metadata=self.metadata,
            instance_mode=ColorMode.IMAGE
        )
        vis_output = visualizer.draw_sem_seg(seg_mask, alpha=0.5)
        vis_image = vis_output.get_image()[:, :, ::-1]  # BGR to RGB

        return seg_mask, vis_image

    def extract_floor_areas(self, segmentation: np.ndarray) -> np.ndarray:
        """Extract floor areas from segmentation"""
        floor_mask = np.zeros_like(segmentation, dtype=bool)

        for class_ids in FLOOR_CLASSES.values():
            for class_id in class_ids:
                floor_mask |= (segmentation == class_id)

        return floor_mask
