"""OneFormer model management for semantic segmentation with labeled visualization."""

import torch
import numpy as np
import cv2
import logging
import sys
import os
from typing import Tuple, Dict, Optional

# Add OneFormer path dynamically
ONEFORMER_PATH = "/home/sww35/OneFormerMentoria"
if ONEFORMER_PATH not in sys.path:
    sys.path.insert(0, ONEFORMER_PATH)

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

# OneFormer imports
try:
    from oneformer import (
        add_oneformer_config,
        add_common_config,
        add_swin_config,
        add_dinat_config,
    )
    from demo.defaults import DefaultPredictor as OneFormerPredictor
    ONEFORMER_AVAILABLE = True
except ImportError as e:
    print(f"OneFormer not available: {e}")
    ONEFORMER_AVAILABLE = False

from config import DEVICE, ONEFORMER_CONFIG, FLOOR_CLASSES, ADE20K_CLASS_NAMES

logger = logging.getLogger(__name__)


class OneFormerManager:
    """Manages OneFormer model loading and inference with enhanced visualization"""

    def __init__(self):
        self.predictor = None
        self.metadata = None
        self.initialized = False
        self.use_high_res = False  # Flag for resolution mode

    def initialize(self, backbone: str = "swin", use_high_res: bool = False):
        """Initialize OneFormer model with optional high resolution"""
        if not ONEFORMER_AVAILABLE:
            logger.error("OneFormer not available")
            return False

        try:
            self.use_high_res = use_high_res
            
            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_common_config(cfg)
            add_swin_config(cfg)
            add_oneformer_config(cfg)
            add_dinat_config(cfg)

            config = ONEFORMER_CONFIG["ADE20K"]
            cfg.merge_from_file(config["swin_cfg"])
            cfg.MODEL.DEVICE = DEVICE

            # Try local model first, then download
            local_model_path = os.path.join(os.path.dirname(__file__), "..", "models", config["swin_file"])
            
            if os.path.exists(local_model_path):
                cfg.MODEL.WEIGHTS = local_model_path
                logger.info(f"Using local model: {local_model_path}")
            else:
                # Download model from HuggingFace without authentication
                try:
                    from huggingface_hub import hf_hub_download
                    model_path = hf_hub_download(
                        repo_id=config["swin_model"],
                        filename=config["swin_file"],
                        token=None  # No authentication required
                    )
                    cfg.MODEL.WEIGHTS = model_path
                    logger.info(f"Downloaded model from HuggingFace: {model_path}")
                except Exception as e:
                    logger.error(f"Failed to download model: {e}")
                    # Try alternative path
                    alternative_path = f"/home/sww35/.cache/huggingface/hub/{config['swin_file']}"
                    if os.path.exists(alternative_path):
                        cfg.MODEL.WEIGHTS = alternative_path
                        logger.info(f"Using cached model: {alternative_path}")
                    else:
                        raise RuntimeError("Model file not found")
            
            # Update input size configuration for higher resolution
            if use_high_res:
                cfg.INPUT.MIN_SIZE_TEST = config.get("high_res_width", 1280)
                cfg.INPUT.MAX_SIZE_TEST = config.get("high_res_width", 1280) * 4
                logger.info(f"Using high resolution: {config.get('high_res_width', 1280)}x{config.get('high_res_height', 1280)}")
            else:
                cfg.INPUT.MIN_SIZE_TEST = config["width"]
                cfg.INPUT.MAX_SIZE_TEST = config["width"] * 4
                logger.info(f"Using standard resolution: {config['width']}x{config.get('height', config['width'])}")
            
            cfg.freeze()

            self.predictor = OneFormerPredictor(cfg)
            self.metadata = MetadataCatalog.get(
                cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
            )
            
            # Add class names to metadata for labeling
            self.metadata.stuff_classes = [ADE20K_CLASS_NAMES.get(i, f"class_{i}") for i in range(150)]
            
            self.initialized = True
            logger.info("OneFormer initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize OneFormer: {e}")
            import traceback
            traceback.print_exc()
            return False

    def semantic_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform semantic segmentation with labeled visualization"""
        if not self.initialized:
            raise RuntimeError("OneFormer not initialized")

        # Get target dimensions based on resolution mode
        config = ONEFORMER_CONFIG["ADE20K"]
        if self.use_high_res:
            target_width = config.get("high_res_width", 1280)
            target_height = config.get("high_res_height", 1280)
        else:
            target_width = config["width"]
            target_height = config.get("height", config["width"])

        # Resize image to target dimensions
        h, w = image.shape[:2]
        if w != target_width or h != target_height:
            # Calculate scale to maintain aspect ratio
            scale = min(target_width / w, target_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Pad to exact target size if needed
            if new_w < target_width or new_h < target_height:
                pad_w = target_width - new_w
                pad_h = target_height - new_h
                pad_left = pad_w // 2
                pad_top = pad_h // 2
                image_resized = cv2.copyMakeBorder(
                    image_resized, 
                    pad_top, pad_h - pad_top, 
                    pad_left, pad_w - pad_left,
                    cv2.BORDER_CONSTANT, 
                    value=[0, 0, 0]
                )
        else:
            image_resized = image

        # Run prediction
        predictions = self.predictor(image_resized, "semantic")
        seg_mask = predictions["sem_seg"].argmax(dim=0).cpu().numpy()

        # Create standard visualization
        visualizer = Visualizer(
            image_resized[:, :, ::-1],
            metadata=self.metadata,
            instance_mode=ColorMode.IMAGE
        )
        vis_output = visualizer.draw_sem_seg(seg_mask, alpha=0.5)
        vis_image = vis_output.get_image()[:, :, ::-1]  # BGR to RGB

        # Create labeled visualization
        labeled_image = self.create_labeled_visualization(image_resized, seg_mask)

        return seg_mask, vis_image, labeled_image

    def create_labeled_visualization(self, image: np.ndarray, seg_mask: np.ndarray) -> np.ndarray:
        """Create a visualization with object labels"""
        # Create overlay image
        overlay = image.copy()
        labeled_image = image.copy()
        
        # Get unique segments
        unique_segments = np.unique(seg_mask)
        
        # Define a color palette for visualization
        np.random.seed(42)  # For consistent colors
        colors = np.random.randint(0, 255, size=(150, 3))
        
        # Process each segment
        for seg_id in unique_segments:
            if seg_id == 255:  # Skip ignore label
                continue
                
            # Get segment mask
            mask = seg_mask == seg_id
            
            # Apply color overlay
            overlay[mask] = colors[seg_id % 150]
            
            # Get segment properties for labeling
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Label significant segments
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area for labeling
                    # Get centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Get class name
                        class_name = ADE20K_CLASS_NAMES.get(seg_id, f"class_{seg_id}")
                        
                        # Draw label with background
                        label = class_name
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 1
                        
                        # Get text size
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, font, font_scale, thickness
                        )
                        
                        # Draw background rectangle
                        cv2.rectangle(
                            labeled_image,
                            (cx - text_width // 2 - 2, cy - text_height - 2),
                            (cx + text_width // 2 + 2, cy + 2),
                            (0, 0, 0),
                            -1
                        )
                        
                        # Draw text
                        cv2.putText(
                            labeled_image,
                            label,
                            (cx - text_width // 2, cy),
                            font,
                            font_scale,
                            (255, 255, 255),
                            thickness
                        )
        
        # Blend overlay with original
        labeled_image = cv2.addWeighted(labeled_image, 0.7, overlay, 0.3, 0)
        
        return labeled_image

    def extract_floor_areas(self, segmentation: np.ndarray) -> np.ndarray:
        """Extract floor areas from segmentation"""
        floor_mask = np.zeros_like(segmentation, dtype=bool)

        for class_ids in FLOOR_CLASSES.values():
            for class_id in class_ids:
                floor_mask |= (segmentation == class_id)

        return floor_mask
