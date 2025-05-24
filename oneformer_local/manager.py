"""OneFormer model management for semantic segmentation with labeled visualization."""

import torch
import numpy as np
import cv2
import logging
import sys
import os
from typing import Tuple, Dict, Optional, List, Set

logger = logging.getLogger(__name__)

# Try to import detectron2 components with fallback
DETECTRON2_AVAILABLE = False
try:
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.engine.defaults import DefaultPredictor
    DETECTRON2_AVAILABLE = True
    logger.info("✅ Detectron2 imports successful")
except ImportError as e:
    logger.warning(f"⚠️ Detectron2 not available: {e}")
    # Define dummy classes for compatibility
    class DefaultPredictor:
        pass
    class MetadataCatalog:
        @staticmethod
        def get(name):
            return None

# OneFormer imports - use the correct import structure
ONEFORMER_AVAILABLE = False
try:
    # Try different import strategies
    try:
        # First try direct import
        from oneformer.config import (
            add_oneformer_config,
            add_common_config,
            add_swin_config,
            add_dinat_config,
        )
        ONEFORMER_AVAILABLE = True
    except ImportError:
        # Try from parent directory
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from oneformer.config import (
            add_oneformer_config,
            add_common_config,
            add_swin_config,
            add_dinat_config,
        )
        ONEFORMER_AVAILABLE = True
    
    from oneformer.oneformer_model import OneFormer
    logger.info("✅ OneFormer imports successful")
    
except ImportError as e:
    logger.warning(f"⚠️ OneFormer not available: {e}")
    ONEFORMER_AVAILABLE = False

from config import DEVICE, ONEFORMER_CONFIG, FLOOR_CLASSES

logger = logging.getLogger(__name__)


class OneFormerPredictor(DefaultPredictor if DETECTRON2_AVAILABLE else object):
    """Custom predictor for OneFormer that handles task conditioning"""
    
    def __init__(self, cfg):
        if DETECTRON2_AVAILABLE:
            super().__init__(cfg)
        else:
            self.cfg = cfg
            logger.warning("Running without detectron2 - limited functionality")
        
    def __call__(self, original_image, task_type="semantic"):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            task_type (str): the task to perform. One of "semantic", "instance", or "panoptic".

        Returns:
            predictions (dict): the output of the model
        """
        if not DETECTRON2_AVAILABLE:
            # Return dummy predictions
            h, w = original_image.shape[:2]
            return {
                "sem_seg": torch.zeros((150, h, w)),
                "instances": None
            }
            
        with torch.no_grad():
            # Apply pre-processing to image
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            
            # Create inputs dict with task type
            inputs = [{
                "image": image, 
                "height": height, 
                "width": width,
                "task": task_type
            }]
            
            predictions = self.model(inputs)[0]
            return predictions


class OneFormerManager:
    """Manages OneFormer model loading and inference with enhanced visualization"""

    def __init__(self):
        self.predictor = None
        self.metadata = None
        self.initialized = False
        self.use_high_res = False
        self.ade20k_classes = None
        
        # Floor-related class IDs from ADE20K
        self.floor_class_ids = set([3, 4, 28, 29, 52, 53, 54, 78, 91])
        
    def initialize(self, backbone: str = "swin", use_high_res: bool = False):
        """Initialize OneFormer model with optional high resolution"""
        if not ONEFORMER_AVAILABLE or not DETECTRON2_AVAILABLE:
            logger.error("OneFormer or Detectron2 not available - cannot initialize")
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
            
            # Try to find config file
            config_paths = [
                config["swin_cfg"],
                os.path.join("configs", "ade20k", "oneformer_swin_large_IN21k_384_bs16_160k.yaml"),
                os.path.join(os.path.dirname(__file__), "..", "configs", "ade20k", "oneformer_swin_large_IN21k_384_bs16_160k.yaml")
            ]
            
            config_found = False
            for config_path in config_paths:
                if os.path.exists(config_path):
                    cfg.merge_from_file(config_path)
                    config_found = True
                    logger.info(f"Using config: {config_path}")
                    break
            
            if not config_found:
                logger.warning("Config file not found, using defaults")
            
            cfg.MODEL.DEVICE = DEVICE

            # Try multiple paths for the model
            model_paths_to_try = [
                os.path.join(os.path.dirname(__file__), "..", "models", config["swin_file"]),
                os.path.expanduser(f"~/.cache/huggingface/hub/{config['swin_file']}"),
                "/home/user/.cache/huggingface/hub/models--shi-labs--oneformer_ade20k_swin_large/snapshots/4a5bac8e64f82681a12db2e151a4c2f4ce6092b2/250_16_swin_l_oneformer_ade20k_160k.pth",
                config["swin_file"]
            ]
            
            model_found = False
            for model_path in model_paths_to_try:
                if model_path and os.path.exists(model_path):
                    cfg.MODEL.WEIGHTS = model_path
                    logger.info(f"Using local model: {model_path}")
                    model_found = True
                    break
            
            if not model_found:
                # Try to download from HuggingFace
                try:
                    from huggingface_hub import hf_hub_download
                    model_path = hf_hub_download(
                        repo_id=config["swin_model"],
                        filename=config["swin_file"],
                        cache_dir=os.path.expanduser("~/.cache/huggingface/hub")
                    )
                    cfg.MODEL.WEIGHTS = model_path
                    logger.info(f"Downloaded model from HuggingFace: {model_path}")
                except Exception as e:
                    logger.error(f"Failed to download model: {e}")
                    logger.warning("Proceeding without model weights - may fail later")
            
            # Update input size configuration for resolution
            if use_high_res:
                cfg.INPUT.MIN_SIZE_TEST = 1280
                cfg.INPUT.MAX_SIZE_TEST = 1280 * 4
                logger.info(f"Using high resolution: 1280x1280")
            else:
                cfg.INPUT.MIN_SIZE_TEST = 640
                cfg.INPUT.MAX_SIZE_TEST = 640 * 4
                logger.info(f"Using standard resolution: 640x640")
            
            cfg.freeze()

            # Use our custom predictor
            self.predictor = OneFormerPredictor(cfg)
            
            # Get metadata
            self.metadata = MetadataCatalog.get(
                cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
            )
            
            # Store class names
            if hasattr(self.metadata, 'stuff_classes'):
                self.ade20k_classes = self.metadata.stuff_classes
                logger.info(f"Using existing ADE20K class names ({len(self.ade20k_classes)} classes)")
            else:
                from config import ADE20K_CLASS_NAMES
                self.ade20k_classes = [ADE20K_CLASS_NAMES.get(i, f"class_{i}") for i in range(150)]
            
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
            # Return dummy results if not initialized
            h, w = image.shape[:2]
            dummy_mask = np.zeros((h, w), dtype=np.uint8)
            return dummy_mask, image, image

        try:
            # Get target dimensions based on resolution mode
            target_size = 1280 if self.use_high_res else 640

            # Store original dimensions
            h_orig, w_orig = image.shape[:2]
            
            # Prepare image for model (maintain aspect ratio)
            scale = min(target_size / w_orig, target_size / h_orig)
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
            
            # Resize maintaining aspect ratio
            image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Pad to square if needed
            if new_w != target_size or new_h != target_size:
                pad_w = target_size - new_w
                pad_h = target_size - new_h
                pad_left = pad_w // 2
                pad_top = pad_h // 2
                
                image_padded = cv2.copyMakeBorder(
                    image_resized, 
                    pad_top, pad_h - pad_top, 
                    pad_left, pad_w - pad_left,
                    cv2.BORDER_CONSTANT, 
                    value=[0, 0, 0]
                )
            else:
                image_padded = image_resized
                pad_left = 0
                pad_top = 0

            # Run prediction
            try:
                predictions = self.predictor(image_padded, "semantic")
            except TypeError:
                predictions = self.predictor(image_padded)
            
            # Extract semantic segmentation
            if "sem_seg" in predictions:
                seg_mask_padded = predictions["sem_seg"].argmax(dim=0).cpu().numpy()
            else:
                # Fallback to dummy mask
                seg_mask_padded = np.zeros((target_size, target_size), dtype=np.uint8)

            # Remove padding from segmentation mask
            if pad_left > 0 or pad_top > 0:
                seg_mask = seg_mask_padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
            else:
                seg_mask = seg_mask_padded

            # Resize segmentation mask back to original size
            seg_mask_original = cv2.resize(seg_mask.astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

            # Create visualizations
            vis_image = self._create_visualization(image, seg_mask_original, with_labels=False)
            labeled_image = self._create_visualization(image, seg_mask_original, with_labels=True)

            return seg_mask_original, vis_image, labeled_image

        except Exception as e:
            logger.error(f"Semantic segmentation failed: {e}")
            # Return original image as fallback
            return np.zeros(image.shape[:2], dtype=np.uint8), image, image

    def _create_visualization(self, image: np.ndarray, seg_mask: np.ndarray, with_labels: bool = False) -> np.ndarray:
        """Create visualization with optional labels"""
        if not DETECTRON2_AVAILABLE:
            return image
            
        try:
            visualizer = Visualizer(
                image[:, :, ::-1],  # Convert RGB to BGR for visualization
                metadata=self.metadata,
                instance_mode=ColorMode.IMAGE
            )
            vis_output = visualizer.draw_sem_seg(seg_mask, alpha=0.5)
            vis_image = vis_output.get_image()[:, :, ::-1]  # Convert back to RGB

            if with_labels:
                vis_image = self._add_labels(vis_image, seg_mask)

            return vis_image
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return image

    def _add_labels(self, image: np.ndarray, seg_mask: np.ndarray) -> np.ndarray:
        """Add labels to visualization"""
        labeled_image = image.copy()
        unique_segments = np.unique(seg_mask)
        
        for seg_id in unique_segments:
            if seg_id == 255 or seg_id >= 150:  # Skip ignore label
                continue
                
            mask = seg_mask == seg_id
            if np.sum(mask) > 500:  # Only label large segments
                # Find centroid
                y_coords, x_coords = np.where(mask)
                if len(y_coords) > 0:
                    cy, cx = int(np.mean(y_coords)), int(np.mean(x_coords))
                    
                    # Get class name
                    if self.ade20k_classes and seg_id < len(self.ade20k_classes):
                        class_name = self.ade20k_classes[int(seg_id)].split(',')[0].strip()
                    else:
                        class_name = f"class_{seg_id}"
                    
                    # Draw label
                    cv2.putText(labeled_image, class_name, (cx-50, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(labeled_image, class_name, (cx-50, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        return labeled_image

    def extract_floor_areas(self, segmentation: np.ndarray) -> np.ndarray:
        """Extract floor areas from segmentation"""
        floor_mask = np.zeros_like(segmentation, dtype=bool)

        for seg_id in self.floor_class_ids:
            floor_mask |= (segmentation == seg_id)
        
        floor_pixels = np.sum(floor_mask)
        if floor_pixels > 0:
            logger.info(f"Floor extraction: Found {floor_pixels} floor pixels ({floor_pixels / segmentation.size * 100:.1f}%)")
        else:
            logger.warning("No floor pixels detected!")

        return floor_mask
