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
    class Visualizer:
        pass
    class ColorMode:
        IMAGE = 0

# OneFormer imports with comprehensive error handling
ONEFORMER_AVAILABLE = False

# First check if oneformer directory is valid
oneformer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'oneformer')
if os.path.exists(oneformer_path):
    # Check if __init__.py contains Git LFS content
    init_file = os.path.join(oneformer_path, '__init__.py')
    if os.path.exists(init_file):
        with open(init_file, 'r') as f:
            content = f.read()
            if 'git-lfs' in content:
                logger.warning("⚠️ OneFormer contains Git LFS pointers - skipping import")
            else:
                # Try to import OneFormer
                try:
                    sys.path.insert(0, os.path.dirname(oneformer_path))
                    from oneformer.config import (
                        add_oneformer_config,
                        add_common_config,
                        add_swin_config,
                        add_dinat_config,
                    )
                    from oneformer.oneformer_model import OneFormer
                    ONEFORMER_AVAILABLE = True
                    logger.info("✅ OneFormer imports successful")
                except ImportError as e:
                    logger.warning(f"⚠️ OneFormer import failed: {e}")
                except Exception as e:
                    logger.warning(f"⚠️ OneFormer import error: {e}")

# Define dummy functions if OneFormer is not available
if not ONEFORMER_AVAILABLE:
    def add_oneformer_config(cfg):
        pass
    def add_common_config(cfg):
        pass
    def add_swin_config(cfg):
        pass
    def add_dinat_config(cfg):
        pass

# Import config constants
try:
    from config import DEVICE, ONEFORMER_CONFIG, FLOOR_CLASSES, ADE20K_CLASS_NAMES
except ImportError:
    logger.warning("Config import failed, using defaults")
    DEVICE = "cpu"
    ONEFORMER_CONFIG = {
        "ADE20K": {
            "key": "ade20k",
            "swin_cfg": "configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml",
            "swin_model": "shi-labs/oneformer_ade20k_swin_large",
            "swin_file": "250_16_swin_l_oneformer_ade20k_160k.pth",
        }
    }
    FLOOR_CLASSES = {}
    ADE20K_CLASS_NAMES = {}


class OneFormerPredictor(DefaultPredictor if DETECTRON2_AVAILABLE else object):
    """Custom predictor for OneFormer that handles task conditioning"""
    
    def __init__(self, cfg):
        if DETECTRON2_AVAILABLE:
            try:
                super().__init__(cfg)
                self.initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize predictor: {e}")
                self.initialized = False
        else:
            self.cfg = cfg
            self.initialized = False
            logger.warning("Running without detectron2 - limited functionality")
        
    def __call__(self, original_image, task_type="semantic"):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            task_type (str): the task to perform. One of "semantic", "instance", or "panoptic".

        Returns:
            predictions (dict): the output of the model
        """
        if not self.initialized or not DETECTRON2_AVAILABLE:
            # Return dummy predictions
            h, w = original_image.shape[:2]
            return {
                "sem_seg": torch.zeros((150, h, w)),
                "instances": None
            }
            
        try:
            with torch.no_grad():
                # Apply pre-processing to image
                if hasattr(self, 'input_format') and self.input_format == "RGB":
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
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            h, w = original_image.shape[:2]
            return {
                "sem_seg": torch.zeros((150, h, w)),
                "instances": None
            }


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
        
        # Define basic ADE20K classes if not available
        self.default_ade20k_classes = {
            0: 'wall', 1: 'building', 2: 'sky', 3: 'floor', 4: 'tree',
            5: 'ceiling', 6: 'road', 7: 'bed', 8: 'window', 9: 'grass',
            10: 'cabinet', 11: 'sidewalk', 12: 'person', 13: 'earth', 14: 'door',
            15: 'table', 16: 'mountain', 17: 'plant', 18: 'curtain', 19: 'chair',
            20: 'car', 21: 'water', 22: 'painting', 23: 'sofa', 24: 'shelf',
            25: 'house', 26: 'sea', 27: 'mirror', 28: 'rug', 29: 'field',
            30: 'armchair'
        }
        
    def initialize(self, backbone: str = "swin", use_high_res: bool = False):
        """Initialize OneFormer model with optional high resolution"""
        if not DETECTRON2_AVAILABLE:
            logger.warning("Detectron2 not available - running in limited mode")
            self.initialized = True  # Allow limited functionality
            return True

        if not ONEFORMER_AVAILABLE:
            logger.warning("OneFormer not available - running in limited mode")
            self.initialized = True  # Allow limited functionality
            return True

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
                logger.warning("Model weights not found - proceeding without weights")
            
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
            if DETECTRON2_AVAILABLE:
                self.metadata = MetadataCatalog.get(
                    cfg.DATASETS.TEST_PANOPTIC[0] if hasattr(cfg.DATASETS, 'TEST_PANOPTIC') and len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
                )
                
                # Store class names
                if hasattr(self.metadata, 'stuff_classes'):
                    self.ade20k_classes = self.metadata.stuff_classes
                    logger.info(f"Using existing ADE20K class names ({len(self.ade20k_classes)} classes)")
            
            if self.ade20k_classes is None:
                self.ade20k_classes = [self.default_ade20k_classes.get(i, f"class_{i}") for i in range(150)]
            
            self.initialized = True
            logger.info("OneFormer manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize OneFormer: {e}")
            import traceback
            traceback.print_exc()
            # Still mark as initialized to allow limited functionality
            self.initialized = True
            return True

    def semantic_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform semantic segmentation with labeled visualization"""
        if not self.initialized:
            logger.warning("OneFormer not properly initialized")
            h, w = image.shape[:2]
            dummy_mask = np.zeros((h, w), dtype=np.uint8)
            return dummy_mask, image, image

        try:
            if self.predictor is None or not hasattr(self.predictor, 'initialized') or not self.predictor.initialized:
                logger.warning("Predictor not available - returning dummy results")
                h, w = image.shape[:2]
                dummy_mask = np.zeros((h, w), dtype=np.uint8)
                return dummy_mask, image, image

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
            predictions = self.predictor(image_padded, "semantic")
            
            # Extract semantic segmentation
            if "sem_seg" in predictions and hasattr(predictions["sem_seg"], 'argmax'):
                seg_mask_padded = predictions["sem_seg"].argmax(dim=0).cpu().numpy()
            else:
                # Create a simple segmentation based on color
                seg_mask_padded = self._simple_color_segmentation(image_padded)

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
            import traceback
            traceback.print_exc()
            # Return original image as fallback
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.uint8), image, image

    def _simple_color_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Simple color-based segmentation as fallback"""
        h, w = image.shape[:2]
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Simple floor detection (brown/beige colors)
        lower_floor = np.array([10, 20, 20])
        upper_floor = np.array([25, 255, 200])
        floor_mask = cv2.inRange(hsv, lower_floor, upper_floor)
        seg_mask[floor_mask > 0] = 3  # Floor class
        
        # Simple wall detection (light colors)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        wall_mask = gray > 200
        seg_mask[wall_mask] = 0  # Wall class
        
        return seg_mask

    def _create_visualization(self, image: np.ndarray, seg_mask: np.ndarray, with_labels: bool = False) -> np.ndarray:
        """Create visualization with optional labels"""
        if DETECTRON2_AVAILABLE and self.metadata is not None:
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
        
        # Fallback visualization
        vis_image = image.copy()
        
        # Apply simple color overlay
        unique_classes = np.unique(seg_mask)
        colors = np.random.randint(0, 255, size=(len(unique_classes), 3))
        
        for i, class_id in enumerate(unique_classes):
            if class_id == 255:  # Skip ignore class
                continue
            mask = seg_mask == class_id
            vis_image[mask] = vis_image[mask] * 0.5 + colors[i] * 0.5
        
        if with_labels:
            vis_image = self._add_labels(vis_image, seg_mask)
            
        return vis_image.astype(np.uint8)

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
                        if isinstance(self.ade20k_classes[int(seg_id)], str):
                            class_name = self.ade20k_classes[int(seg_id)].split(',')[0].strip()
                        else:
                            class_name = str(self.ade20k_classes[int(seg_id)])
                    else:
                        class_name = self.default_ade20k_classes.get(int(seg_id), f"class_{seg_id}")
                    
                    # Draw label with background
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    
                    # Get text size
                    (text_width, text_height), _ = cv2.getTextSize(
                        class_name, font, font_scale, thickness
                    )
                    
                    # Draw background rectangle
                    cv2.rectangle(
                        labeled_image,
                        (cx - text_width // 2 - 5, cy - text_height - 5),
                        (cx + text_width // 2 + 5, cy + 5),
                        (0, 0, 0),
                        -1
                    )
                    
                    # Draw text
                    cv2.putText(
                        labeled_image, 
                        class_name, 
                        (cx - text_width // 2, cy), 
                        font, 
                        font_scale, 
                        (255, 255, 255), 
                        thickness
                    )
        
        return labeled_image

    def extract_floor_areas(self, segmentation: np.ndarray) -> np.ndarray:
        """Extract floor areas from segmentation"""
        floor_mask = np.zeros_like(segmentation, dtype=bool)

        for seg_id in self.floor_class_ids:
            floor_mask |= (segmentation == seg_id)
        
        # If no floor detected, assume bottom 30% is floor
        if not np.any(floor_mask):
            h = segmentation.shape[0]
            floor_mask[int(h * 0.7):, :] = True
            logger.warning("No floor segments detected - using bottom 30% as floor")
        
        floor_pixels = np.sum(floor_mask)
        if floor_pixels > 0:
            logger.info(f"Floor extraction: Found {floor_pixels} floor pixels ({floor_pixels / segmentation.size * 100:.1f}%)")

        return floor_mask
