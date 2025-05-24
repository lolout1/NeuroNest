"""OneFormer model management for semantic segmentation with git-lfs support."""

import torch
import numpy as np
import cv2
import logging
import sys
import os
from typing import Tuple, Dict, Optional, List, Set
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import detectron2 components
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

# OneFormer imports from git-lfs version
ONEFORMER_AVAILABLE = False
try:
    # For HuggingFace Spaces, oneformer is in the repo root
    project_root = Path(__file__).parent.parent
    oneformer_path = project_root / "oneformer"
    
    if oneformer_path.exists():
        sys.path.insert(0, str(project_root))
        logger.info(f"Using OneFormer from git-lfs: {oneformer_path}")
    
    from oneformer.config import (
        add_oneformer_config,
        add_common_config,
        add_swin_config,
        add_dinat_config,
    )
    from oneformer.oneformer_model import OneFormer
    ONEFORMER_AVAILABLE = True
    logger.info("✅ OneFormer imports successful from git-lfs")
    
except ImportError as e:
    logger.warning(f"⚠️ OneFormer not available: {e}")
    ONEFORMER_AVAILABLE = False

from config import DEVICE, ONEFORMER_CONFIG, FLOOR_CLASSES, ADE20K_CLASS_NAMES


class OneFormerPredictor(DefaultPredictor if DETECTRON2_AVAILABLE else object):
    """Custom predictor for OneFormer that handles task conditioning"""
    
    def __init__(self, cfg):
        if DETECTRON2_AVAILABLE:
            super().__init__(cfg)
        else:
            self.cfg = cfg
            logger.warning("Running without detectron2")
        
    def __call__(self, original_image, task_type="semantic"):
        if not DETECTRON2_AVAILABLE:
            h, w = original_image.shape[:2]
            return {
                "sem_seg": torch.zeros((150, h, w)),
                "instances": None
            }
            
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            
            inputs = [{
                "image": image, 
                "height": height, 
                "width": width,
                "task": task_type
            }]
            
            predictions = self.model(inputs)[0]
            return predictions


class OneFormerManager:
    """Manages OneFormer model with git-lfs support"""

    def __init__(self):
        self.predictor = None
        self.metadata = None
        self.initialized = False
        self.use_high_res = False
        self.ade20k_classes = None
        
        # Floor-related class IDs
        self.floor_class_ids = set([3, 4, 28, 29, 52, 53, 54, 78, 91])
        
    def initialize(self, backbone: str = "swin", use_high_res: bool = False):
        """Initialize OneFormer model from git-lfs location"""
        if not ONEFORMER_AVAILABLE or not DETECTRON2_AVAILABLE:
            logger.error("OneFormer or Detectron2 not available")
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
            
            # Find config file in git-lfs oneformer directory
            project_root = Path(__file__).parent.parent
            config_paths = [
                project_root / "oneformer" / "configs" / "ade20k" / "oneformer_swin_large_IN21k_384_bs16_160k.yaml",
                project_root / config["swin_cfg"],
                Path(config["swin_cfg"])
            ]
            
            config_found = False
            for config_path in config_paths:
                if config_path.exists():
                    cfg.merge_from_file(str(config_path))
                    config_found = True
                    logger.info(f"✅ Using config: {config_path}")
                    break
            
            if not config_found:
                logger.warning("Config file not found, using defaults")
            
            cfg.MODEL.DEVICE = DEVICE

            # Model weights paths - check git-lfs locations
            model_filename = config["swin_file"]
            model_paths = [
                # Git-lfs location in repo
                project_root / "oneformer" / "models" / model_filename,
                project_root / "models" / model_filename,
                # HuggingFace cache
                Path.home() / ".cache" / "huggingface" / "hub" / model_filename,
                # Direct path
                Path(model_filename)
            ]
            
            model_found = False
            for model_path in model_paths:
                if model_path.exists():
                    cfg.MODEL.WEIGHTS = str(model_path)
                    logger.info(f"✅ Using model from git-lfs: {model_path}")
                    model_found = True
                    break
            
            if not model_found:
                # Try downloading if not in git-lfs
                try:
                    from huggingface_hub import hf_hub_download
                    logger.info("Model not in git-lfs, downloading from HuggingFace...")
                    model_path = hf_hub_download(
                        repo_id=config["swin_model"],
                        filename=model_filename,
                        cache_dir=str(project_root / "models")
                    )
                    cfg.MODEL.WEIGHTS = model_path
                    logger.info(f"✅ Downloaded model: {model_path}")
                except Exception as e:
                    logger.error(f"Failed to download model: {e}")
                    return False
            
            # Resolution settings
            if use_high_res:
                cfg.INPUT.MIN_SIZE_TEST = 1280
                cfg.INPUT.MAX_SIZE_TEST = 1280 * 4
                logger.info("Using high resolution: 1280x1280")
            else:
                cfg.INPUT.MIN_SIZE_TEST = 640
                cfg.INPUT.MAX_SIZE_TEST = 640 * 4
                logger.info("Using standard resolution: 640x640")
            
            cfg.freeze()

            # Initialize predictor
            self.predictor = OneFormerPredictor(cfg)
            
            # Get metadata
            self.metadata = MetadataCatalog.get(
                cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
            )
            
            # Class names
            if hasattr(self.metadata, 'stuff_classes'):
                self.ade20k_classes = self.metadata.stuff_classes
                logger.info(f"Loaded {len(self.ade20k_classes)} ADE20K classes")
            else:
                self.ade20k_classes = [ADE20K_CLASS_NAMES.get(i, f"class_{i}") for i in range(150)]
            
            self.initialized = True
            logger.info("✅ OneFormer initialized successfully with git-lfs model")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize OneFormer: {e}")
            import traceback
            traceback.print_exc()
            return False

    def semantic_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform semantic segmentation"""
        if not self.initialized:
            h, w = image.shape[:2]
            dummy_mask = np.zeros((h, w), dtype=np.uint8)
            return dummy_mask, image, image

        try:
            target_size = 1280 if self.use_high_res else 640
            h_orig, w_orig = image.shape[:2]
            
            # Resize maintaining aspect ratio
            scale = min(target_size / w_orig, target_size / h_orig)
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
            
            image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Pad to square
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
            
            # Extract segmentation
            if "sem_seg" in predictions:
                seg_mask_padded = predictions["sem_seg"].argmax(dim=0).cpu().numpy()
            else:
                seg_mask_padded = np.zeros((target_size, target_size), dtype=np.uint8)

            # Remove padding
            if pad_left > 0 or pad_top > 0:
                seg_mask = seg_mask_padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
            else:
                seg_mask = seg_mask_padded

            # Resize back
            seg_mask_original = cv2.resize(seg_mask.astype(np.uint8), (w_orig, h_orig), 
                                          interpolation=cv2.INTER_NEAREST)

            # Create visualizations
            vis_image = self._create_visualization(image, seg_mask_original, with_labels=False)
            labeled_image = self._create_visualization(image, seg_mask_original, with_labels=True)

            return seg_mask_original, vis_image, labeled_image

        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8), image, image

    def _create_visualization(self, image: np.ndarray, seg_mask: np.ndarray, 
                            with_labels: bool = False) -> np.ndarray:
        """Create visualization"""
        if not DETECTRON2_AVAILABLE:
            return image
            
        try:
            visualizer = Visualizer(
                image[:, :, ::-1],
                metadata=self.metadata,
                instance_mode=ColorMode.IMAGE
            )
            vis_output = visualizer.draw_sem_seg(seg_mask, alpha=0.5)
            vis_image = vis_output.get_image()[:, :, ::-1]

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
            if seg_id == 255 or seg_id >= 150:
                continue
                
            mask = seg_mask == seg_id
            if np.sum(mask) > 500:
                y_coords, x_coords = np.where(mask)
                if len(y_coords) > 0:
                    cy, cx = int(np.mean(y_coords)), int(np.mean(x_coords))
                    
                    if self.ade20k_classes and seg_id < len(self.ade20k_classes):
                        class_name = self.ade20k_classes[int(seg_id)].split(',')[0].strip()
                    else:
                        class_name = f"class_{seg_id}"
                    
                    # Draw label with background
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
            logger.info(f"Floor extraction: {floor_pixels} pixels ({floor_pixels / segmentation.size * 100:.1f}%)")
        else:
            logger.warning("No floor pixels detected")

        return floor_mask
