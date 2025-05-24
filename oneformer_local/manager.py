"""OneFormer model management for semantic segmentation with labeled visualization."""

import torch
import numpy as np
import cv2
import logging
import sys
import os
from typing import Tuple, Dict, Optional, List, Set

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine.defaults import DefaultPredictor

# OneFormer imports - use the correct import structure
try:
    # Import from local oneformer directory
    from oneformer.config import (
        add_oneformer_config,
        add_common_config,
        add_swin_config,
        add_dinat_config,
    )
    from oneformer.oneformer_model import OneFormer
    
    ONEFORMER_AVAILABLE = True
except ImportError as e:
    print(f"OneFormer not available: {e}")
    print("Current path:", sys.path)
    import traceback
    traceback.print_exc()
    ONEFORMER_AVAILABLE = False

from config import DEVICE, ONEFORMER_CONFIG, FLOOR_CLASSES

logger = logging.getLogger(__name__)


class OneFormerPredictor(DefaultPredictor):
    """Custom predictor for OneFormer that handles task conditioning"""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def __call__(self, original_image, task_type="semantic"):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            task_type (str): the task to perform. One of "semantic", "instance", or "panoptic".

        Returns:
            predictions (dict): the output of the model
        """
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
        self.floor_class_ids = set([3, 4, 28, 29, 52, 53, 54, 78, 91])  # floor, wood floor, rug, field, path, stairs, runway, mat, dirt track
        
        # Define which objects can be adjacent (touching)
        self.adjacency_rules = {
            'floor': {'wall', 'door', 'sofa', 'chair', 'table', 'bed', 'cabinet', 'stairs', 'desk', 'toilet', 'bathtub', 'rug'},
            'wall': {'floor', 'door', 'window', 'ceiling', 'mirror', 'picture', 'sofa', 'chair', 'table', 'cabinet', 'television'},
            'ceiling': {'wall', 'lamp', 'chandelier'},
            'door': {'wall', 'floor'},
            'stairs': {'floor', 'wall', 'railing'},
            # Add more as needed
        }

    def initialize(self, backbone: str = "swin", use_high_res: bool = False):
        """Initialize OneFormer model with optional high resolution"""
        if not ONEFORMER_AVAILABLE:
            logger.error("OneFormer not available - please check your installation")
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

            # Try multiple paths for the model
            model_paths_to_try = [
                # Local model in project
                os.path.join(os.path.dirname(__file__), "..", "models", config["swin_file"]),
                # User's home directory cache
                os.path.expanduser(f"~/.cache/huggingface/hub/{config['swin_file']}"),
                # Full path from cache
                "/home/sww35/.cache/huggingface/hub/models--shi-labs--oneformer_ade20k_swin_large/snapshots/4a5bac8e64f82681a12db2e151a4c2f4ce6092b2/250_16_swin_l_oneformer_ade20k_160k.pth",
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
                        token=None  # No authentication required
                    )
                    cfg.MODEL.WEIGHTS = model_path
                    logger.info(f"Downloaded model from HuggingFace: {model_path}")
                except Exception as e:
                    logger.error(f"Failed to download model: {e}")
                    raise RuntimeError(
                        f"Model file '{config['swin_file']}' not found. "
                        f"Please download it manually"
                    )
            
            # Update input size configuration for resolution
            if use_high_res:
                # High resolution is 1280x1280
                cfg.INPUT.MIN_SIZE_TEST = 1280
                cfg.INPUT.MAX_SIZE_TEST = 1280 * 4
                logger.info(f"Using high resolution: 1280x1280")
            else:
                # Standard resolution is 640x640
                cfg.INPUT.MIN_SIZE_TEST = 640
                cfg.INPUT.MAX_SIZE_TEST = 640 * 4
                logger.info(f"Using standard resolution: 640x640")
            
            cfg.freeze()

            # Use our custom predictor
            self.predictor = OneFormerPredictor(cfg)
            
            # Get metadata but DON'T modify stuff_classes - use existing ones
            self.metadata = MetadataCatalog.get(
                cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
            )
            
            # Store the existing class names for our use
            if hasattr(self.metadata, 'stuff_classes'):
                self.ade20k_classes = self.metadata.stuff_classes
                logger.info(f"Using existing ADE20K class names ({len(self.ade20k_classes)} classes)")
            else:
                # Fallback to simple names if not available
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
            raise RuntimeError("OneFormer not initialized")

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
        logger.info(f"Resized image from {w_orig}x{h_orig} to {new_w}x{new_h} (scale: {scale:.2f})")
        
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

        # Run prediction on padded image
        try:
            # Call predictor with task type
            predictions = self.predictor(image_padded, "semantic")
        except TypeError:
            # Fallback: predictor might not accept task type
            predictions = self.predictor(image_padded)
            
        # Extract semantic segmentation
        if "sem_seg" in predictions:
            seg_mask_padded = predictions["sem_seg"].argmax(dim=0).cpu().numpy()
        elif "semantic" in predictions:
            seg_mask_padded = predictions["semantic"].argmax(dim=0).cpu().numpy()
        else:
            # Try to find any segmentation output
            for key in predictions.keys():
                if "seg" in key.lower():
                    seg_mask_padded = predictions[key].argmax(dim=0).cpu().numpy()
                    break
            else:
                raise ValueError(f"No segmentation found in predictions. Keys: {predictions.keys()}")

        # Remove padding from segmentation mask
        if pad_left > 0 or pad_top > 0:
            seg_mask = seg_mask_padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
        else:
            seg_mask = seg_mask_padded

        # Resize segmentation mask back to original size
        seg_mask_original = cv2.resize(seg_mask.astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # Create visualizations at a larger display size for better visibility
        display_width = max(1920, w_orig)
        display_height = int(display_width * h_orig / w_orig)
        
        # Resize image for display
        display_image = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_CUBIC)
        seg_mask_display = cv2.resize(seg_mask_original.astype(np.uint8), (display_width, display_height), interpolation=cv2.INTER_NEAREST)

        # Create standard visualization
        visualizer = Visualizer(
            display_image[:, :, ::-1],  # Convert RGB to BGR for visualization
            metadata=self.metadata,
            instance_mode=ColorMode.IMAGE
        )
        vis_output = visualizer.draw_sem_seg(seg_mask_display, alpha=0.5)
        vis_image = vis_output.get_image()[:, :, ::-1]  # Convert back to RGB

        # Create labeled visualization
        labeled_image = self.create_labeled_visualization(display_image, seg_mask_display)

        return seg_mask_original, vis_image, labeled_image

    def create_labeled_visualization(self, image: np.ndarray, seg_mask: np.ndarray) -> np.ndarray:
        """Create a visualization with object labels"""
        # Create overlay image
        overlay = image.copy()
        labeled_image = image.copy()
        
        # Get unique segments
        unique_segments = np.unique(seg_mask)
        
        # Define a color palette for visualization
        np.random.seed(42)  # For consistent colors
        colors = np.random.randint(0, 255, size=(150, 3), dtype=np.uint8)
        
        # Process each segment
        for seg_id in unique_segments:
            if seg_id == 255 or seg_id >= 150:  # Skip ignore label and out of bounds
                continue
                
            # Get segment mask
            mask = seg_mask == seg_id
            
            # Apply color overlay
            overlay[mask] = colors[int(seg_id) % 150]
            
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
                        
                        # Get class name - use simplified version
                        if self.ade20k_classes and seg_id < len(self.ade20k_classes):
                            full_name = self.ade20k_classes[int(seg_id)]
                            # Simplify the name - take first part before comma
                            class_name = full_name.split(',')[0].strip()
                        else:
                            from config import ADE20K_CLASS_NAMES
                            class_name = ADE20K_CLASS_NAMES.get(int(seg_id), f"class_{seg_id}")
                        
                        # Draw label with background
                        label = class_name
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 2
                        
                        # Get text size
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, font, font_scale, thickness
                        )
                        
                        # Calculate label position (ensure it stays within bounds)
                        label_x = max(2, min(cx - text_width // 2, image.shape[1] - text_width - 2))
                        label_y = max(text_height + 2, min(cy, image.shape[0] - 2))
                        
                        # Background points
                        bg_pt1 = (label_x - 2, label_y - text_height - 2)
                        bg_pt2 = (label_x + text_width + 2, label_y + 2)
                        
                        # Ensure points are valid
                        if bg_pt1[0] >= 0 and bg_pt1[1] >= 0 and bg_pt2[0] <= image.shape[1] and bg_pt2[1] <= image.shape[0]:
                            # Draw background rectangle
                            cv2.rectangle(
                                labeled_image,
                                bg_pt1,
                                bg_pt2,
                                (0, 0, 0),
                                -1
                            )
                            
                            # Draw text
                            cv2.putText(
                                labeled_image,
                                label,
                                (label_x, label_y),
                                font,
                                font_scale,
                                (255, 255, 255),
                                thickness,
                                cv2.LINE_AA
                            )
        
        # Blend overlay with original
        labeled_image = cv2.addWeighted(labeled_image, 0.7, overlay, 0.3, 0)
        
        return labeled_image

    def extract_floor_areas(self, segmentation: np.ndarray) -> np.ndarray:
        """Extract floor areas from segmentation including rugs, carpets, mats"""
        floor_mask = np.zeros_like(segmentation, dtype=bool)

        # Include all floor-related classes
        for seg_id in self.floor_class_ids:
            floor_mask |= (segmentation == seg_id)
        
        # Log what we found
        floor_pixels = np.sum(floor_mask)
        if floor_pixels > 0:
            logger.info(f"Floor extraction: Found {floor_pixels} floor pixels ({floor_pixels / segmentation.size * 100:.1f}%)")
            unique_floor_classes = np.unique(segmentation[floor_mask])
            logger.info(f"Floor classes detected: {unique_floor_classes.tolist()}")
        else:
            logger.warning("No floor pixels detected!")

        return floor_mask

    def detect_blackspots_pixel_based(self, image: np.ndarray, floor_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect black spots on floor using pixel-based method"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Multiple thresholds for different levels of black
        very_black_threshold = 30   # Very dark black
        black_threshold = 50        # Regular black
        dark_threshold = 70         # Dark colors that could be blackspots
        
        # Create masks for different darkness levels
        very_black_mask = (gray < very_black_threshold) & floor_mask
        black_mask = (gray < black_threshold) & floor_mask
        dark_mask = (gray < dark_threshold) & floor_mask
        
        # Also check RGB values to exclude grays (where R≈G≈B)
        color_diff = np.std(image, axis=2)  # Standard deviation across RGB channels
        not_gray_mask = color_diff < 15  # Low std means R≈G≈B (gray)
        
        # Combine: we want dark areas that are either very black OR black but not gray
        blackspot_candidates = (very_black_mask | (black_mask & ~not_gray_mask)) & floor_mask
        
        # Remove small noise
        kernel = np.ones((5, 5), np.uint8)
        blackspot_candidates = cv2.morphologyEx(blackspot_candidates.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        blackspot_candidates = cv2.morphologyEx(blackspot_candidates, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(blackspot_candidates.astype(np.uint8))
        
        final_blackspot_mask = np.zeros_like(blackspot_candidates, dtype=bool)
        pixel_confidence_scores = []
        
        for label_id in range(1, num_labels):
            component_mask = labels == label_id
            component_size = np.sum(component_mask)
            
            if component_size >= 50:  # Minimum 50 pixels
                final_blackspot_mask |= component_mask
                
                # Calculate confidence based on darkness and size
                component_gray_values = gray[component_mask]
                avg_darkness = np.mean(component_gray_values)
                
                # Confidence calculation
                darkness_score = 1.0 - (avg_darkness / dark_threshold)
                size_score = min(1.0, component_size / 1000)  # Normalize by 1000 pixels
                confidence = (darkness_score * 0.7 + size_score * 0.3)
                
                pixel_confidence_scores.append(confidence)
                logger.info(f"Pixel method: Blackspot {label_id} - size: {component_size}px, "
                          f"avg darkness: {avg_darkness:.1f}, confidence: {confidence:.3f}")
        
        avg_confidence = np.mean(pixel_confidence_scores) if pixel_confidence_scores else 0.0
        
        return final_blackspot_mask, avg_confidence

    def check_adjacency(self, seg_mask: np.ndarray, class1: str, class2: str, 
                       id1: int, id2: int) -> bool:
        """Check if two objects are actually adjacent (touching)"""
        # Same object type shouldn't be compared
        if class1 == class2:
            return False
        
        # Check adjacency rules
        if class1 in self.adjacency_rules:
            if class2 not in self.adjacency_rules[class1]:
                return False
        
        # Check if they actually touch
        mask1 = seg_mask == id1
        mask2 = seg_mask == id2
        
        # Dilate masks to check for adjacency
        kernel = np.ones((3, 3), np.uint8)
        dilated1 = cv2.dilate(mask1.astype(np.uint8), kernel, iterations=1)
        dilated2 = cv2.dilate(mask2.astype(np.uint8), kernel, iterations=1)
        
        # Check if dilated masks overlap
        overlap = dilated1 & dilated2
        
        return np.any(overlap)

    def get_object_class_name(self, seg_id: int) -> str:
        """Get simplified class name for a segment ID"""
        if self.ade20k_classes and seg_id < len(self.ade20k_classes):
            full_name = self.ade20k_classes[int(seg_id)]
            return full_name.split(',')[0].strip().lower()
        else:
            from config import ADE20K_CLASS_NAMES
            name = ADE20K_CLASS_NAMES.get(int(seg_id), f"unknown_{seg_id}")
            return name.lower()
