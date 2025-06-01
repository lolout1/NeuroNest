import torch
import numpy as np
from PIL import Image
import cv2
import os
import sys
import time
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import gradio as gr
from huggingface_hub import hf_hub_download
import warnings
warnings.filterwarnings("ignore")

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor as DetectronPredictor
from detectron2 import model_zoo
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

########################################
# GLOBAL CONFIGURATIONS
########################################

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU_DEVICE = torch.device("cpu")
torch.set_num_threads(4)

# ADE20K class mappings for floor detection
FLOOR_CLASSES = {
    'floor': [3, 4, 13],  # floor, wood floor, rug
    'carpet': [28],       # carpet
    'mat': [78],          # mat
}

# OneFormer configurations
ONEFORMER_CONFIG = {
    "ADE20K": {
        "key": "ade20k",
        "swin_cfg": "configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml",
        "swin_model": "shi-labs/oneformer_ade20k_swin_large",
        "swin_file": "250_16_swin_l_oneformer_ade20k_160k.pth",
        "width": 640
    }
}

########################################
# IMPORT UNIVERSAL CONTRAST ANALYZER
########################################

from utils.universal_contrast_analyzer import UniversalContrastAnalyzer

# Keep old class for compatibility but deprecated
class RobustContrastAnalyzer:
    """Advanced contrast analyzer for Alzheimer's-friendly environments"""
    
    def __init__(self, wcag_threshold: float = 4.5):
        self.wcag_threshold = wcag_threshold
        
        # ADE20K class mappings for important objects
        self.semantic_classes = {
            'floor': [3, 4, 13, 28, 78],  # floor, wood floor, rug, carpet, mat
            'wall': [0, 1, 9],            # wall, building, brick
            'ceiling': [5],               # ceiling
            'furniture': [10, 19, 15, 7, 18, 23],  # sofa, chair, table, bed, armchair, cabinet
            'door': [25],                 # door
            'window': [8],                # window
            'stairs': [53],               # stairs
        }
        
        # Priority relationships for safety
        self.priority_relationships = {
            ('floor', 'furniture'): ('critical', 'Furniture must be clearly visible against floor'),
            ('floor', 'stairs'): ('critical', 'Stairs must have clear contrast with floor'),
            ('floor', 'door'): ('high', 'Door should be easily distinguishable from floor'),
            ('wall', 'furniture'): ('high', 'Furniture should stand out from walls'),
            ('wall', 'door'): ('high', 'Doors should be clearly visible on walls'),
            ('wall', 'window'): ('medium', 'Windows should have adequate contrast'),
            ('ceiling', 'wall'): ('low', 'Ceiling-wall contrast is less critical'),
        }
    
    def get_object_category(self, class_id: int) -> str:
        """Map segmentation class to object category"""
        for category, class_ids in self.semantic_classes.items():
            if class_id in class_ids:
                return category
        return 'other'
    
    def calculate_wcag_contrast(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """Calculate WCAG contrast ratio"""
        def relative_luminance(rgb):
            rgb_norm = rgb / 255.0
            rgb_linear = np.where(rgb_norm <= 0.03928, 
                                rgb_norm / 12.92, 
                                ((rgb_norm + 0.055) / 1.055) ** 2.4)
            return np.dot(rgb_linear, [0.2126, 0.7152, 0.0722])
        
        lum1 = relative_luminance(color1)
        lum2 = relative_luminance(color2)
        
        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)
        
        return (lighter + 0.05) / (darker + 0.05)
    
    def extract_dominant_color(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract dominant color from masked region"""
        if not np.any(mask):
            return np.array([128, 128, 128])
        
        masked_pixels = image[mask]
        if len(masked_pixels) == 0:
            return np.array([128, 128, 128])
        
        # Use median for robustness against outliers
        return np.median(masked_pixels, axis=0).astype(int)
    
    def find_adjacent_segments(self, seg1_mask: np.ndarray, seg2_mask: np.ndarray, 
                             min_boundary_length: int = 30) -> np.ndarray:
        """Find clean boundaries between segments"""
        kernel = np.ones((3, 3), np.uint8)
        dilated1 = cv2.dilate(seg1_mask.astype(np.uint8), kernel, iterations=1)
        dilated2 = cv2.dilate(seg2_mask.astype(np.uint8), kernel, iterations=1)
        
        boundary = dilated1 & dilated2
        
        # Remove small disconnected components
        contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_boundary = np.zeros_like(boundary)
        
        for contour in contours:
            if cv2.contourArea(contour) >= min_boundary_length:
                cv2.fillPoly(clean_boundary, [contour], 1)
        
        return clean_boundary.astype(bool)
    
    def analyze_contrast(self, image: np.ndarray, segmentation: np.ndarray) -> Dict:
        """Perform comprehensive contrast analysis"""
        h, w = segmentation.shape
        results = {
            'critical_issues': [],
            'high_issues': [],
            'medium_issues': [],
            'visualization': image.copy(),
            'statistics': {}
        }
        
        # Build segment information
        unique_segments = np.unique(segmentation)
        segment_info = {}
        
        for seg_id in unique_segments:
            if seg_id == 0:  # Skip background
                continue
                
            mask = segmentation == seg_id
            if np.sum(mask) < 100:  # Skip very small segments
                continue
                
            category = self.get_object_category(seg_id)
            if category == 'other':
                continue
                
            segment_info[seg_id] = {
                'category': category,
                'mask': mask,
                'color': self.extract_dominant_color(image, mask),
                'area': np.sum(mask)
            }
        
        # Analyze priority relationships
        issue_counts = {'critical': 0, 'high': 0, 'medium': 0}
        
        for seg_id1, info1 in segment_info.items():
            for seg_id2, info2 in segment_info.items():
                if seg_id1 >= seg_id2:
                    continue
                
                # Check if this is a priority relationship
                relationship = tuple(sorted([info1['category'], info2['category']]))
                if relationship not in self.priority_relationships:
                    continue
                
                priority, description = self.priority_relationships[relationship]
                
                # Check if segments are adjacent
                boundary = self.find_adjacent_segments(info1['mask'], info2['mask'])
                if not np.any(boundary):
                    continue
                
                # Calculate contrast
                wcag_contrast = self.calculate_wcag_contrast(info1['color'], info2['color'])
                
                # Determine if there's an issue
                if wcag_contrast < self.wcag_threshold:
                    issue = {
                        'categories': (info1['category'], info2['category']),
                        'contrast_ratio': wcag_contrast,
                        'boundary_area': np.sum(boundary),
                        'description': description,
                        'priority': priority
                    }
                    
                    # Color-code boundaries and store issues
                    if priority == 'critical':
                        results['critical_issues'].append(issue)
                        results['visualization'][boundary] = [255, 0, 0]  # Red
                        issue_counts['critical'] += 1
                    elif priority == 'high':
                        results['high_issues'].append(issue)
                        results['visualization'][boundary] = [255, 165, 0]  # Orange
                        issue_counts['high'] += 1
                    elif priority == 'medium':
                        results['medium_issues'].append(issue)
                        results['visualization'][boundary] = [255, 255, 0]  # Yellow
                        issue_counts['medium'] += 1
        
        # Calculate statistics
        results['statistics'] = {
            'total_segments': len(segment_info),
            'total_issues': sum(issue_counts.values()),
            'critical_count': issue_counts['critical'],
            'high_count': issue_counts['high'],
            'medium_count': issue_counts['medium'],
            'wcag_threshold': self.wcag_threshold
        }
        
        return results

########################################
# ONEFORMER INTEGRATION
########################################

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


########################################
# ENHANCED BLACKSPOT DETECTION WITH CLEAR VISUALIZATION
########################################

class BlackspotDetector:
    """Manages blackspot detection with MaskRCNN - Enhanced Version"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.predictor = None
    
    def initialize(self, threshold: float = 0.5) -> bool:
        """Initialize MaskRCNN model"""
        try:
            cfg = get_cfg()
            cfg.merge_from_file(
                model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            )
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # [floors, blackspot]
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
            cfg.MODEL.WEIGHTS = self.model_path
            cfg.MODEL.DEVICE = DEVICE
            
            self.predictor = DetectronPredictor(cfg)
            logger.info("MaskRCNN blackspot detector initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize blackspot detector: {e}")
            return False
    
    def create_enhanced_visualizations(self, image: np.ndarray, floor_mask: np.ndarray, 
                                     blackspot_mask: np.ndarray) -> Dict:
        """Create multiple enhanced visualizations of blackspot detection"""
        
        # 1. Pure Segmentation View (like semantic segmentation output)
        segmentation_view = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
        segmentation_view[floor_mask] = [34, 139, 34]      # Forest green for floor
        segmentation_view[blackspot_mask] = [255, 0, 0]    # Bright red for blackspots
        segmentation_view[~(floor_mask | blackspot_mask)] = [128, 128, 128]  # Gray for other areas
        
        # 2. High Contrast Overlay
        high_contrast_overlay = image.copy()
        # Make background slightly darker to emphasize blackspots
        high_contrast_overlay = cv2.convertScaleAbs(high_contrast_overlay, alpha=0.6, beta=0)
        # Add bright overlays
        high_contrast_overlay[floor_mask] = cv2.addWeighted(
            high_contrast_overlay[floor_mask], 0.7, 
            np.full_like(high_contrast_overlay[floor_mask], [0, 255, 0]), 0.3, 0
        )
        high_contrast_overlay[blackspot_mask] = [255, 0, 255]  # Magenta for maximum visibility
        
        # 3. Blackspot-only View (white blackspots on black background)
        blackspot_only = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
        blackspot_only[blackspot_mask] = [255, 255, 255]  # White blackspots
        blackspot_only[floor_mask & ~blackspot_mask] = [64, 64, 64]  # Dark gray for floor areas
        
        # 4. Side-by-side comparison
        h, w = image.shape[:2]
        side_by_side = np.zeros((h, w * 2, 3), dtype=np.uint8)
        side_by_side[:, :w] = image
        side_by_side[:, w:] = segmentation_view
        
        # Add text labels
        cv2.putText(side_by_side, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(side_by_side, "Blackspot Detection", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 5. Annotated view with bounding boxes and labels
        annotated_view = image.copy()
        
        # Find blackspot contours for bounding boxes
        blackspot_contours, _ = cv2.findContours(
            blackspot_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for i, contour in enumerate(blackspot_contours):
            if cv2.contourArea(contour) > 50:  # Filter small artifacts
                # Draw bounding box
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(annotated_view, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Draw contour
                cv2.drawContours(annotated_view, [contour], -1, (255, 0, 255), 2)
                
                # Add label
                area = cv2.contourArea(contour)
                label = f"Blackspot {i+1}: {area:.0f}px"
                cv2.putText(annotated_view, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return {
            'segmentation_view': segmentation_view,
            'high_contrast_overlay': high_contrast_overlay,
            'blackspot_only': blackspot_only,
            'side_by_side': side_by_side,
            'annotated_view': annotated_view
        }
    
    def detect_blackspots(self, image: np.ndarray, floor_prior: Optional[np.ndarray] = None) -> Dict:
        """Detect blackspots with enhanced visualizations"""
        if self.predictor is None:
            raise RuntimeError("Blackspot detector not initialized")
        
        # Get original image dimensions
        original_h, original_w = image.shape[:2]
        
        # Handle floor prior shape mismatch
        processed_image = image.copy()
        if floor_prior is not None:
            prior_h, prior_w = floor_prior.shape
            
            # Resize floor_prior to match original image if needed
            if (prior_h, prior_w) != (original_h, original_w):
                logger.info(f"Resizing floor prior from {(prior_h, prior_w)} to {(original_h, original_w)}")
                floor_prior_resized = cv2.resize(
                    floor_prior.astype(np.uint8), 
                    (original_w, original_h), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                floor_prior_resized = floor_prior
        else:
            floor_prior_resized = None
        
        # Run detection on the processed image
        try:
            outputs = self.predictor(processed_image)
            instances = outputs["instances"].to("cpu")
        except Exception as e:
            logger.error(f"Error in MaskRCNN prediction: {e}")
            # Return empty results
            empty_mask = np.zeros(image.shape[:2], dtype=bool)
            return {
                'visualization': image,
                'floor_mask': empty_mask,
                'blackspot_mask': empty_mask,
                'floor_area': 0,
                'blackspot_area': 0,
                'coverage_percentage': 0,
                'num_detections': 0,
                'avg_confidence': 0.0,
                'enhanced_views': self.create_enhanced_visualizations(image, empty_mask, empty_mask)
            }
        
        # Process results
        if len(instances) == 0:
            # No detections
            combined_floor = floor_prior_resized if floor_prior_resized is not None else np.zeros(image.shape[:2], dtype=bool)
            combined_blackspot = np.zeros(image.shape[:2], dtype=bool)
            blackspot_scores = []
        else:
            pred_classes = instances.pred_classes.numpy()
            pred_masks = instances.pred_masks.numpy()
            scores = instances.scores.numpy()
            
            # Separate floor and blackspot masks
            floor_indices = pred_classes == 0
            blackspot_indices = pred_classes == 1
            
            floor_masks = pred_masks[floor_indices] if np.any(floor_indices) else []
            blackspot_masks = pred_masks[blackspot_indices] if np.any(blackspot_indices) else []
            blackspot_scores = scores[blackspot_indices] if np.any(blackspot_indices) else []
            
            # Combine masks
            combined_floor = np.zeros(image.shape[:2], dtype=bool)
            combined_blackspot = np.zeros(image.shape[:2], dtype=bool)
            
            for mask in floor_masks:
                combined_floor |= mask
            
            for mask in blackspot_masks:
                combined_blackspot |= mask
            
            # Apply floor prior if available
            if floor_prior_resized is not None:
                # Combine OneFormer floor detection with MaskRCNN floor detection
                combined_floor |= floor_prior_resized
                # Keep only blackspots that are on floors
                combined_blackspot &= combined_floor
        
        # Create all enhanced visualizations
        enhanced_views = self.create_enhanced_visualizations(image, combined_floor, combined_blackspot)
        
        # Calculate statistics
        floor_area = int(np.sum(combined_floor))
        blackspot_area = int(np.sum(combined_blackspot))
        coverage_percentage = (blackspot_area / floor_area * 100) if floor_area > 0 else 0
        
        # Count individual blackspot instances
        blackspot_contours, _ = cv2.findContours(
            combined_blackspot.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        actual_detections = len([c for c in blackspot_contours if cv2.contourArea(c) > 50])
        
        return {
            'visualization': enhanced_views['high_contrast_overlay'],  # Main view
            'floor_mask': combined_floor,
            'blackspot_mask': combined_blackspot,
            'floor_area': floor_area,
            'blackspot_area': blackspot_area,
            'coverage_percentage': coverage_percentage,
            'num_detections': actual_detections,
            'avg_confidence': float(np.mean(blackspot_scores)) if len(blackspot_scores) > 0 else 0.0,
            'enhanced_views': enhanced_views  # All visualization options
        }
########################################
# FIXED MAIN APPLICATION CLASS
########################################

class NeuroNestApp:
    """Main application class integrating all components - FIXED VERSION"""
    
    def __init__(self):
        self.oneformer = OneFormerManager()
        self.blackspot_detector = None
        self.contrast_analyzer = UniversalContrastAnalyzer()
        self.initialized = False
    
    def initialize(self, blackspot_model_path: str = "./output_floor_blackspot/model_0004999.pth"):
        """Initialize all components"""
        logger.info("Initializing NeuroNest application...")
        
        # Initialize OneFormer
        oneformer_success = self.oneformer.initialize()
        
        # Initialize blackspot detector if model exists
        blackspot_success = False
        if os.path.exists(blackspot_model_path):
            self.blackspot_detector = BlackspotDetector(blackspot_model_path)
            blackspot_success = True
        else:
            logger.warning(f"Blackspot model not found at {blackspot_model_path}")
        
        self.initialized = oneformer_success
        return oneformer_success, blackspot_success
    
    def analyze_image(self, 
                     image_path: str,
                     blackspot_threshold: float = 0.5,
                     contrast_threshold: float = 4.5,
                     enable_blackspot: bool = True,
                     enable_contrast: bool = True) -> Dict:
        """Perform complete image analysis - FIXED VERSION"""
        
        if not self.initialized:
            return {"error": "Application not properly initialized"}
        
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image"}
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.info(f"Loaded image with shape: {image_rgb.shape}")
            
            results = {
                'original_image': image_rgb,
                'segmentation': None,
                'blackspot': None,
                'contrast': None,
                'statistics': {}
            }
            
            # 1. Semantic Segmentation (always performed)
            logger.info("Running semantic segmentation...")
            seg_mask, seg_visualization = self.oneformer.semantic_segmentation(image_rgb)
            logger.info(f"Segmentation mask shape: {seg_mask.shape}")
            
            results['segmentation'] = {
                'visualization': seg_visualization,
                'mask': seg_mask
            }
            
            # Extract floor areas for blackspot detection
            floor_prior = self.oneformer.extract_floor_areas(seg_mask)
            logger.info(f"Floor prior shape: {floor_prior.shape}, total floor pixels: {np.sum(floor_prior)}")
            
            # 2. Blackspot Detection (if enabled and model available)
            if enable_blackspot and self.blackspot_detector is not None:
                logger.info("Running blackspot detection...")
                try:
                    self.blackspot_detector.initialize(threshold=blackspot_threshold)
                    blackspot_results = self.blackspot_detector.detect_blackspots(image_rgb, floor_prior)
                    results['blackspot'] = blackspot_results
                    logger.info("Blackspot detection completed successfully")
                except Exception as e:
                    logger.error(f"Error in blackspot detection: {e}")
                    # Continue without blackspot results
                    results['blackspot'] = None
            
            # 3. Contrast Analysis (if enabled)
            if enable_contrast:
                logger.info("Running contrast analysis...")
                try:
                    # Use the resized image for contrast analysis to match segmentation
                    width = ONEFORMER_CONFIG["ADE20K"]["width"]
                    h, w = image_rgb.shape[:2]
                    if w != width:
                        scale = width / w
                        new_h = int(h * scale)
                        image_for_contrast = cv2.resize(image_rgb, (width, new_h))
                    else:
                        image_for_contrast = image_rgb
                    
                    contrast_results = self.contrast_analyzer.analyze_contrast(image_for_contrast, seg_mask)
                    results['contrast'] = contrast_results
                    logger.info("Contrast analysis completed successfully")
                except Exception as e:
                    logger.error(f"Error in contrast analysis: {e}")
                    # Continue without contrast results
                    results['contrast'] = None
            
            # 4. Generate combined statistics
            stats = self._generate_statistics(results)
            results['statistics'] = stats
            
            logger.info("Image analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _generate_statistics(self, results: Dict) -> Dict:
        """Generate comprehensive statistics"""
        stats = {}
        
        # Segmentation stats
        if results['segmentation']:
            unique_classes = np.unique(results['segmentation']['mask'])
            stats['segmentation'] = {
                'num_classes': len(unique_classes),
                'image_size': results['segmentation']['mask'].shape
            }
        
        # Blackspot stats
        if results['blackspot']:
            bs = results['blackspot']
            stats['blackspot'] = {
                'floor_area_pixels': bs['floor_area'],
                'blackspot_area_pixels': bs['blackspot_area'],
                'coverage_percentage': bs['coverage_percentage'],
                'num_detections': bs['num_detections'],
                'avg_confidence': bs['avg_confidence']
            }
        
        # Contrast stats
        if results['contrast']:
            cs = results['contrast']['statistics']
            # Count issues by severity
            critical_count = sum(1 for issue in results['contrast'].get('issues', []) if issue['severity'] == 'critical')
            high_count = sum(1 for issue in results['contrast'].get('issues', []) if issue['severity'] == 'high')
            medium_count = sum(1 for issue in results['contrast'].get('issues', []) if issue['severity'] == 'medium')
            
            stats['contrast'] = {
                'total_issues': cs.get('low_contrast_pairs', 0),
                'critical_issues': critical_count,
                'high_priority_issues': high_count,
                'medium_priority_issues': medium_count,
                'segments_analyzed': cs.get('total_segments', 0),
                'floor_object_issues': cs.get('floor_object_issues', 0)
            }
        
        return stats
########################################
# GRADIO INTERFACE
########################################

########################################
# ENHANCED GRADIO INTERFACE WITH MULTIPLE BLACKSPOT VIEWS
########################################

def create_gradio_interface():
    """Create the enhanced Gradio interface with better blackspot visualization"""
    
    # Initialize the application
    app = NeuroNestApp()
    oneformer_ok, blackspot_ok = app.initialize()
    
    if not oneformer_ok:
        raise RuntimeError("Failed to initialize OneFormer")
    
    def analyze_wrapper(image_path, blackspot_threshold, contrast_threshold, 
                       enable_blackspot, enable_contrast, blackspot_view_type):
        """Enhanced wrapper function for Gradio interface"""
        if image_path is None:
            return None, None, None, None, None, "Please upload an image"
        
        results = app.analyze_image(
            image_path=image_path,
            blackspot_threshold=blackspot_threshold,
            contrast_threshold=contrast_threshold,
            enable_blackspot=enable_blackspot,
            enable_contrast=enable_contrast
        )
        
        if "error" in results:
            return None, None, None, None, None, f"Error: {results['error']}"
        
        # Extract outputs
        seg_output = results['segmentation']['visualization'] if results['segmentation'] else None
        
        # Enhanced blackspot output selection
        blackspot_output = None
        blackspot_segmentation = None
        if results['blackspot'] and 'enhanced_views' in results['blackspot']:
            views = results['blackspot']['enhanced_views']
            
            # Select view based on user choice
            if blackspot_view_type == "High Contrast":
                blackspot_output = views['high_contrast_overlay']
            elif blackspot_view_type == "Segmentation Only":
                blackspot_output = views['segmentation_view']
            elif blackspot_view_type == "Blackspots Only":
                blackspot_output = views['blackspot_only']
            elif blackspot_view_type == "Side by Side":
                blackspot_output = views['side_by_side']
            elif blackspot_view_type == "Annotated":
                blackspot_output = views['annotated_view']
            else:
                blackspot_output = views['high_contrast_overlay']
            
            # Always provide segmentation view for the dedicated tab
            blackspot_segmentation = views['segmentation_view']
        
        contrast_output = results['contrast']['visualization'] if results['contrast'] else None
        
        # Generate report
        report = generate_analysis_report(results)
        
        return seg_output, blackspot_output, blackspot_segmentation, contrast_output, report
    
    # Update the generate_analysis_report function
    def generate_analysis_report(results: Dict) -> str:
        """Generate enhanced analysis report text"""
        report = ["# NeuroNest Analysis Report\n"]
        
        # Segmentation results
        if results['segmentation']:
            stats = results['statistics'].get('segmentation', {})
            report.append(f"## 🎯 Semantic Segmentation")
            report.append(f"- **Objects detected:** {stats.get('num_classes', 'N/A')}")
            report.append(f"- **Image size:** {stats.get('image_size', 'N/A')}")
            report.append("")
        
        # Enhanced blackspot results
        if results['blackspot']:
            bs_stats = results['statistics'].get('blackspot', {})
            report.append(f"## ⚫ Blackspot Detection")
            report.append(f"- **Floor area:** {bs_stats.get('floor_area_pixels', 0):,} pixels")
            report.append(f"- **Blackspot area:** {bs_stats.get('blackspot_area_pixels', 0):,} pixels")
            report.append(f"- **Coverage:** {bs_stats.get('coverage_percentage', 0):.2f}% of floor")
            report.append(f"- **Individual blackspots:** {bs_stats.get('num_detections', 0)}")
            report.append(f"- **Average confidence:** {bs_stats.get('avg_confidence', 0):.2f}")
            
            # Risk assessment
            coverage = bs_stats.get('coverage_percentage', 0)
            if coverage > 5:
                report.append(f"- **⚠️ Risk Level:** HIGH - Significant blackspot coverage detected")
            elif coverage > 1:
                report.append(f"- **⚠️ Risk Level:** MEDIUM - Moderate blackspot coverage")
            elif coverage > 0:
                report.append(f"- **✓ Risk Level:** LOW - Minimal blackspot coverage")
            else:
                report.append(f"- **✓ Risk Level:** NONE - No blackspots detected")
            report.append("")
        
        # Contrast analysis results (updated for universal analyzer)
        if results['contrast']:
            contrast_stats = results['statistics'].get('contrast', {})
            report.append(f"## 🎨 Universal Contrast Analysis")
            report.append(f"- **Adjacent pairs analyzed:** {results['contrast']['statistics'].get('analyzed_pairs', 0)}")
            report.append(f"- **Total contrast issues:** {contrast_stats.get('total_issues', 0)}")
            report.append(f"- **🔴 Critical:** {contrast_stats.get('critical_issues', 0)}")
            report.append(f"- **🟠 High priority:** {contrast_stats.get('high_priority_issues', 0)}")
            report.append(f"- **🟡 Medium priority:** {contrast_stats.get('medium_priority_issues', 0)}")
            report.append(f"- **⚠️ Floor-object issues:** {contrast_stats.get('floor_object_issues', 0)}")
            report.append("")
            
            # Add detailed issues
            issues = results['contrast'].get('issues', [])
            if issues:
                # Group by severity
                critical_issues = [i for i in issues if i['severity'] == 'critical']
                high_issues = [i for i in issues if i['severity'] == 'high']
                
                if critical_issues:
                    report.append("### 🔴 Critical Issues (Immediate Attention Required)")
                    for issue in critical_issues[:5]:  # Show top 5
                        cats = f"{issue['categories'][0]} ↔ {issue['categories'][1]}"
                        ratio = issue['wcag_ratio']
                        report.append(f"- **{cats}**: {ratio:.1f}:1 contrast ratio")
                        if issue['is_floor_object']:
                            report.append(f"  _⚠️ Object on floor - high visibility required!_")
                    report.append("")
                
                if high_issues:
                    report.append("### 🟠 High Priority Issues")
                    for issue in high_issues[:3]:  # Show top 3
                        cats = f"{issue['categories'][0]} ↔ {issue['categories'][1]}"
                        ratio = issue['wcag_ratio']
                        report.append(f"- **{cats}**: {ratio:.1f}:1 contrast ratio")
                    report.append("")
        
        # Enhanced recommendations
        report.append("## 📋 Recommendations")
        
        # Blackspot-specific recommendations
        if results['blackspot']:
            coverage = results['statistics'].get('blackspot', {}).get('coverage_percentage', 0)
            if coverage > 0:
                report.append("### Blackspot Mitigation")
                report.append("- Remove or replace dark-colored floor materials in detected areas")
                report.append("- Improve lighting in blackspot areas")
                report.append("- Consider using light-colored rugs or mats to cover blackspots")
                report.append("- Add visual cues like contrasting tape around problem areas")
                report.append("")
        
        # Contrast-specific recommendations
        contrast_issues = results['statistics'].get('contrast', {}).get('total_issues', 0)
        if contrast_issues > 0:
            report.append("### Contrast Improvements")
            report.append("- Increase lighting in low-contrast areas")
            report.append("- Use contrasting colors for furniture and floors")
            report.append("- Add visual markers for important boundaries")
            report.append("- Consider color therapy guidelines for dementia")
            report.append("")
        
        if coverage == 0 and contrast_issues == 0:
            report.append("✅ **Environment Assessment: EXCELLENT**")
            report.append("No significant safety issues detected. This environment appears well-suited for individuals with Alzheimer's.")
        
        return "\n".join(report)
    
    # Create the interface with enhanced controls
    title = "🧠 NeuroNest: Advanced Environment Analysis for Alzheimer's Care"
    description = """
    **Comprehensive analysis system for creating Alzheimer's-friendly environments**
    
    This application integrates:
    - **Semantic Segmentation**: Identifies rooms, furniture, and objects
    - **Enhanced Blackspot Detection**: Locates and visualizes dangerous black areas on floors
    - **Contrast Analysis**: Evaluates color contrast for visual accessibility
    """
    
    with gr.Blocks(
        title=title,
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue"),
        css="""
        .main-header { text-align: center; margin-bottom: 2rem; }
        .analysis-section { border: 2px solid #f0f0f0; border-radius: 10px; padding: 1rem; margin: 1rem 0; }
        .critical-text { color: #ff0000; font-weight: bold; }
        .high-text { color: #ff8800; font-weight: bold; }
        .medium-text { color: #ffaa00; font-weight: bold; }
        """
    ) as interface:
        
        gr.Markdown(f"# {title}")
        gr.Markdown(description)
        
        with gr.Row():
            # Input Column
            with gr.Column(scale=1):
                # Image upload
                image_input = gr.Image(
                    label="📸 Upload Room Image",
                    type="filepath",
                    height=300
                )
                
                # Analysis settings
                with gr.Accordion("🔧 Analysis Settings", open=True):
                    enable_blackspot = gr.Checkbox(
                        value=blackspot_ok,
                        label="Enable Blackspot Detection",
                        interactive=blackspot_ok
                    )
                    
                    blackspot_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.05,
                        label="Blackspot Detection Threshold",
                        visible=blackspot_ok
                    )
                    
                    # NEW: Blackspot visualization options
                    blackspot_view_type = gr.Radio(
                        choices=["High Contrast", "Segmentation Only", "Blackspots Only", "Side by Side", "Annotated"],
                        value="High Contrast",
                        label="Blackspot Visualization Style",
                        visible=blackspot_ok
                    )
                    
                    enable_contrast = gr.Checkbox(
                        value=True,
                        label="Enable Contrast Analysis"
                    )
                    
                    contrast_threshold = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=4.5,
                        step=0.1,
                        label="WCAG Contrast Threshold"
                    )
                
                # Analysis button
                analyze_button = gr.Button(
                    "🔍 Analyze Environment",
                    variant="primary",
                    size="lg"
                )
            
            # Output Column
            with gr.Column(scale=2):
                # Main display (Segmentation by default)
                main_display = gr.Image(
                    label="🎯 Object Detection & Segmentation",
                    height=400,
                    interactive=False
                )
                
                # Enhanced analysis tabs
                with gr.Tabs():
                    with gr.Tab("📊 Analysis Report"):
                        analysis_report = gr.Markdown(
                            value="Upload an image and click 'Analyze Environment' to see results.",
                            elem_classes=["analysis-section"]
                        )
                    
                    if blackspot_ok:
                        with gr.Tab("⚫ Blackspot Detection"):
                            blackspot_display = gr.Image(
                                label="Blackspot Analysis (Selected View)",
                                height=300,
                                interactive=False
                            )
                        
                        with gr.Tab("🔍 Blackspot Segmentation"):
                            blackspot_segmentation_display = gr.Image(
                                label="Pure Blackspot Segmentation",
                                height=300,
                                interactive=False
                            )
                    else:
                        blackspot_display = gr.Image(visible=False)
                        blackspot_segmentation_display = gr.Image(visible=False)
                    
                    with gr.Tab("🎨 Contrast Analysis"):
                        contrast_display = gr.Image(
                            label="Contrast Issues Visualization",
                            height=300,
                            interactive=False
                        )
        
        # Connect the interface
        analyze_button.click(
            fn=analyze_wrapper,
            inputs=[
                image_input,
                blackspot_threshold,
                contrast_threshold,
                enable_blackspot,
                enable_contrast,
                blackspot_view_type
            ],
            outputs=[
                main_display,
                blackspot_display,
                blackspot_segmentation_display,
                contrast_display,
                analysis_report
            ]
        )
        
        # Example images (optional)
        example_dir = Path("examples")
        if example_dir.exists():
            examples = [
                [str(img), 0.5, 4.5, True, True, "High Contrast"] 
                for img in example_dir.glob("*.jpg")
            ]
            
            if examples:
                gr.Examples(
                    examples=examples[:3],  # Show max 3 examples
                    inputs=[
                        image_input,
                        blackspot_threshold,
                        contrast_threshold,
                        enable_blackspot,
                        enable_contrast,
                        blackspot_view_type
                    ],
                    outputs=[
                        main_display,
                        blackspot_display,
                        blackspot_segmentation_display,
                        contrast_display,
                        analysis_report
                    ],
                    fn=analyze_wrapper,
                    label="🖼️ Example Images"
                )
        
        # Footer
        gr.Markdown("""
            ---
            **NeuroNest** - Advanced AI for Alzheimer's-friendly environments  
            *Helping create safer, more accessible spaces for cognitive health*
            """)
    
    return interface

###############################
# MAIN EXECUTION - FIXED
########################################

if __name__ == "__main__":
    print(f"🚀 Starting NeuroNest on {DEVICE}")
    print(f"OneFormer available: {ONEFORMER_AVAILABLE}")
    
    try:
        interface = create_gradio_interface()
        
        # Fixed launch call - removed incompatible parameters
        interface.queue(max_size=10).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True
        )
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        raise
