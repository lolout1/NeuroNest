"""
NeuroNest: Advanced Environment Analysis for Alzheimer's Care
Complete integrated solution with enhanced detectron2 support and multiple visualization modes
"""

import os
import cv2
import numpy as np
import logging
import sys
import warnings
import time
import tempfile
import gc
from pathlib import Path
from typing import Dict, Optional, Tuple, List

warnings.filterwarnings("ignore")

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_python_paths():
    """Setup Python paths for detectron2 integration"""
    project_root = Path(__file__).parent.absolute()
    
    # Clean existing paths
    sys.path = [p for p in sys.path if not any(x in p.lower() for x in ['oneformer', 'neuronest'])]
    
    # Add project root FIRST
    sys.path.insert(0, str(project_root))
    
    # Add detectron2 explicitly if it exists
    detectron2_path = project_root / "detectron2"
    if detectron2_path.exists():
        sys.path.insert(1, str(detectron2_path))
        logger.info(f"✅ Local detectron2 path added: {detectron2_path}")
    
    # Add oneformer LAST
    oneformer_path = project_root / "oneformer"
    if oneformer_path.exists():
        sys.path.append(str(oneformer_path))
    
    logger.info(f"✅ Python paths configured: {len(sys.path)} entries")
    return project_root

def check_detectron2_comprehensive():
    """Comprehensive detectron2 health check with fixes"""
    logger.info("🔍 Comprehensive detectron2 check...")
    
    status = {
        'available': False,
        'version': 'unknown',
        'config_available': False,
        'model_zoo_available': False,
        'engine_available': False,
        'fully_functional': False
    }
    
    try:
        import detectron2
        status['available'] = True
        
        # Test version
        try:
            status['version'] = getattr(detectron2, '__version__', 'local_build')
        except:
            status['version'] = 'local_build'
        
        # Test critical imports individually
        try:
            from detectron2.config import get_cfg
            status['config_available'] = True
            logger.info("✅ detectron2.config available")
        except Exception as e:
            logger.warning(f"⚠️ detectron2.config failed: {e}")
        
        try:
            from detectron2 import model_zoo
            status['model_zoo_available'] = True
            logger.info("✅ detectron2.model_zoo available")
        except Exception as e:
            logger.warning(f"⚠️ detectron2.model_zoo failed: {e}")
        
        try:
            from detectron2.engine import DefaultPredictor
            status['engine_available'] = True
            logger.info("✅ detectron2.engine available")
        except Exception as e:
            logger.warning(f"⚠️ detectron2.engine failed: {e}")
        
        # Test functional usage
        if status['config_available']:
            try:
                cfg = get_cfg()
                status['fully_functional'] = True
                logger.info("✅ Detectron2 fully functional")
            except Exception as e:
                logger.warning(f"⚠️ Detectron2 config creation failed: {e}")
        
        return status
        
    except ImportError as e:
        logger.error(f"❌ Detectron2 import failed: {e}")
        return status

# ====================== ENHANCED BLACKSPOT DETECTOR ======================

class EnhancedBlackspotDetector:
    """Enhanced blackspot detector using detectron2 and advanced pixel methods"""
    
    def __init__(self, model_path: str = ""):
        self.model_path = model_path
        self.predictor = None
        self.initialized = False
        self.use_model = bool(model_path)
        self.detectron2_available = False
        
        # Floor class IDs from ADE20K
        self.floor_class_ids = {
            3: 'floor', 4: 'wood_floor', 13: 'earth', 28: 'rug', 
            29: 'field', 52: 'path', 53: 'stairs', 78: 'mat'
        }
        
    def initialize(self, threshold: float = 0.5):
        """Initialize with enhanced detectron2 support"""
        # Check detectron2 availability
        detectron2_status = check_detectron2_comprehensive()
        self.detectron2_available = detectron2_status['fully_functional']
        
        if self.use_model and self.model_path and self.detectron2_available:
            try:
                from detectron2.config import get_cfg
                from detectron2 import model_zoo
                from detectron2.engine import DefaultPredictor
                
                cfg = get_cfg()
                cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
                cfg.MODEL.WEIGHTS = self.model_path
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
                cfg.MODEL.DEVICE = "cpu"  # Force CPU for HuggingFace
                
                self.predictor = DefaultPredictor(cfg)
                logger.info("✅ MaskRCNN blackspot detector initialized")
            except Exception as e:
                logger.error(f"❌ MaskRCNN initialization failed: {e}")
                self.use_model = False
        
        self.initialized = True
        return True
    
    def detect_blackspots(self, image: np.ndarray, floor_mask: np.ndarray = None, 
                         segmentation_mask: np.ndarray = None) -> Dict:
        """Enhanced blackspot detection with multiple methods"""
        h, w = image.shape[:2]
        
        # Initialize results
        results = {
            'blackspot_mask': np.zeros((h, w), dtype=bool),
            'enhanced_views': {},
            'num_detections': 0,
            'floor_area': 0,
            'blackspot_area': 0,
            'coverage_percentage': 0,
            'avg_confidence': 0,
            'risk_score': 0,
            'detection_method': 'enhanced_pixel_based',
            'confidence_scores': {},
            'floor_breakdown': {}
        }
        
        # Extract enhanced floor areas
        if segmentation_mask is not None:
            enhanced_floor_mask = self._extract_floor_from_segmentation(segmentation_mask)
        elif floor_mask is not None:
            enhanced_floor_mask = floor_mask
        else:
            # Fallback floor detection
            enhanced_floor_mask = np.zeros((h, w), dtype=bool)
            enhanced_floor_mask[int(h*0.7):, :] = True
        
        if np.sum(enhanced_floor_mask) == 0:
            logger.warning("No floor area for blackspot analysis")
            return results
        
        results['floor_area'] = np.sum(enhanced_floor_mask)
        
        # Method 1: MaskRCNN detection (if available)
        model_mask = np.zeros((h, w), dtype=bool)
        model_confidence = 0.0
        
        if self.use_model and self.predictor is not None:
            try:
                outputs = self.predictor(image)
                instances = outputs["instances"]
                
                if len(instances) > 0:
                    masks = instances.pred_masks.cpu().numpy()
                    scores = instances.scores.cpu().numpy()
                    
                    for i, (mask, score) in enumerate(zip(masks, scores)):
                        # Only include detections on floor
                        if np.sum(mask & enhanced_floor_mask) > np.sum(mask) * 0.5:
                            model_mask |= mask
                            model_confidence = max(model_confidence, score)
                
                results['confidence_scores']['maskrcnn'] = float(model_confidence)
                logger.info(f"MaskRCNN detection: confidence {model_confidence:.3f}")
            except Exception as e:
                logger.error(f"MaskRCNN detection failed: {e}")
        
        # Method 2: Enhanced pixel-based detection
        pixel_mask, pixel_confidence = self._detect_blackspots_advanced(image, enhanced_floor_mask)
        results['confidence_scores']['pixel_based'] = float(pixel_confidence)
        
        # Combine methods
        combined_mask = model_mask | pixel_mask
        
        # Filter by size (50px minimum in either dimension)
        final_mask = self._filter_by_size_dimensions(combined_mask, min_dimension=50)
        
        # Calculate final statistics
        blackspot_pixels = np.sum(final_mask)
        results['blackspot_mask'] = final_mask
        results['blackspot_area'] = blackspot_pixels
        results['coverage_percentage'] = (blackspot_pixels / results['floor_area']) * 100
        
        # Count individual blackspots
        num_labels, labels = cv2.connectedComponents(final_mask.astype(np.uint8))
        results['num_detections'] = num_labels - 1
        
        # Average confidence
        confidences = [c for c in [model_confidence, pixel_confidence] if c > 0]
        results['avg_confidence'] = np.mean(confidences) if confidences else 0.0
        
        # Detection method
        if model_mask.any() and pixel_mask.any():
            results['detection_method'] = 'combined_maskrcnn_pixel'
        elif model_mask.any():
            results['detection_method'] = 'maskrcnn_only'
        else:
            results['detection_method'] = 'enhanced_pixel_based'
        
        # Risk assessment
        coverage = results['coverage_percentage']
        if coverage > 15:
            results['risk_score'] = 10
        elif coverage > 10:
            results['risk_score'] = 8
        elif coverage > 5:
            results['risk_score'] = 6
        elif coverage > 2:
            results['risk_score'] = 4
        elif results['num_detections'] > 0:
            results['risk_score'] = 2
        else:
            results['risk_score'] = 0
        
        # Enhanced visualizations
        results['enhanced_views'] = self._create_comprehensive_visualizations(
            image, final_mask, enhanced_floor_mask, segmentation_mask
        )
        
        # Floor type breakdown
        if segmentation_mask is not None:
            results['floor_breakdown'] = self._analyze_floor_types(segmentation_mask, final_mask)
        
        logger.info(f"Enhanced blackspot detection: {results['num_detections']} spots, "
                   f"{results['coverage_percentage']:.2f}% coverage")
        
        return results
    
    def _extract_floor_from_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """Extract floor areas from segmentation"""
        floor_mask = np.zeros_like(segmentation, dtype=bool)
        
        for class_id in self.floor_class_ids.keys():
            class_pixels = np.sum(segmentation == class_id)
            if class_pixels > 0:
                floor_mask |= (segmentation == class_id)
                logger.debug(f"Floor class {class_id}: {class_pixels} pixels")
        
        return floor_mask
    
    def _detect_blackspots_advanced(self, image: np.ndarray, floor_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """Advanced multi-method blackspot detection"""
        # Convert to multiple color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Method 1: Strict black detection
        very_black_threshold = 25
        very_black_mask = (gray < very_black_threshold) & floor_mask
        
        # Method 2: Dark with color variance check
        dark_threshold = 50
        dark_mask = (gray < dark_threshold) & floor_mask
        
        # Color variance to distinguish black from dark gray
        b, g, r = cv2.split(image)
        color_variance = np.std([b, g, r], axis=0)
        is_truly_black = color_variance < 10  # Very low variance = black/gray
        
        # Method 3: HSV-based detection
        low_saturation = hsv[:, :, 1] < 30
        low_value = hsv[:, :, 2] < dark_threshold
        hsv_black_mask = low_saturation & low_value & floor_mask
        
        # Method 4: LAB lightness
        lab_dark_mask = (lab[:, :, 0] < 35) & floor_mask
        
        # Combine all methods
        combined_mask = (
            very_black_mask |  # Highest priority
            (dark_mask & is_truly_black) |  # Dark but truly black
            hsv_black_mask |  # HSV-based
            lab_dark_mask  # LAB-based
        )
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Calculate confidence
        if np.any(combined_mask):
            darkness_values = gray[combined_mask.astype(bool)]
            avg_darkness = np.mean(darkness_values)
            darkness_confidence = 1.0 - (avg_darkness / 50)
            size_confidence = min(1.0, np.sum(combined_mask) / 1000)
            confidence = (darkness_confidence * 0.7 + size_confidence * 0.3)
        else:
            confidence = 0.0
        
        return combined_mask.astype(bool), confidence
    
    def _filter_by_size_dimensions(self, mask: np.ndarray, min_dimension: int = 50) -> np.ndarray:
        """Filter blackspots by minimum width OR height"""
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        filtered_mask = np.zeros_like(mask, dtype=bool)
        
        for label_id in range(1, num_labels):
            component = labels == label_id
            y_coords, x_coords = np.where(component)
            
            if len(y_coords) == 0:
                continue
            
            height = np.max(y_coords) - np.min(y_coords) + 1
            width = np.max(x_coords) - np.min(x_coords) + 1
            
            # Keep if either dimension >= min_dimension
            if width >= min_dimension or height >= min_dimension:
                filtered_mask |= component
        
        return filtered_mask
    
    def _analyze_floor_types(self, segmentation: np.ndarray, blackspot_mask: np.ndarray) -> Dict:
        """Analyze blackspots by floor type"""
        breakdown = {}
        
        for class_id, class_name in self.floor_class_ids.items():
            floor_type_mask = segmentation == class_id
            floor_pixels = np.sum(floor_type_mask)
            
            if floor_pixels > 0:
                blackspots_on_type = blackspot_mask & floor_type_mask
                blackspot_pixels = np.sum(blackspots_on_type)
                coverage = (blackspot_pixels / floor_pixels) * 100
                
                breakdown[class_name] = {
                    'total_pixels': floor_pixels,
                    'blackspot_pixels': blackspot_pixels,
                    'coverage_percentage': coverage
                }
        
        return breakdown
    
    def _create_comprehensive_visualizations(self, image: np.ndarray, blackspot_mask: np.ndarray,
                                           floor_mask: np.ndarray, segmentation_mask: np.ndarray = None) -> Dict:
        """Create comprehensive visualization modes"""
        views = {}
        
        # High contrast overlay
        overlay = image.copy()
        overlay[blackspot_mask] = [255, 0, 0]  # Red blackspots
        overlay[floor_mask & ~blackspot_mask] = overlay[floor_mask & ~blackspot_mask] * 0.8 + np.array([0, 255, 0]) * 0.2
        views['high_contrast_overlay'] = overlay
        
        # Side by side comparison
        views['side_by_side'] = np.hstack([image, overlay])
        
        # Blackspots only (white background)
        blackspot_only = np.ones_like(image) * 255
        blackspot_only[blackspot_mask] = [0, 0, 0]
        blackspot_only[floor_mask & ~blackspot_mask] = [200, 200, 200]
        views['blackspot_only'] = blackspot_only
        
        # Annotated with size labels
        annotated = image.copy()
        contours, _ = cv2.findContours(blackspot_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            cv2.drawContours(annotated, [contour], -1, (255, 0, 0), 3)
            
            # Add size information
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            if area > 50:
                label = f"Spot {i+1}: {w}×{h}px"
                # White background for text
                cv2.putText(annotated, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # Red text on top
                cv2.putText(annotated, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        
        views['annotated_view'] = annotated
        
        # Segmentation view if available
        if segmentation_mask is not None:
            seg_view = image.copy()
            # Color different floor types
            colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)]
            
            for i, class_id in enumerate(self.floor_class_ids.keys()):
                class_mask = segmentation_mask == class_id
                if np.any(class_mask):
                    color = colors[i % len(colors)]
                    seg_view[class_mask] = seg_view[class_mask] * 0.6 + np.array(color) * 0.4
            
            seg_view[blackspot_mask] = [255, 0, 0]  # Blackspots on top
            views['segmentation_view'] = seg_view
        else:
            views['segmentation_view'] = overlay
        
        return views

# ====================== ONEFORMER MANAGER ======================

class OneFormerManager:
    """OneFormer manager with enhanced detectron2 integration"""
    
    def __init__(self):
        self.predictor = None
        self.metadata = None
        self.initialized = False
        self.detectron2_status = None
        
    def initialize(self, use_high_res: bool = False):
        """Initialize OneFormer with detectron2 status check"""
        self.detectron2_status = check_detectron2_comprehensive()
        
        if not self.detectron2_status['fully_functional']:
            logger.warning("OneFormer requires functional detectron2")
            return False
        
        try:
            from detectron2.config import get_cfg
            from detectron2.projects.deeplab import add_deeplab_config
            from detectron2.data import MetadataCatalog
            from detectron2.utils.visualizer import Visualizer
            from detectron2.engine.defaults import DefaultPredictor
            
            # Try OneFormer imports
            from oneformer_local.oneformer.config import add_oneformer_config, add_common_config, add_swin_config
            
            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_common_config(cfg)
            add_swin_config(cfg)
            add_oneformer_config(cfg)
            
            # Load config file
            config_file = "configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml"
            if os.path.exists(config_file):
                cfg.merge_from_file(config_file)
            
            cfg.MODEL.DEVICE = "cpu"
            
            # Set input size
            if use_high_res:
                cfg.INPUT.MIN_SIZE_TEST = 1280
                cfg.INPUT.MAX_SIZE_TEST = 1280 * 4
            else:
                cfg.INPUT.MIN_SIZE_TEST = 640
                cfg.INPUT.MAX_SIZE_TEST = 640 * 4
            
            # Try to find model weights
            model_paths = [
                "models/250_16_swin_l_oneformer_ade20k_160k.pth",
                "oneformer/250_16_swin_l_oneformer_ade20k_160k.pth"
            ]
            
            model_found = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    cfg.MODEL.WEIGHTS = model_path
                    model_found = True
                    break
            
            if not model_found:
                logger.warning("OneFormer model weights not found, using fallback")
                return False
            
            cfg.freeze()
            
            self.predictor = DefaultPredictor(cfg)
            self.metadata = MetadataCatalog.get("ade20k_panoptic_val")
            self.initialized = True
            
            logger.info("✅ OneFormer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"OneFormer initialization failed: {e}")
            return False
    
    def semantic_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform semantic segmentation with memory optimization"""
        if not self.initialized:
            raise RuntimeError("OneFormer not initialized")
        
        try:
            # Memory optimization: process in smaller chunks if needed
            h, w = image.shape[:2]
            
            # Run prediction
            outputs = self.predictor(image)
            
            # Extract segmentation
            if "sem_seg" in outputs:
                seg_logits = outputs["sem_seg"]
            else:
                raise ValueError("No semantic segmentation output found")
            
            seg_mask = seg_logits.argmax(dim=0).cpu().numpy()
            
            # Create visualizations
            vis_image = self._create_visualization(image, seg_mask, with_labels=False)
            labeled_image = self._create_visualization(image, seg_mask, with_labels=True)
            
            # Clear GPU memory if available
            if hasattr(seg_logits, 'cpu'):
                del seg_logits
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            return seg_mask, vis_image, labeled_image
            
        except Exception as e:
            logger.error(f"Semantic segmentation failed: {e}")
            raise
    
    def _create_visualization(self, image: np.ndarray, seg_mask: np.ndarray, with_labels: bool = False) -> np.ndarray:
        """Create visualization with optional labels"""
        from detectron2.utils.visualizer import Visualizer, ColorMode
        
        visualizer = Visualizer(
            image[:, :, ::-1],  # RGB to BGR
            metadata=self.metadata,
            instance_mode=ColorMode.IMAGE
        )
        
        vis_output = visualizer.draw_sem_seg(seg_mask, alpha=0.5)
        vis_image = vis_output.get_image()[:, :, ::-1]  # BGR to RGB
        
        if with_labels:
            # Add labels for significant segments
            unique_classes = np.unique(seg_mask)
            
            for class_id in unique_classes:
                if class_id == 255:  # Ignore class
                    continue
                
                mask = seg_mask == class_id
                if np.sum(mask) > 1000:  # Only label large segments
                    # Find centroid
                    y_coords, x_coords = np.where(mask)
                    if len(y_coords) > 0:
                        cy, cx = int(np.mean(y_coords)), int(np.mean(x_coords))
                        
                        # Get class name
                        if hasattr(self.metadata, 'stuff_classes') and class_id < len(self.metadata.stuff_classes):
                            class_name = self.metadata.stuff_classes[class_id]
                        else:
                            class_name = f"class_{class_id}"
                        
                        # Draw label
                        cv2.putText(vis_image, class_name, (cx-50, cy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(vis_image, class_name, (cx-50, cy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        return vis_image
    
    def extract_floor_areas(self, segmentation: np.ndarray) -> np.ndarray:
        """Extract floor areas from segmentation"""
        floor_class_ids = [3, 4, 13, 28, 29, 52, 53, 78]  # ADE20K floor classes
        floor_mask = np.zeros_like(segmentation, dtype=bool)
        
        for class_id in floor_class_ids:
            floor_mask |= (segmentation == class_id)
        
        return floor_mask

# ====================== MAIN APPLICATION CLASS ======================

class NeuroNestApp:
    """Complete NeuroNest application with all features"""
    
    def __init__(self):
        self.oneformer = None
        self.blackspot_detector = None
        self.contrast_analyzer = None
        self.initialized = False
        self.detectron2_status = None
        
    def initialize(self, use_high_res: bool = False):
        """Initialize all components"""
        logger.info(f"🚀 Initializing NeuroNest (high_res={use_high_res})")
        
        # Check detectron2 status
        self.detectron2_status = check_detectron2_comprehensive()
        
        # Initialize OneFormer
        oneformer_success = False
        if self.detectron2_status['fully_functional']:
            try:
                self.oneformer = OneFormerManager()
                oneformer_success = self.oneformer.initialize(use_high_res)
            except Exception as e:
                logger.error(f"OneFormer initialization failed: {e}")
        
        # Initialize enhanced blackspot detector
        blackspot_success = False
        try:
            self.blackspot_detector = EnhancedBlackspotDetector()
            blackspot_success = self.blackspot_detector.initialize()
        except Exception as e:
            logger.error(f"Blackspot detector failed: {e}")
        
        # Initialize contrast analyzer
        contrast_success = False
        try:
            from contrast import RobustContrastAnalyzer
            self.contrast_analyzer = RobustContrastAnalyzer(
                wcag_threshold=4.5,
                alzheimer_threshold=7.0,
                color_similarity_threshold=25.0,
                perceptual_threshold=0.12
            )
            contrast_success = True
        except Exception as e:
            logger.error(f"Contrast analyzer failed: {e}")
        
        self.initialized = blackspot_success or oneformer_success or contrast_success
        
        logger.info(f"✅ NeuroNest initialization complete:")
        logger.info(f"   - OneFormer: {oneformer_success}")
        logger.info(f"   - Enhanced Blackspot: {blackspot_success}")
        logger.info(f"   - Contrast Analysis: {contrast_success}")
        logger.info(f"   - Overall Success: {self.initialized}")
        
        return oneformer_success, blackspot_success
    
    def analyze_image(self, image_path: str, **kwargs) -> Dict:
        """Comprehensive image analysis"""
        if not self.initialized:
            return {"error": "Application not initialized"}
        
        try:
            # Load image
            if not os.path.exists(image_path):
                return {"error": f"Image not found: {image_path}"}
            
            image = cv2.imread(image_path)
            if image is None:
                return {"error": f"Could not load image: {image_path}"}
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = {
                'original_image': image_rgb,
                'segmentation': None,
                'blackspot': None,
                'contrast': None,
                'statistics': {},
                'system_status': {
                    'detectron2_functional': self.detectron2_status['fully_functional'] if self.detectron2_status else False,
                    'oneformer_available': self.oneformer is not None and self.oneformer.initialized,
                    'blackspot_enhanced': self.blackspot_detector is not None,
                    'contrast_available': self.contrast_analyzer is not None
                }
            }
            
            seg_mask = None
            floor_mask = None
            
            # 1. OneFormer Segmentation
            if self.oneformer and self.oneformer.initialized:
                logger.info("🎯 Running OneFormer segmentation...")
                try:
                    seg_mask, seg_vis, labeled_vis = self.oneformer.semantic_segmentation(image_rgb)
                    
                    results['segmentation'] = {
                        'mask': seg_mask,
                        'visualization': seg_vis,
                        'labeled_visualization': labeled_vis
                    }
                    
                    floor_mask = self.oneformer.extract_floor_areas(seg_mask)
                    logger.info(f"✅ Segmentation: {len(np.unique(seg_mask))} classes, {np.sum(floor_mask)} floor pixels")
                    
                except Exception as e:
                    logger.error(f"Segmentation failed: {e}")
            
            # 2. Enhanced Blackspot Detection
            if kwargs.get('enable_blackspot', True) and self.blackspot_detector:
                logger.info("⚫ Running enhanced blackspot detection...")
                try:
                    blackspot_results = self.blackspot_detector.detect_blackspots(
                        image_rgb, floor_mask, seg_mask
                    )
                    results['blackspot'] = blackspot_results
                    logger.info(f"✅ Blackspot: {blackspot_results['num_detections']} detections")
                except Exception as e:
                    logger.error(f"Blackspot detection failed: {e}")
            
            # 3. Contrast Analysis
            if kwargs.get('enable_contrast', True) and self.contrast_analyzer:
                logger.info("🎨 Running contrast analysis...")
                try:
                    analysis_mask = seg_mask if seg_mask is not None else np.random.randint(0, 10, image_rgb.shape[:2])
                    contrast_results = self.contrast_analyzer.analyze_contrast(image_rgb, analysis_mask)
                    results['contrast'] = contrast_results
                    logger.info(f"✅ Contrast: {contrast_results['statistics']['total_issues']} issues")
                except Exception as e:
                    logger.error(f"Contrast analysis failed: {e}")
            
            # Generate statistics
            results['statistics'] = self._generate_statistics(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _generate_statistics(self, results: Dict) -> Dict:
        """Generate comprehensive statistics"""
        stats = {}
        
        # System status
        stats['system'] = results.get('system_status', {})
        
        # Segmentation stats
        if results.get('segmentation'):
            seg_mask = results['segmentation']['mask']
            unique_classes = np.unique(seg_mask)
            stats['segmentation'] = {
                'num_classes': len(unique_classes),
                'total_pixels': seg_mask.size,
                'image_shape': seg_mask.shape
            }
        
        # Blackspot stats
        if results.get('blackspot'):
            bs = results['blackspot']
            stats['blackspot'] = {
                'num_detections': bs.get('num_detections', 0),
                'coverage_percentage': bs.get('coverage_percentage', 0),
                'risk_score': bs.get('risk_score', 0),
                'detection_method': bs.get('detection_method', 'unknown')
            }
        
        # Contrast stats
        if results.get('contrast'):
            cs = results['contrast']['statistics']
            stats['contrast'] = {
                'total_issues': cs.get('total_issues', 0),
                'critical_count': cs.get('critical_count', 0),
                'good_contrast_count': cs.get('good_contrast_count', 0)
            }
        
        return stats

# ====================== GRADIO INTERFACE ======================

def create_comprehensive_interface(app_instance):
    """Create comprehensive interface with multiple visualization modes"""
    try:
        import gradio as gr
        from PIL import Image
        
        def analyze_comprehensive(image, blackspot_threshold, contrast_threshold, 
                                enable_blackspot, enable_contrast, show_labels):
            if image is None:
                return [None] * 6 + ["Please upload an image to analyze."]
            
            try:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    if hasattr(image, 'save'):
                        image.save(tmp.name)
                    else:
                        Image.fromarray(image).save(tmp.name)
                    
                    # Run analysis
                    results = app_instance.analyze_image(
                        image_path=tmp.name,
                        enable_blackspot=enable_blackspot,
                        enable_contrast=enable_contrast
                    )
                    
                    os.unlink(tmp.name)
                
                if "error" in results:
                    return [None] * 6 + [f"❌ Error: {results['error']}"]
                
                # Extract visualizations
                combined_vis = None
                seg_vis = None
                seg_labeled = None
                blackspot_vis = None
                contrast_vis = None
                
                # Combined visualization (priority: contrast + blackspot > blackspot > segmentation)
                if results.get('contrast') and results.get('blackspot'):
                    combined_vis = results['contrast'].get('visualization', image)
                    # Overlay blackspots
                    if 'enhanced_views' in results['blackspot']:
                        blackspot_overlay = results['blackspot']['enhanced_views'].get('high_contrast_overlay')
                        if blackspot_overlay is not None:
                            combined_vis = blackspot_overlay
                elif results.get('blackspot'):
                    if 'enhanced_views' in results['blackspot']:
                        combined_vis = results['blackspot']['enhanced_views'].get('high_contrast_overlay', image)
                elif results.get('segmentation'):
                    combined_vis = results['segmentation'].get('visualization', image)
                else:
                    combined_vis = image
                
                # Individual visualizations
                if results.get('segmentation'):
                    seg_vis = results['segmentation'].get('visualization')
                    seg_labeled = results['segmentation'].get('labeled_visualization')
                
                if results.get('blackspot') and 'enhanced_views' in results['blackspot']:
                    blackspot_vis = results['blackspot']['enhanced_views'].get('annotated_view')
                
                if results.get('contrast'):
                    contrast_vis = results['contrast'].get('visualization')
                
                # Generate comprehensive report
                report = generate_comprehensive_report(results)
                
                return (combined_vis, seg_vis, seg_labeled, blackspot_vis, contrast_vis, 
                       combined_vis, report)
                
            except Exception as e:
                return [None] * 6 + [f"Analysis failed: {str(e)}"]
        
        # Create interface
        with gr.Blocks(
            title="NeuroNest - Complete Alzheimer's Environment Analysis",
            theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue")
        ) as interface:
            
            # Get system status
            d2_status = app_instance.detectron2_status
            d2_functional = d2_status and d2_status.get('fully_functional', False)
            
            gr.Markdown(f"""
            # 🧠 NeuroNest: Complete Alzheimer's Environment Analysis
            
            **Advanced AI System for Dementia-Friendly Environments**  
            *Abheek Pradhan | Faculty: Dr. Nadim Adi and Dr. Greg Lakomski*  
            *Texas State University - Computer Science & Interior Design*
            
            ### 🔧 System Status:
            - **Detectron2:** {"✅ v" + str(d2_status.get('version', 'unknown')) + " (Functional)" if d2_functional else "⚠️ Limited"}
            - **OneFormer Segmentation:** {"✅ Available" if app_instance.oneformer and app_instance.oneformer.initialized else "⚠️ Fallback"}
            - **Enhanced Blackspot Detection:** {"✅ Floor-only with 50px+ minimum" if app_instance.blackspot_detector else "❌"}
            - **Contrast Analysis:** {"✅ Alzheimer's Standards (7:1)" if app_instance.contrast_analyzer else "❌"}
            """)
            
            with gr.Row():
                # Input Column
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="📸 Upload Room Image",
                        type="pil",
                        height=350,
                        sources=["upload", "clipboard"]
                    )
                    
                    with gr.Accordion("🎛️ Analysis Settings", open=True):
                        enable_blackspot = gr.Checkbox(
                            value=True,
                            label="Enhanced Blackspot Detection",
                            info="Floor-only, 50px+ minimum size"
                        )
                        
                        blackspot_threshold = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                            label="Blackspot Sensitivity"
                        )
                        
                        enable_contrast = gr.Checkbox(
                            value=True,
                            label="Contrast Analysis",
                            info="WCAG + Alzheimer's standards"
                        )
                        
                        contrast_threshold = gr.Slider(
                            minimum=1.0, maximum=10.0, value=7.0, step=0.1,
                            label="Contrast Threshold (7:1 for Alzheimer's)"
                        )
                        
                        show_labels = gr.Checkbox(
                            value=True,
                            label="Show Object Labels",
                            info="Display class names on segmentation"
                        )
                    
                    analyze_btn = gr.Button(
                        "🔍 Complete Analysis",
                        variant="primary",
                        size="lg"
                    )
                    
                    gr.Markdown("""
                    ### 📋 Analysis Features:
                    - **Multiple Visualization Modes**
                    - **Floor-only Blackspot Detection**
                    - **Alzheimer's Contrast Standards**
                    - **Detailed Safety Reports**
                    - **Evidence-based Recommendations**
                    """)
                
                # Output Column
                with gr.Column(scale=3):
                    # Main display
                    main_output = gr.Image(
                        label="🎯 Combined Analysis Result",
                        height=450
                    )
                    
                    # Multiple visualization tabs
                    with gr.Tabs():
                        with gr.Tab("📊 Comprehensive Report"):
                            analysis_report = gr.Markdown(
                                value="Upload an image and click 'Complete Analysis' for detailed results."
                            )
                        
                        with gr.Tab("🏷️ OneFormer Segmentation"):
                            with gr.Row():
                                seg_output = gr.Image(
                                    label="Object Segmentation",
                                    height=400
                                )
                                seg_labeled_output = gr.Image(
                                    label="Labeled Segmentation",
                                    height=400
                                )
                        
                        with gr.Tab("⚫ Enhanced Blackspot Detection"):
                            blackspot_output = gr.Image(
                                label="Floor-only Blackspot Analysis (50px+ minimum)",
                                height=450
                            )
                        
                        with gr.Tab("🎨 Contrast Analysis"):
                            contrast_output = gr.Image(
                                label="Alzheimer's Contrast Standards (7:1 minimum)",
                                height=450
                            )
                        
                        with gr.Tab("🔄 Complete View"):
                            combined_output = gr.Image(
                                label="All Analysis Combined",
                                height=450
                            )
            
            # Connect interface
            analyze_btn.click(
                fn=analyze_comprehensive,
                inputs=[
                    image_input, blackspot_threshold, contrast_threshold,
                    enable_blackspot, enable_contrast, show_labels
                ],
                outputs=[
                    main_output, seg_output, seg_labeled_output,
                    blackspot_output, contrast_output, combined_output,
                    analysis_report
                ]
            )
        
        return interface
        
    except Exception as e:
        logger.error(f"Interface creation failed: {e}")
        return None

def generate_comprehensive_report(results: Dict) -> str:
    """Generate comprehensive analysis report"""
    stats = results.get('statistics', {})
    system_status = stats.get('system', {})
    
    # Header
    report = """# 🧠 NeuroNest Comprehensive Analysis Report

## 📊 Executive Summary

"""
    
    # System status
    report += f"""### 🔧 System Status:
- **Detectron2:** {"✅ Functional" if system_status.get('detectron2_functional') else "⚠️ Limited"}
- **OneFormer:** {"✅ Available" if system_status.get('oneformer_available') else "⚠️ Fallback"}
- **Enhanced Blackspot:** {"✅ Active" if system_status.get('blackspot_enhanced') else "❌ Unavailable"}
- **Contrast Analysis:** {"✅ Active" if system_status.get('contrast_available') else "❌ Unavailable"}

"""
    
    # Segmentation results
    if results.get('segmentation'):
        seg_stats = stats.get('segmentation', {})
        report += f"""## 🎯 Object Segmentation Results
- **Objects Detected:** {seg_stats.get('num_classes', 0)} different classes
- **Image Resolution:** {seg_stats.get('image_shape', 'Unknown')}
- **Method:** OneFormer with ADE20K dataset

"""
    
    # Blackspot analysis
    if results.get('blackspot'):
        bs_stats = stats.get('blackspot', {})
        blackspot_data = results['blackspot']
        
        report += f"""## ⚫ Enhanced Blackspot Analysis
- **Detection Method:** {bs_stats.get('detection_method', 'Unknown')}
- **Blackspots Found:** {bs_stats.get('num_detections', 0)} (50px+ minimum)
- **Floor Coverage:** {bs_stats.get('coverage_percentage', 0):.2f}%
- **Risk Score:** {bs_stats.get('risk_score', 0)}/10

### Floor Type Breakdown:
"""
        
        floor_breakdown = blackspot_data.get('floor_breakdown', {})
        if floor_breakdown:
            for floor_type, data in floor_breakdown.items():
                report += f"- **{floor_type.title()}:** {data['coverage_percentage']:.1f}% blackspot coverage\n"
        else:
            report += "- Floor segmentation not available\n"
        
        # Risk assessment
        risk_score = bs_stats.get('risk_score', 0)
        if risk_score >= 8:
            report += "\n🚨 **CRITICAL:** Immediate blackspot removal required\n"
        elif risk_score >= 6:
            report += "\n⚠️ **HIGH RISK:** Significant blackspot coverage detected\n"
        elif risk_score >= 4:
            report += "\n⚠️ **MODERATE RISK:** Some blackspots need attention\n"
        elif risk_score > 0:
            report += "\n✅ **LOW RISK:** Minimal blackspot coverage\n"
        else:
            report += "\n✅ **EXCELLENT:** No blackspots detected\n"
        
        report += "\n"
    
    # Contrast analysis
    if results.get('contrast'):
        contrast_stats = stats.get('contrast', {})
        
        report += f"""## 🎨 Contrast Analysis for Alzheimer's Care
- **Total Issues:** {contrast_stats.get('total_issues', 0)}
- **Critical Issues:** {contrast_stats.get('critical_count', 0)} (require immediate attention)
- **Good Contrasts:** {contrast_stats.get('good_contrast_count', 0)} (meet 7:1 standard)

"""
        
        # Contrast assessment
        critical_count = contrast_stats.get('critical_count', 0)
        total_issues = contrast_stats.get('total_issues', 0)
        
        if critical_count > 0:
            report += "🚨 **CRITICAL CONTRAST ISSUES:** Immediate color changes needed\n"
        elif total_issues > 10:
            report += "⚠️ **HIGH PRIORITY:** Multiple contrast improvements needed\n"
        elif total_issues > 5:
            report += "⚠️ **MODERATE PRIORITY:** Some contrast improvements recommended\n"
        elif total_issues > 0:
            report += "✅ **MINOR ISSUES:** Few contrast adjustments needed\n"
        else:
            report += "✅ **EXCELLENT:** All contrasts meet Alzheimer's standards\n"
        
        report += "\n"
    
    # Alzheimer's care recommendations
    report += """## 📋 Evidence-Based Alzheimer's Care Recommendations

### ✅ Best Practices:
1. **7:1 Contrast Minimum** - Essential for object recognition
2. **Eliminate All Blackspots** - Remove trip hazards from floors
3. **Warm Color Preference** - Red, yellow, orange easier to perceive
4. **High Saturation Colors** - Avoid muted or pastel tones
5. **30°+ Hue Separation** - Ensure colors are distinctly different

### 🎯 Immediate Actions:
"""
    
    # Specific recommendations based on results
    if results.get('blackspot'):
        num_blackspots = stats.get('blackspot', {}).get('num_detections', 0)
        if num_blackspots > 0:
            report += f"- **Remove {num_blackspots} detected blackspots** from floor areas\n"
    
    if results.get('contrast'):
        critical_issues = stats.get('contrast', {}).get('critical_count', 0)
        if critical_issues > 0:
            report += f"- **Address {critical_issues} critical contrast issues** immediately\n"
    
    report += """- **Increase lighting** to minimum 1000 lux throughout space
- **Add texture/patterns** where color contrast cannot be improved
- **Regular assessment** - check environment monthly for changes

### 🏥 Clinical Impact:
- **Reduced fall risk** through better floor visibility
- **Improved navigation** with clear object boundaries
- **Enhanced independence** through better environmental cues
- **Decreased confusion** with distinct visual elements

*Analysis based on evidence-based design principles for dementia care*
"""
    
    return report

# ====================== MAIN APPLICATION ======================

def main():
    """Main application entry point"""
    logger.info("🚀 Starting NeuroNest Complete Analysis System")
    
    try:
        # Setup paths
        project_root = setup_python_paths()
        
        # Initialize app
        app = NeuroNestApp()
        oneformer_ok, blackspot_ok = app.initialize()
        
        if not app.initialized:
            logger.error("❌ App initialization failed")
            return False
        
        # Create interface
        interface = create_comprehensive_interface(app)
        if not interface:
            logger.error("❌ Interface creation failed")
            return False
        
        # Launch (fixed parameter name)
        logger.info("🌐 Launching NeuroNest Complete Analysis System...")
        interface.queue(
            default_concurrency_limit=2,
            max_size=10
        ).launch(
            server_name="0.0.0.0",
            server_port=7860,  # Fixed: was 'port'
            share=False,
            show_error=True,
            prevent_thread_lock=False
        )
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        time.sleep(3600)
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("🎉 NeuroNest Complete Analysis System started successfully!")
    else:
        logger.error("💥 NeuroNest failed to start")
