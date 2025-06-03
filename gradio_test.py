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
# IMPROVED BLACKSPOT DETECTION
########################################

class ImprovedBlackspotDetector:
    """Enhanced blackspot detector that only detects on floor surfaces"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.predictor = None
        
        # Expanded floor-related classes in ADE20K
        self.floor_classes = [3, 4, 13, 28, 78]  # floor, wood floor, rug, carpet, mat
        
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
    
    def is_on_floor_surface(self, blackspot_mask: np.ndarray, segmentation: np.ndarray, 
                           floor_mask: np.ndarray, overlap_threshold: float = 0.8) -> bool:
        """Check if a blackspot is actually on a floor surface"""
        if np.sum(blackspot_mask) == 0:
            return False
        
        # Check overlap with floor mask
        overlap = blackspot_mask & floor_mask
        overlap_ratio = np.sum(overlap) / np.sum(blackspot_mask)
        
        if overlap_ratio < overlap_threshold:
            return False
        
        # Additional check: verify the underlying segmentation class
        blackspot_pixels = segmentation[blackspot_mask]
        if len(blackspot_pixels) == 0:
            return False
        
        # Check if majority of pixels are floor-related classes
        unique_classes, counts = np.unique(blackspot_pixels, return_counts=True)
        floor_pixel_count = sum(counts[unique_classes == cls] for cls in self.floor_classes 
                               if cls in unique_classes)
        floor_ratio = floor_pixel_count / len(blackspot_pixels)
        
        return floor_ratio > 0.7  # At least 70% of blackspot should be on floor classes
    
    def filter_non_floor_blackspots(self, blackspot_masks: List[np.ndarray], 
                                   segmentation: np.ndarray, floor_mask: np.ndarray) -> List[np.ndarray]:
        """Filter out blackspots that are not on floor surfaces"""
        filtered_masks = []
        
        for mask in blackspot_masks:
            if self.is_on_floor_surface(mask, segmentation, floor_mask):
                filtered_masks.append(mask)
            else:
                logger.debug(f"Filtered out non-floor blackspot with area {np.sum(mask)}")
        
        return filtered_masks
    
    def detect_blackspots(self, image: np.ndarray, segmentation: np.ndarray, 
                         floor_prior: Optional[np.ndarray] = None) -> Dict:
        """Detect blackspots only on floor surfaces"""
        if self.predictor is None:
            raise RuntimeError("Blackspot detector not initialized")
        
        # Get original image dimensions
        original_h, original_w = image.shape[:2]
        
        # Ensure all masks have same dimensions
        if floor_prior is not None and floor_prior.shape != (original_h, original_w):
            floor_prior = cv2.resize(
                floor_prior.astype(np.uint8), 
                (original_w, original_h), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        
        if segmentation.shape != (original_h, original_w):
            segmentation = cv2.resize(
                segmentation.astype(np.uint8),
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Run detection
        try:
            outputs = self.predictor(image)
            instances = outputs["instances"].to("cpu")
        except Exception as e:
            logger.error(f"Error in MaskRCNN prediction: {e}")
            return self._empty_results(image)
        
        if len(instances) == 0:
            return self._empty_results(image)
        
        # Process results
        pred_classes = instances.pred_classes.numpy()
        pred_masks = instances.pred_masks.numpy()
        scores = instances.scores.numpy()
        
        # Separate floor and blackspot masks
        blackspot_indices = pred_classes == 1
        blackspot_masks = pred_masks[blackspot_indices] if np.any(blackspot_indices) else []
        blackspot_scores = scores[blackspot_indices] if np.any(blackspot_indices) else []
        
        # Create or use floor mask
        if floor_prior is not None:
            floor_mask = floor_prior
        else:
            # Create floor mask from segmentation
            floor_mask = np.zeros(segmentation.shape, dtype=bool)
            for cls in self.floor_classes:
                floor_mask |= (segmentation == cls)
        
        # Filter blackspots to only those on floor surfaces
        filtered_blackspot_masks = self.filter_non_floor_blackspots(
            blackspot_masks, segmentation, floor_mask
        )
        
        # Combine filtered masks
        combined_blackspot = np.zeros(image.shape[:2], dtype=bool)
        for mask in filtered_blackspot_masks:
            combined_blackspot |= mask
        
        # Create visualizations
        visualization = self.create_visualization(image, floor_mask, combined_blackspot)
        
        # Calculate statistics
        floor_area = int(np.sum(floor_mask))
        blackspot_area = int(np.sum(combined_blackspot))
        coverage_percentage = (blackspot_area / floor_area * 100) if floor_area > 0 else 0
        
        return {
            'visualization': visualization,
            'floor_mask': floor_mask,
            'blackspot_mask': combined_blackspot,
            'floor_area': floor_area,
            'blackspot_area': blackspot_area,
            'coverage_percentage': coverage_percentage,
            'num_detections': len(filtered_blackspot_masks),
            'avg_confidence': float(np.mean(blackspot_scores)) if len(blackspot_scores) > 0 else 0.0
        }
    
    def create_visualization(self, image: np.ndarray, floor_mask: np.ndarray, 
                           blackspot_mask: np.ndarray) -> np.ndarray:
        """Create clear visualization of blackspots on floors only"""
        vis = image.copy()
        
        # Semi-transparent green overlay for floors
        floor_overlay = vis.copy()
        floor_overlay[floor_mask] = [0, 255, 0]
        vis = cv2.addWeighted(vis, 0.7, floor_overlay, 0.3, 0)
        
        # Bright red for blackspots
        vis[blackspot_mask] = [255, 0, 0]
        
        # Add contours for clarity
        blackspot_contours, _ = cv2.findContours(
            blackspot_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, blackspot_contours, -1, (255, 255, 0), 2)
        
        return vis
    
    def _empty_results(self, image: np.ndarray) -> Dict:
        """Return empty results structure"""
        empty_mask = np.zeros(image.shape[:2], dtype=bool)
        return {
            'visualization': image,
            'floor_mask': empty_mask,
            'blackspot_mask': empty_mask,
            'floor_area': 0,
            'blackspot_area': 0,
            'coverage_percentage': 0,
            'num_detections': 0,
            'avg_confidence': 0.0
        }


########################################
# MAIN APPLICATION CLASS
########################################

class NeuroNestApp:
    """Main application class integrating all components"""
    
    def __init__(self):
        self.oneformer = OneFormerManager()
        self.blackspot_detector = None
        self.contrast_analyzer = UniversalContrastAnalyzer(wcag_threshold=4.5)
        self.initialized = False
    
    def initialize(self, blackspot_model_path: str = "./output_floor_blackspot/model_0004999.pth"):
        """Initialize all components"""
        logger.info("Initializing NeuroNest application...")
        
        # Initialize OneFormer
        oneformer_success = self.oneformer.initialize()
        
        # Initialize blackspot detector if model exists
        blackspot_success = False
        if os.path.exists(blackspot_model_path):
            self.blackspot_detector = ImprovedBlackspotDetector(blackspot_model_path)
            blackspot_success = self.blackspot_detector.initialize()
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
        """Perform complete image analysis"""
        
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
            
            # 1. Semantic Segmentation
            logger.info("Running semantic segmentation...")
            seg_mask, seg_visualization = self.oneformer.semantic_segmentation(image_rgb)
            
            results['segmentation'] = {
                'visualization': seg_visualization,
                'mask': seg_mask
            }
            
            # Extract floor areas
            floor_prior = self.oneformer.extract_floor_areas(seg_mask)
            
            # 2. Blackspot Detection (improved to only detect on floors)
            if enable_blackspot and self.blackspot_detector is not None:
                logger.info("Running blackspot detection...")
                try:
                    # Resize segmentation mask to match original image if needed
                    h_orig, w_orig = image_rgb.shape[:2]
                    h_seg, w_seg = seg_mask.shape
                    
                    if (h_seg, w_seg) != (h_orig, w_orig):
                        seg_mask_resized = cv2.resize(
                            seg_mask.astype(np.uint8),
                            (w_orig, h_orig),
                            interpolation=cv2.INTER_NEAREST
                        )
                    else:
                        seg_mask_resized = seg_mask
                    
                    blackspot_results = self.blackspot_detector.detect_blackspots(
                        image_rgb, seg_mask_resized, floor_prior
                    )
                    results['blackspot'] = blackspot_results
                    logger.info("Blackspot detection completed")
                except Exception as e:
                    logger.error(f"Error in blackspot detection: {e}")
                    results['blackspot'] = None
            
            # 3. Universal Contrast Analysis
            if enable_contrast:
                logger.info("Running universal contrast analysis...")
                try:
                    # Resize image to match segmentation size
                    h_seg, w_seg = seg_mask.shape
                    image_for_contrast = cv2.resize(image_rgb, (w_seg, h_seg))
                    
                    contrast_results = self.contrast_analyzer.analyze_contrast(
                        image_for_contrast, seg_mask
                    )
                    results['contrast'] = contrast_results
                    logger.info("Contrast analysis completed")
                except Exception as e:
                    logger.error(f"Error in contrast analysis: {e}")
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
            stats['contrast'] = {
                'total_segments': cs.get('total_segments', 0),
                'analyzed_pairs': cs.get('analyzed_pairs', 0),
                'low_contrast_pairs': cs.get('low_contrast_pairs', 0),
                'critical_issues': cs.get('critical_issues', 0),
                'high_priority_issues': cs.get('high_priority_issues', 0),
                'medium_priority_issues': cs.get('medium_priority_issues', 0),
                'floor_object_issues': cs.get('floor_object_issues', 0)
            }
        
        return stats


########################################
# GRADIO INTERFACE
########################################

def create_gradio_interface():
    """Create the Gradio interface"""
    
    # Initialize the application
    app = NeuroNestApp()
    oneformer_ok, blackspot_ok = app.initialize()
    
    if not oneformer_ok:
        raise RuntimeError("Failed to initialize OneFormer")
    
    def analyze_wrapper(image_path, blackspot_threshold, contrast_threshold, 
                       enable_blackspot, enable_contrast):
        """Wrapper function for Gradio interface"""
        if image_path is None:
            return None, None, None, None, "Please upload an image"
        
        results = app.analyze_image(
            image_path=image_path,
            blackspot_threshold=blackspot_threshold,
            contrast_threshold=contrast_threshold,
            enable_blackspot=enable_blackspot,
            enable_contrast=enable_contrast
        )
        
        if "error" in results:
            return None, None, None, None, f"Error: {results['error']}"
        
        # Extract outputs
        seg_output = results['segmentation']['visualization'] if results['segmentation'] else None
        blackspot_output = results['blackspot']['visualization'] if results['blackspot'] else None
        contrast_output = results['contrast']['visualization'] if results['contrast'] else None
        
        # Generate universal contrast report
        if results['contrast']:
            contrast_report = app.contrast_analyzer.generate_report(results['contrast'])
        else:
            contrast_report = "Contrast analysis not performed."
        
        # Generate full report
        report = generate_comprehensive_report(results, contrast_report)
        
        return seg_output, blackspot_output, contrast_output, report
    
    def generate_comprehensive_report(results: Dict, contrast_report: str) -> str:
        """Generate comprehensive analysis report"""
        report = ["# 🧠 NeuroNest Analysis Report\n"]
        report.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        # Segmentation results
        if results['segmentation']:
            stats = results['statistics'].get('segmentation', {})
            report.append("## 🎯 Object Segmentation")
            report.append(f"- **Classes detected:** {stats.get('num_classes', 'N/A')}")
            report.append(f"- **Resolution:** {stats.get('image_size', 'N/A')}")
            report.append("")
        
        # Blackspot results
        if results['blackspot']:
            bs_stats = results['statistics'].get('blackspot', {})
            report.append("## ⚫ Blackspot Detection (Floor Surfaces Only)")
            report.append(f"- **Floor area:** {bs_stats.get('floor_area_pixels', 0):,} pixels")
            report.append(f"- **Blackspot area:** {bs_stats.get('blackspot_area_pixels', 0):,} pixels")
            report.append(f"- **Coverage:** {bs_stats.get('coverage_percentage', 0):.2f}% of floor")
            report.append(f"- **Detections:** {bs_stats.get('num_detections', 0)}")
            
            # Risk assessment
            coverage = bs_stats.get('coverage_percentage', 0)
            if coverage > 10:
                report.append("- **⚠️ Risk:** CRITICAL - Immediate intervention required")
            elif coverage > 5:
                report.append("- **⚠️ Risk:** HIGH - Significant fall hazard")
            elif coverage > 1:
                report.append("- **⚠️ Risk:** MEDIUM - Potential safety concern")
            elif coverage > 0:
                report.append("- **✓ Risk:** LOW - Minor concern")
            else:
                report.append("- **✓ Risk:** NONE - No blackspots detected")
            report.append("")
        
        # Universal contrast analysis
        report.append("## 🎨 Universal Contrast Analysis")
        report.append(contrast_report)
        report.append("")
        
        # Recommendations
        report.append("## 📋 Recommendations for Alzheimer's Care")
        
        has_issues = False
        
        if results['blackspot'] and results['statistics']['blackspot']['coverage_percentage'] > 0:
            has_issues = True
            report.append("\n### Blackspot Mitigation:")
            report.append("- Replace dark flooring materials with lighter alternatives")
            report.append("- Install additional lighting in affected areas")
            report.append("- Use light-colored rugs or runners to cover dark spots")
            report.append("- Add contrasting tape or markers around blackspot perimeters")
        
        if results['contrast'] and results['statistics']['contrast']['low_contrast_pairs'] > 0:
            has_issues = True
            report.append("\n### Contrast Improvements:")
            report.append("- Paint furniture in colors that contrast with floors/walls")
            report.append("- Add colored tape or markers to furniture edges")
            report.append("- Install LED strip lighting under furniture edges")
            report.append("- Use contrasting placemats, cushions, or covers")
        
        if not has_issues:
            report.append("\n✅ **Excellent!** This environment appears well-optimized for individuals with Alzheimer's.")
            report.append("No significant visual hazards detected.")
        
        return "\n".join(report)
    
    # Create the interface
    title = "🧠 NeuroNest: AI-Powered Environment Safety Analysis"
    description = """
    **Advanced visual analysis for Alzheimer's and dementia care environments**
    
    This system provides:
    - **Object Segmentation**: Identifies all room elements (floors, walls, furniture)
    - **Floor-Only Blackspot Detection**: Locates dangerous dark areas on walking surfaces
    - **Universal Contrast Analysis**: Evaluates visibility between ALL adjacent objects
    
    *Following WCAG 2.1 guidelines for visual accessibility*
    """
    
    with gr.Blocks(title=title, theme=gr.themes.Soft()) as interface:
        
        gr.Markdown(f"# {title}")
        gr.Markdown(description)
        
        with gr.Row():
            # Input Column
            with gr.Column(scale=1):
                # Image upload
                image_input = gr.Image(
                    label="📸 Upload Room Image",
                    type="filepath",
                    height=400
                )
                
                # Analysis settings
                with gr.Accordion("⚙️ Analysis Settings", open=True):
                    enable_blackspot = gr.Checkbox(
                        value=blackspot_ok,
                        label="Enable Floor Blackspot Detection",
                        interactive=blackspot_ok
                    )
                    
                    blackspot_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.05,
                        label="Detection Sensitivity",
                        visible=blackspot_ok
                    )
                    
                    enable_contrast = gr.Checkbox(
                        value=True,
                        label="Enable Universal Contrast Analysis"
                    )
                    
                    contrast_threshold = gr.Slider(
                        minimum=3.0,
                        maximum=7.0,
                        value=4.5,
                        step=0.1,
                        label="WCAG Contrast Threshold (4.5:1 recommended)"
                    )
                
                # Analysis button
                analyze_button = gr.Button(
                    "🔍 Analyze Environment",
                    variant="primary",
                    size="lg"
                )
            
            # Output Column
            with gr.Column(scale=2):
                # Analysis tabs
                with gr.Tabs():
                    with gr.Tab("📊 Analysis Report"):
                        analysis_report = gr.Markdown(
                            value="Upload an image and click 'Analyze Environment' to begin."
                        )
                    
                    with gr.Tab("🎯 Object Segmentation"):
                        seg_display = gr.Image(
                            label="Detected Objects",
                            height=400,
                            interactive=False
                        )
                    
                    if blackspot_ok:
                        with gr.Tab("⚫ Floor Blackspots"):
                            blackspot_display = gr.Image(
                                label="Blackspot Detection (Floors Only)",
                                height=400,
                                interactive=False
                            )
                    else:
                        blackspot_display = gr.Image(visible=False)
                    
                    with gr.Tab("🎨 Contrast Analysis"):
                        contrast_display = gr.Image(
                            label="Low Contrast Areas (All Objects)",
                            height=400,
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
                enable_contrast
            ],
            outputs=[
                seg_display,
                blackspot_display,
                contrast_display,
                analysis_report
            ]
        )
        
        # Footer
        gr.Markdown("""
            ---
            **NeuroNest** v2.0 - Enhanced with floor-only blackspot detection and universal contrast analysis  
            *Creating safer environments for cognitive health through AI*
            """)
    
    return interface


########################################
# MAIN EXECUTION
########################################

if __name__ == "__main__":
    print(f"🚀 Starting NeuroNest on {DEVICE}")
    print(f"OneFormer available: {ONEFORMER_AVAILABLE}")
    
    try:
        interface = create_gradio_interface()
        interface.queue(max_size=10).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        raise
