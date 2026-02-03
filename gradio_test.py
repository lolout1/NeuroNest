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

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU_DEVICE = torch.device("cpu")
torch.set_num_threads(4)

FLOOR_CLASSES = {
    'floor': [3, 4, 13],
    'carpet': [28],
    'mat': [78],
}

ONEFORMER_CONFIG = {
    "ADE20K": {
        "key": "ade20k",
        "swin_cfg": "configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml",
        "swin_model": "shi-labs/oneformer_ade20k_swin_large",
        "swin_file": "250_16_swin_l_oneformer_ade20k_160k.pth",
        "process_size": 640,
        "max_size": 2560
    }
}

BLACKSPOT_MODEL_REPO = "lolout1/txstNeuroNest"
BLACKSPOT_MODEL_FILE = "model_0004999.pth"

DISPLAY_MAX_WIDTH = 1920
DISPLAY_MAX_HEIGHT = 1080

from universal_contrast_analyzer import UniversalContrastAnalyzer

def resize_image_for_processing(image: np.ndarray, target_size: int = 640, max_size: int = 2560) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = target_size / min(h, w)
    if scale * max(h, w) > max_size:
        scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    new_w = (new_w // 32) * 32
    new_h = (new_h // 32) * 32
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return resized, scale

def resize_mask_to_original(mask: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

def prepare_display_image(image: np.ndarray, max_width: int = DISPLAY_MAX_WIDTH, max_height: int = DISPLAY_MAX_HEIGHT) -> np.ndarray:
    h, w = image.shape[:2]
    scale = 1.0
    if w > max_width:
        scale = max_width / w
    if h * scale > max_height:
        scale = max_height / h
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return image

class OneFormerManager:
    def __init__(self):
        self.predictor = None
        self.metadata = None
        self.initialized = False
        self.process_size = ONEFORMER_CONFIG["ADE20K"]["process_size"]
        self.max_size = ONEFORMER_CONFIG["ADE20K"]["max_size"]

    def initialize(self, backbone: str = "swin"):
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
        if not self.initialized:
            raise RuntimeError("OneFormer not initialized")
        original_size = (image.shape[0], image.shape[1])
        image_processed, scale = resize_image_for_processing(image, self.process_size, self.max_size)
        logger.info(f"Processing image at {image_processed.shape}, scale: {scale}")
        predictions = self.predictor(image_processed, "semantic")
        seg_mask_processed = predictions["sem_seg"].argmax(dim=0).cpu().numpy()
        seg_mask_original = resize_mask_to_original(seg_mask_processed, original_size)
        visualizer = Visualizer(
            image[:, :, ::-1],
            metadata=self.metadata,
            instance_mode=ColorMode.IMAGE,
            scale=1.0
        )
        vis_output = visualizer.draw_sem_seg(seg_mask_original, alpha=0.6)
        vis_image = vis_output.get_image()[:, :, ::-1]
        vis_image_display = prepare_display_image(vis_image)
        return seg_mask_original, vis_image_display

    def extract_floor_areas(self, segmentation: np.ndarray) -> np.ndarray:
        floor_mask = np.zeros_like(segmentation, dtype=bool)
        for class_ids in FLOOR_CLASSES.values():
            for class_id in class_ids:
                floor_mask |= (segmentation == class_id)
        return floor_mask

class ImprovedBlackspotDetector:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.predictor = None
        self.floor_classes = [3, 4, 13, 28, 78]

    def download_model(self) -> str:
        # Check root directory first (HF Spaces LFS location)
        root_path = f"./{BLACKSPOT_MODEL_FILE}"
        if os.path.exists(root_path):
            logger.info(f"Using local blackspot model from root: {root_path}")
            return root_path

        # Try downloading from HuggingFace Hub
        try:
            model_path = hf_hub_download(
                repo_id=BLACKSPOT_MODEL_REPO,
                filename=BLACKSPOT_MODEL_FILE
            )
            logger.info(f"Downloaded blackspot model to: {model_path}")
            return model_path
        except Exception as e:
            logger.warning(f"Could not download blackspot model from HF: {e}")

            # Fall back to output directory
            local_path = f"./output_floor_blackspot/{BLACKSPOT_MODEL_FILE}"
            if os.path.exists(local_path):
                logger.info(f"Using local blackspot model: {local_path}")
                return local_path
            return None

    def initialize(self, threshold: float = 0.5) -> bool:
        try:
            if self.model_path is None:
                self.model_path = self.download_model()
            if self.model_path is None:
                logger.error("No blackspot model available")
                return False
            cfg = get_cfg()
            cfg.merge_from_file(
                model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            )
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
            cfg.MODEL.WEIGHTS = self.model_path
            cfg.MODEL.DEVICE = DEVICE
            self.predictor = DefaultPredictor(cfg)
            logger.info("MaskRCNN blackspot detector initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize blackspot detector: {e}")
            return False

    def is_on_floor_surface(
        self,
        blackspot_mask: np.ndarray,
        segmentation: np.ndarray,
        floor_mask: np.ndarray,
        overlap_threshold: float = 0.8
    ) -> bool:
        if np.sum(blackspot_mask) == 0:
            return False
        overlap = blackspot_mask & floor_mask
        overlap_ratio = np.sum(overlap) / np.sum(blackspot_mask)
        if overlap_ratio < overlap_threshold:
            return False
        blackspot_pixels = segmentation[blackspot_mask]
        if len(blackspot_pixels) == 0:
            return False
        unique_classes, counts = np.unique(blackspot_pixels, return_counts=True)
        floor_pixel_count = sum(
            counts[unique_classes == cls] for cls in self.floor_classes if cls in unique_classes
        )
        floor_ratio = floor_pixel_count / len(blackspot_pixels)
        return floor_ratio > 0.7

    def filter_non_floor_blackspots(
        self,
        blackspot_masks: List[np.ndarray],
        segmentation: np.ndarray,
        floor_mask: np.ndarray
    ) -> List[np.ndarray]:
        filtered_masks = []
        for mask in blackspot_masks:
            if self.is_on_floor_surface(mask, segmentation, floor_mask):
                filtered_masks.append(mask)
            else:
                logger.debug(f"Filtered out non-floor blackspot with area {np.sum(mask)}")
        return filtered_masks

    def detect_blackspots(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        floor_prior: Optional[np.ndarray] = None
    ) -> Dict:
        if self.predictor is None:
            raise RuntimeError("Blackspot detector not initialized")
        original_h, original_w = image.shape[:2]
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
        try:
            outputs = self.predictor(image)
            instances = outputs["instances"].to("cpu")
        except Exception as e:
            logger.error(f"Error in MaskRCNN prediction: {e}")
            return self._empty_results(image)
        if len(instances) == 0:
            return self._empty_results(image)
        pred_classes = instances.pred_classes.numpy()
        pred_masks = instances.pred_masks.numpy()
        scores = instances.scores.numpy()
        blackspot_indices = pred_classes == 1
        blackspot_masks = pred_masks[blackspot_indices] if np.any(blackspot_indices) else []
        blackspot_scores = scores[blackspot_indices] if np.any(blackspot_indices) else []
        if floor_prior is not None:
            floor_mask = floor_prior
        else:
            floor_mask = np.zeros(segmentation.shape, dtype=bool)
            for cls in self.floor_classes:
                floor_mask |= (segmentation == cls)
        filtered_blackspot_masks = self.filter_non_floor_blackspots(
            blackspot_masks, segmentation, floor_mask
        )
        combined_blackspot = np.zeros(image.shape[:2], dtype=bool)
        for mask in filtered_blackspot_masks:
            combined_blackspot |= mask
        visualization = self.create_visualization(image, floor_mask, combined_blackspot)
        visualization_display = prepare_display_image(visualization)
        floor_area = int(np.sum(floor_mask))
        blackspot_area = int(np.sum(combined_blackspot))
        coverage_percentage = (blackspot_area / floor_area * 100) if floor_area > 0 else 0
        return {
            'visualization': visualization_display,
            'floor_mask': floor_mask,
            'blackspot_mask': combined_blackspot,
            'floor_area': floor_area,
            'blackspot_area': blackspot_area,
            'coverage_percentage': coverage_percentage,
            'num_detections': len(filtered_blackspot_masks),
            'avg_confidence': float(np.mean(blackspot_scores)) if len(blackspot_scores) > 0 else 0.0
        }

    def create_visualization(
        self,
        image: np.ndarray,
        floor_mask: np.ndarray,
        blackspot_mask: np.ndarray
    ) -> np.ndarray:
        vis = image.copy()
        floor_overlay = vis.copy()
        floor_overlay[floor_mask] = [0, 255, 0]
        vis = cv2.addWeighted(vis, 0.7, floor_overlay, 0.3, 0)
        vis[blackspot_mask] = [255, 0, 0]
        blackspot_contours, _ = cv2.findContours(
            blackspot_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, blackspot_contours, -1, (255, 255, 0), 4)
        return vis

    def _empty_results(self, image: np.ndarray) -> Dict:
        empty_mask = np.zeros(image.shape[:2], dtype=bool)
        visualization_display = prepare_display_image(image)
        return {
            'visualization': visualization_display,
            'floor_mask': empty_mask,
            'blackspot_mask': empty_mask,
            'floor_area': 0,
            'blackspot_area': 0,
            'coverage_percentage': 0,
            'num_detections': 0,
            'avg_confidence': 0.0
        }

class NeuroNestApp:
    def __init__(self):
        self.oneformer = OneFormerManager()
        self.blackspot_detector = None
        self.contrast_analyzer = UniversalContrastAnalyzer(wcag_threshold=4.5)
        self.initialized = False

    def initialize(self):
        logger.info("Initializing NeuroNest application...")
        oneformer_success = self.oneformer.initialize()
        blackspot_success = False
        try:
            self.blackspot_detector = ImprovedBlackspotDetector()
            blackspot_success = self.blackspot_detector.initialize()
        except Exception as e:
            logger.warning(f"Could not initialize blackspot detector: {e}")
        self.initialized = oneformer_success
        return oneformer_success, blackspot_success

    def analyze_image(
        self,
        image_path: str,
        blackspot_threshold: float = 0.5,
        contrast_threshold: float = 4.5,
        enable_blackspot: bool = True,
        enable_contrast: bool = True
    ) -> Dict:
        if not self.initialized:
            return {"error": "Application not properly initialized"}
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
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
            logger.info("Running semantic segmentation...")
            seg_mask, seg_visualization = self.oneformer.semantic_segmentation(image_rgb)
            results['segmentation'] = {
                'visualization': seg_visualization,
                'mask': seg_mask
            }
            floor_prior = self.oneformer.extract_floor_areas(seg_mask)
            if enable_blackspot and self.blackspot_detector is not None:
                logger.info("Running blackspot detection...")
                try:
                    blackspot_results = self.blackspot_detector.detect_blackspots(
                        image_rgb, seg_mask, floor_prior
                    )
                    results['blackspot'] = blackspot_results
                    logger.info("Blackspot detection completed")
                except Exception as e:
                    logger.error(f"Error in blackspot detection: {e}")
                    results['blackspot'] = None
            if enable_contrast:
                logger.info("Running universal contrast analysis...")
                try:
                    contrast_results = self.contrast_analyzer.analyze_contrast(
                        image_rgb, seg_mask
                    )
                    contrast_viz_display = prepare_display_image(contrast_results['visualization'])
                    contrast_results['visualization'] = contrast_viz_display
                    results['contrast'] = contrast_results
                    logger.info("Contrast analysis completed")
                except Exception as e:
                    logger.error(f"Error in contrast analysis: {e}")
                    results['contrast'] = None
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
        stats = {}
        if results['segmentation']:
            unique_classes = np.unique(results['segmentation']['mask'])
            stats['segmentation'] = {
                'num_classes': len(unique_classes),
                'image_size': results['segmentation']['mask'].shape
            }
        if results['blackspot']:
            bs = results['blackspot']
            stats['blackspot'] = {
                'floor_area_pixels': bs['floor_area'],
                'blackspot_area_pixels': bs['blackspot_area'],
                'coverage_percentage': bs['coverage_percentage'],
                'num_detections': bs['num_detections'],
                'avg_confidence': bs['avg_confidence']
            }
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

def create_gradio_interface():
    app = NeuroNestApp()
    oneformer_ok, blackspot_ok = app.initialize()
    if not oneformer_ok:
        raise RuntimeError("Failed to initialize OneFormer")
    
    # Define sample images
    SAMPLE_IMAGES = [
        "samples/example1.png",
        "samples/example2.png",
        "samples/example3.png"
    ]
    
    # Check if sample images exist
    sample_images_available = all(os.path.exists(img) for img in SAMPLE_IMAGES)
    
    def analyze_wrapper(
        image_path,
        blackspot_threshold,
        contrast_threshold,
        enable_blackspot,
        enable_contrast
    ):
        if image_path is None:
            return None, None, None, "Please upload an image"
        results = app.analyze_image(
            image_path=image_path,
            blackspot_threshold=blackspot_threshold,
            contrast_threshold=contrast_threshold,
            enable_blackspot=enable_blackspot,
            enable_contrast=enable_contrast
        )
        if "error" in results:
            return None, None, None, f"Error: {results['error']}"
        seg_output = results['segmentation']['visualization'] if results['segmentation'] else None
        blackspot_output = results['blackspot']['visualization'] if results['blackspot'] else None
        contrast_output = results['contrast']['visualization'] if results['contrast'] else None
        if results['contrast']:
            contrast_report = app.contrast_analyzer.generate_report(results['contrast'])
        else:
            contrast_report = "Contrast analysis not performed."
        if results['blackspot']:
            bs = results['blackspot']
            blackspot_report = (
                f"**Floor Area:** {bs['floor_area']:,} pixels  \n"
                f"**Blackspot Area:** {bs['blackspot_area']:,} pixels  \n"
                f"**Coverage:** {bs['coverage_percentage']:.2f}%  \n"
                f"**Detections:** {bs['num_detections']}  \n"
                f"**Average Confidence:** {bs['avg_confidence']:.2f}"
            )
        else:
            blackspot_report = "Blackspot analysis not performed."
        report = generate_comprehensive_report(results, contrast_report, blackspot_report)
        return seg_output, blackspot_output, contrast_output, report
    
    def generate_comprehensive_report(results: Dict, contrast_report: str, blackspot_report: str) -> str:
        report = ["# 🧠 NeuroNest Analysis Report\n"]
        report.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
        if results['segmentation']:
            stats = results['statistics'].get('segmentation', {})
            report.append("## 🎯 Object Segmentation")
            report.append(f"- **Classes detected:** {stats.get('num_classes', 'N/A')}")
            report.append(f"- **Resolution:** {stats.get('image_size', 'N/A')}")
            report.append("")
        report.append("## ⚫ Blackspot Analysis")
        report.append(blackspot_report)
        report.append("")
        report.append("## 🎨 Universal Contrast Analysis")
        report.append(contrast_report)
        report.append("")
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
            contrast_issues = results['contrast']['issues']
            critical_issues = [i for i in contrast_issues if i['severity'] == 'critical']
            high_issues = [i for i in contrast_issues if i['severity'] == 'high']
            if critical_issues:
                report.append("\n**CRITICAL - Immediate attention required:**")
                for issue in critical_issues[:3]:
                    cat1, cat2 = issue['categories']
                    report.append(f"- {cat1.title()} ↔ {cat2.title()}: Increase contrast to 7:1 minimum")
            if high_issues:
                report.append("\n**HIGH PRIORITY:**")
                for issue in high_issues[:3]:
                    cat1, cat2 = issue['categories']
                    report.append(f"- {cat1.title()} ↔ {cat2.title()}: Increase contrast to 4.5:1 minimum")
            report.append("\n**General recommendations:**")
            report.append("- Paint furniture in colors that contrast with floors/walls")
            report.append("- Add colored tape or markers to furniture edges")
            report.append("- Install LED strip lighting under furniture edges")
            report.append("- Use contrasting placemats, cushions, or covers")
        if not has_issues:
            report.append("\n✅ **Excellent!** This environment appears well-optimized for individuals with Alzheimer's.")
            report.append("No significant visual hazards detected.")
        return "\n".join(report)
    
    title = "🧠 NeuroNest: AI-Powered Environment Safety Analysis"
    description = """
    **This is the backend of NeuroNest - an object detection and visual analysis application intended to improve the lives of those affected by Alzheimers.**
    
    **This version uses the free-tier CPU inferencing, it will take up to 3 minutes to process a picture** 
    
    **Texas State CS && Interior Design Dept. - Abheek Pradhan, Dr. Nadim Adi, Dr. Greg Lakomski**

    **People with Alzheimers can find many things uncomfortable such as black spots on floors and objects in the room having low contrast.** 
    
    This system provides:
    - **Object Segmentation**: Identifies all room elements (floors, walls, furniture)
    - **Floor-Only Blackspot Detection**: Locates dangerous dark areas on walking surfaces
    - **Universal Contrast Analysis**: Evaluates visibility between ALL adjacent objects
    *Following WCAG 2.1 guidelines for visual accessibility  | Upload a Picture. Click 'Analyze Environment'.Then scroll down.*
    """
    
    with gr.Blocks(css="""
        .container { max-width: 100%; margin: auto; padding: 20px; }
        .image-output { margin: 20px 0; }
        .image-output img { 
            width: 100%; 
            height: auto; 
            max-width: 1920px; 
            margin: 0 auto; 
            display: block;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .controls-row { margin-bottom: 30px; background: #f5f5f5; padding: 20px; border-radius: 8px; }
        .main-button { height: 80px !important; font-size: 1.3em !important; font-weight: bold !important; }
        .report-box { max-width: 1200px; margin: 30px auto; padding: 30px; background: #f9f9f9; border-radius: 8px; }
        h2 { margin-top: 40px; margin-bottom: 20px; color: #333; }
        .sample-section { 
            margin-bottom: 30px; 
            padding: 20px; 
            background: #fafafa; 
            border-radius: 12px;
            border: 1px solid #e0e0e0;
        }
        .examples-holder .examples-table {
            display: flex !important;
            justify-content: center !important;
            gap: 20px !important;
            margin-top: 15px !important;
        }
        .examples-holder img {
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            border: 2px solid transparent;
        }
        .examples-holder img:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            border: 2px solid #4A90E2;
        }
    """, theme=gr.themes.Base()) as interface:
        with gr.Column(elem_classes="container"):
            gr.Markdown(f"# {title}")
            gr.Markdown(description)
            if not blackspot_ok:
                gr.Markdown("""
                ⚠️ **Note:** Blackspot detection model not available. 
                To enable blackspot detection, upload the model to HuggingFace or ensure it's in the local directory.
                """)
            
            # First create a hidden image input that will be used by Examples
            with gr.Row(visible=False):
                image_input = gr.Image(
                    label="📸 Upload Room Image",
                    type="filepath",
                    height=500
                )
            
            # Sample images section at the top with the actual clickable examples
            with gr.Column(elem_classes="sample-section"):
                gr.Markdown("### 🖼️ Try Sample Images")
                gr.Markdown("*Click any image below to load it for analysis or upload your own. || Then scroll down and click analyze environment*")
                
                if sample_images_available:
                    gr.Examples(
                        examples=SAMPLE_IMAGES,
                        inputs=image_input,
                        label="",
                        examples_per_page=3
                    )
                else:
                    gr.Markdown("*Sample images not found in samples/ directory*")
            
            with gr.Row(elem_classes="controls-row"):
                with gr.Column(scale=1):
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
                        label="Blackspot Sensitivity",
                        visible=blackspot_ok
                    )
                with gr.Column(scale=1):
                    enable_contrast = gr.Checkbox(
                        value=True,
                        label="Enable Universal Contrast Analysis"
                    )
                    contrast_threshold = gr.Slider(
                        minimum=3.0,
                        maximum=7.0,
                        value=4.5,
                        step=0.1,
                        label="WCAG Contrast Threshold"
                    )
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Now show the actual visible image input
                    image_input_display = gr.Image(
                        label="📸 Upload Room Image",
                        type="filepath",
                        height=500
                    )
                    # Connect the hidden input to the visible one
                    image_input.change(
                        fn=lambda x: x,
                        inputs=image_input,
                        outputs=image_input_display
                    )
                with gr.Column(scale=1):
                    analyze_button = gr.Button(
                        "🔍 Analyze Environment",
                        variant="primary",
                        elem_classes="main-button"
                    )
            
            gr.Markdown("---")
            gr.Markdown("## 🎯 Segmented Objects")
            seg_display = gr.Image(
                label=None,
                interactive=False,
                show_label=False,
                elem_classes="image-output"
            )
            if blackspot_ok:
                gr.Markdown("## ⚫ Blackspot Detection")
                blackspot_display = gr.Image(
                    label=None,
                    interactive=False,
                    show_label=False,
                    elem_classes="image-output"
                )
            else:
                blackspot_display = gr.Image(visible=False)
            gr.Markdown("## 🎨 Contrast Analysis")
            contrast_display = gr.Image(
                label=None,
                interactive=False,
                show_label=False,
                elem_classes="image-output"
            )
            gr.Markdown("---")
            analysis_report = gr.Markdown(
                value="Upload an image and click 'Analyze Environment' to begin.",
                elem_classes="report-box"
            )
            
            # Use image_input_display for the analysis
            analyze_button.click(
                fn=analyze_wrapper,
                inputs=[
                    image_input_display,
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
            gr.Markdown("""
                ---
                **NeuroNest** v2.0 - Enhanced with floor-only blackspot detection and universal contrast analysis  
                *Creating safer environments for cognitive health through AI*
                """)
    return interface

if __name__ == "__main__":
    print(f"🚀 Starting NeuroNest on {DEVICE}")
    print(f"OneFormer available: {ONEFORMER_AVAILABLE}")
    try:
        interface = create_gradio_interface()
        interface.queue(max_size=10).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True
        )
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        raise
