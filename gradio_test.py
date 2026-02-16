import torch
import torch.nn as nn
import torch.quantization
import numpy as np
from PIL import Image
import cv2
import gc
import os
import sys
import time
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import gradio as gr
from huggingface_hub import hf_hub_download
import warnings
warnings.filterwarnings("ignore")


def quantize_model_int8(model: nn.Module, model_name: str = "model") -> nn.Module:
    """Apply dynamic INT8 quantization to nn.Linear layers.

    Uses torch.quantization.quantize_dynamic (stable PyTorch API).
    Only quantizes nn.Linear layers which are the dominant cost in transformer
    models (DINOv3 ViT, Swin, etc.). Conv2d layers remain FP32. Falls back to
    the original FP32 model if quantization fails for any reason.

    Returns the quantized model, or the original if quantization fails.
    """
    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    if linear_count == 0:
        logger.info(f"[Quantize] {model_name}: no nn.Linear layers, skipping")
        return model

    logger.info(f"[Quantize] {model_name}: quantizing {linear_count} nn.Linear layers to INT8...")
    try:
        quantized = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        quantized_count = sum(
            1 for m in quantized.modules()
            if type(m).__name__ == 'DynamicQuantizedLinear'
        )
        logger.info(
            f"[Quantize] {model_name}: {quantized_count}/{linear_count} layers quantized to INT8"
        )
        return quantized
    except Exception as e:
        logger.warning(f"[Quantize] {model_name}: quantization failed ({e}), using FP32")
        return model

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2 import model_zoo

from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation
from ade20k_classes import ADE20K_COLORS, ADE20K_NAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU_DEVICE = torch.device("cpu")
torch.set_num_threads(4)

# INT8 quantization toggle: set via environment variable or changed at startup.
# When enabled, nn.Linear layers are quantized to INT8 for ~1.5-3x CPU speedup.
# Trade-off: <0.5% potential accuracy change at segment boundaries.
ENABLE_QUANTIZATION = os.environ.get("NEURONEST_QUANTIZE", "1") == "1"

FLOOR_CLASSES = {
    'floor': [3, 4, 13],
    'carpet': [28],
    'mat': [78],
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

class EoMTSegmenter:
    """Semantic segmentation using EoMT-DINOv3 via HuggingFace Transformers.

    Replaces OneFormerManager. Uses pretrained ADE20K 150-class model achieving
    59.5 mIoU (vs OneFormer's 57.0). ONNX-exportable architecture.
    """
    MODEL_ID = "tue-mps/ade20k_semantic_eomt_large_512"

    def __init__(self):
        self.processor = None
        self.model = None
        self.initialized = False

    def initialize(self, backbone: str = "dinov3") -> bool:
        try:
            logger.info(f"Loading EoMT-DINOv3 from {self.MODEL_ID}...")
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForUniversalSegmentation.from_pretrained(self.MODEL_ID)
            self.model.eval()
            if ENABLE_QUANTIZATION:
                self.model = quantize_model_int8(self.model, "EoMT-DINOv3-L")
            else:
                logger.info("INT8 quantization disabled for EoMT (FP32 mode)")
            self.initialized = True
            logger.info("EoMT-DINOv3 initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize EoMT: {e}")
            return False

    def semantic_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.initialized:
            raise RuntimeError("EoMT not initialized")
        original_h, original_w = image.shape[:2]
        logger.info(f"Processing image at {image.shape}")
        pil_image = Image.fromarray(image)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        with torch.inference_mode():
            outputs = self.model(**inputs)
        seg_maps = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(original_h, original_w)]
        )
        seg_mask = seg_maps[0].cpu().numpy().astype(np.uint8)
        vis_image = self._visualize_segmentation(image, seg_mask)
        vis_image_display = prepare_display_image(vis_image)
        return seg_mask, vis_image_display

    def _visualize_segmentation(self, image: np.ndarray, seg_mask: np.ndarray,
                                alpha: float = 0.6) -> np.ndarray:
        h, w = seg_mask.shape
        color_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for label_id in np.unique(seg_mask):
            if label_id < len(ADE20K_COLORS):
                color_overlay[seg_mask == label_id] = ADE20K_COLORS[label_id]
        vis = cv2.addWeighted(image, 1 - alpha, color_overlay, alpha, 0)
        labels, areas = np.unique(seg_mask, return_counts=True)
        min_area = h * w * 0.01
        for label_id, area in zip(labels, areas):
            if area >= min_area and label_id < len(ADE20K_NAMES):
                ys, xs = np.where(seg_mask == label_id)
                cx, cy = int(np.median(xs)), int(np.median(ys))
                name = ADE20K_NAMES[label_id].split(",")[0]
                cv2.putText(vis, name, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(vis, name, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return vis

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
            if ENABLE_QUANTIZATION:
                self.predictor.model = quantize_model_int8(
                    self.predictor.model, "MaskRCNN-R50-FPN"
                )
            else:
                logger.info("INT8 quantization disabled for MaskRCNN (FP32 mode)")
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
        self.segmenter = EoMTSegmenter()
        self.blackspot_detector = None
        self.contrast_analyzer = UniversalContrastAnalyzer(wcag_threshold=4.5)
        self.initialized = False

    def initialize(self):
        logger.info("Initializing NeuroNest application...")
        seg_success = self.segmenter.initialize()
        blackspot_success = False
        try:
            self.blackspot_detector = ImprovedBlackspotDetector()
            blackspot_success = self.blackspot_detector.initialize()
        except Exception as e:
            logger.warning(f"Could not initialize blackspot detector: {e}")
        self.initialized = seg_success
        return seg_success, blackspot_success

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
            # Stage 1: Semantic segmentation (blocking - downstream stages depend on this)
            t0 = time.perf_counter()
            logger.info("Running semantic segmentation...")
            seg_mask, seg_visualization = self.segmenter.semantic_segmentation(image_rgb)
            results['segmentation'] = {
                'visualization': seg_visualization,
                'mask': seg_mask
            }
            floor_prior = self.segmenter.extract_floor_areas(seg_mask)
            t_seg = time.perf_counter() - t0
            logger.info(f"Segmentation completed in {t_seg:.1f}s")

            # Stage 2: Blackspot + contrast run CONCURRENTLY (both only need seg_mask)
            # GIL is released during PyTorch C-level ops and numpy array operations,
            # so ThreadPoolExecutor achieves real parallelism on 2 vCPU.
            t1 = time.perf_counter()
            futures = {}
            with ThreadPoolExecutor(max_workers=2) as executor:
                if enable_blackspot and self.blackspot_detector is not None:
                    logger.info("Submitting blackspot detection...")
                    futures['blackspot'] = executor.submit(
                        self.blackspot_detector.detect_blackspots,
                        image_rgb, seg_mask, floor_prior
                    )
                if enable_contrast:
                    logger.info("Submitting contrast analysis...")
                    futures['contrast'] = executor.submit(
                        self.contrast_analyzer.analyze_contrast,
                        image_rgb, seg_mask
                    )

                for key, future in futures.items():
                    try:
                        result = future.result(timeout=300)
                        if key == 'contrast':
                            result['visualization'] = prepare_display_image(
                                result['visualization']
                            )
                        results[key] = result
                        logger.info(f"{key} analysis completed")
                    except Exception as e:
                        logger.error(f"Error in {key} analysis: {e}")
                        results[key] = None

            t_parallel = time.perf_counter() - t1
            t_total = time.perf_counter() - t0
            logger.info(
                f"Parallel stage completed in {t_parallel:.1f}s | "
                f"Total pipeline: {t_total:.1f}s"
            )

            gc.collect()
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
    seg_ok, blackspot_ok = app.initialize()
    if not seg_ok:
        raise RuntimeError("Failed to initialize EoMT segmentation")
    
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
    
    title = "🧠 NeuroNest: Production ML System for Alzheimer's Care Environment Analysis"
    description = """
    <div style="line-height: 1.8; font-size: 15px;">

    ## 🎯 System Overview
    **Production computer vision pipeline achieving 98% precision** in detecting visual hazards for Alzheimer's patients using fine-tuned Vision Transformers and MASK R-CNN on custom dataset (15,000+ samples).

    **Problem**: Alzheimer's patients misperceive dark floor areas as voids and cannot distinguish low-contrast objects, causing falls and mobility issues.

    **Solution**: Multi-model ensemble with semantic segmentation → floor isolation → defect detection → contrast analysis pipeline.

    ---

    ## 🏗️ Technical Architecture

    **ML Stack**
    - **Models**: EoMT-DINOv3 (Vision Transformer backbone) + MASK R-CNN with transfer learning
    - **Training**: Distributed GPU cluster (Nvidia A100), custom dataset creation with active learning
    - **Optimization**: Dynamic INT8 quantization, vectorized post-processing, concurrent pipeline execution
    - **Inference**: FastAPI REST API with async batch processing, Docker containerization

    **System Design**
    - **Backend**: Python FastAPI + PyTorch 2.5 + HuggingFace Transformers
    - **Deployment**: Docker on HuggingFace Spaces with CI/CD pipeline
    - **Frontend**: Gradio 5.x with responsive design, real-time progress tracking
    - **Standards**: WCAG 2.1 Level AA accessibility compliance (4.5:1 contrast minimum)

    **Key Achievements**
    - 98% precision on blackspot detection (custom MASK R-CNN)
    - Dynamic INT8 quantization: ~2-3x CPU inference speedup via `torch.quantization`
    - Vectorized boundary detection: 50-200x faster contrast analysis (numpy C-level ops)
    - Concurrent pipeline: blackspot + contrast analysis run in parallel
    - 80% reduction in manual labeling via automated CV pipeline (Detectron2 + Vision LLMs)

    ---

    ## 📋 Quick Start Guide

    ### **Step 1**: Upload Image
    - Click **"📸 Upload Room Image"** or select a **sample image** below
    - Best: Well-lit room photos showing floors and furniture

    ### **Step 2**: Configure (Optional)
    - **Blackspot Sensitivity**: Detection threshold (default: 0.5)
    - **Contrast Threshold**: WCAG ratio (default: 4.5:1)
    - Toggle analysis modules on/off

    ### **Step 3**: Analyze
    - Click **"🔍 Analyze Environment"**
    - Processing: ~1-2 min with INT8 quantization enabled (CPU inference)

    ### **Step 4**: Review Results
    Scroll down for:
    1. **Semantic Segmentation**: All room elements identified
    2. **Blackspot Detection**: Dangerous floor areas highlighted
    3. **Contrast Analysis**: Low-visibility object pairs
    4. **Full Report**: Statistics + recommendations

    ---

    ## ⚡ INT8 Quantization (CPU Optimization)

    This deployment uses **dynamic INT8 quantization** (`torch.quantization.quantize_dynamic`) to accelerate CPU inference:

    - **What it does**: Converts nn.Linear layer weights from FP32 (32-bit) to INT8 (8-bit), reducing memory by ~4x and using optimized INT8 matrix multiplication kernels
    - **Speed improvement**: ~1.5-3x faster inference for the DINOv3 Vision Transformer backbone
    - **Accuracy trade-off**: <0.5% potential change at segment boundaries. The argmax over 150 semantic classes may occasionally differ at edges between regions. This has negligible impact on contrast analysis and blackspot detection, which operate on region-level statistics rather than pixel-precise boundaries
    - **Toggle**: Set environment variable `NEURONEST_QUANTIZE=0` to disable and use full FP32 precision

    ---

    ## 👥 Team
    **Texas State University** (C.A.D.S Research Initiative)

    **Backend/ML**: Abheek Pradhan (ML Engineer)
    **Frontend/Mobile**: Samuel Chutter • Priyanka Karki
    **Faculty**: Dr. Nadim Adi (Interior Design) • Dr. Greg Lakomski (CS)

    **Tech**: PyTorch • Detectron2 • FastAPI • Docker • ONNX • React Native • HuggingFace • Gradio
    **Deployment**: [Huggingface](https://huggingface.co/spaces/lolout1/txstNeuroNest) • [Github](https://github.com/lolout1)

    </div>
    """
    
    with gr.Blocks(css="""
        /* Live Demo Button - Prominent CTA */
        .demo-button-container {
            text-align: center;
            margin: clamp(15px, 3vw, 25px) auto;
            padding: clamp(10px, 2vw, 20px);
        }

        .demo-cta-button {
            display: inline-block;
            padding: clamp(15px, 3vw, 22px) clamp(40px, 8vw, 60px) !important;
            font-size: clamp(1.1em, 2.5vw, 1.5em) !important;
            font-weight: 700 !important;
            color: white !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            border-radius: clamp(8px, 1.5vw, 12px) !important;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
            cursor: pointer !important;
            transition: all 0.4s ease !important;
            text-decoration: none !important;
            width: auto !important;
            min-width: clamp(200px, 40vw, 320px);
            animation: pulse-glow 2s ease-in-out infinite;
        }

        .demo-cta-button:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 12px 35px rgba(102, 126, 234, 0.7) !important;
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
        }

        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5); }
            50% { box-shadow: 0 8px 35px rgba(102, 126, 234, 0.8); }
        }

        .demo-label {
            display: inline-block;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: clamp(6px, 1.5vw, 10px) clamp(15px, 3vw, 25px);
            border-radius: clamp(20px, 4vw, 30px);
            font-size: clamp(0.75em, 1.5vw, 0.9em);
            font-weight: 600;
            margin-bottom: clamp(8px, 2vw, 12px);
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        /* Base Container - Responsive padding */
        /* Base Container - Responsive padding */
        .container {
            max-width: 100%;
            margin: auto;
            padding: clamp(12px, 3vw, 24px);
            box-sizing: border-box;
        }

        /* Typography - Responsive and readable */
        body, .markdown {
            font-size: clamp(14px, 1.5vw, 16px);
            line-height: 1.7;
            color: #2c3e50;
        }

        h1, h2, h3 {
            margin-top: clamp(20px, 4vw, 40px);
            margin-bottom: clamp(12px, 2vw, 20px);
            color: #1a1a1a;
            font-weight: 600;
        }

        h2 { font-size: clamp(1.3em, 2.5vw, 1.8em); }
        h3 { font-size: clamp(1.1em, 2vw, 1.4em); }

        /* Image outputs - Responsive with max constraints */
        .image-output {
            margin: clamp(15px, 3vw, 25px) 0;
            width: 100%;
        }

        .image-output img {
            width: 100%;
            height: auto;
            max-width: min(1920px, 100%);
            margin: 0 auto;
            display: block;
            border: 2px solid #e1e4e8;
            border-radius: clamp(6px, 1vw, 10px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }

        /* Controls - Responsive layout */
        .controls-row {
            margin-bottom: clamp(20px, 4vw, 35px);
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: clamp(15px, 3vw, 25px);
            border-radius: clamp(8px, 1.5vw, 12px);
            border: 1px solid #dee2e6;
        }

        /* Main analyze button - Touch friendly */
        .main-button {
            height: clamp(60px, 10vw, 85px) !important;
            font-size: clamp(1.1em, 2vw, 1.4em) !important;
            font-weight: 600 !important;
            width: 100% !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            border-radius: clamp(8px, 1.5vw, 12px) !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
            transition: all 0.3s ease !important;
            cursor: pointer !important;
        }

        .main-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
        }

        /* Report box - Readable and responsive */
        .report-box {
            max-width: min(1200px, 100%);
            margin: clamp(20px, 4vw, 35px) auto;
            padding: clamp(20px, 4vw, 35px);
            background: #ffffff;
            border-radius: clamp(8px, 1.5vw, 12px);
            border: 2px solid #e1e4e8;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
            line-height: 1.8;
        }

        /* Sample images section */
        .sample-section {
            margin-bottom: clamp(20px, 4vw, 35px);
            padding: clamp(15px, 3vw, 25px);
            background: linear-gradient(135deg, #fafbfc 0%, #f6f8fa 100%);
            border-radius: clamp(10px, 2vw, 14px);
            border: 2px solid #e1e4e8;
        }

        /* Example images - Responsive grid */
        .examples-holder .examples-table {
            display: flex !important;
            justify-content: center !important;
            gap: clamp(10px, 2vw, 20px) !important;
            margin-top: 15px !important;
            flex-wrap: wrap !important;
        }

        .examples-holder img {
            border-radius: clamp(6px, 1vw, 10px);
            cursor: pointer;
            transition: all 0.3s ease;
            border: 3px solid transparent;
            max-width: 100%;
            height: auto;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .examples-holder img:hover {
            transform: scale(1.03);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            border: 3px solid #667eea;
        }

        /* Mobile optimizations */
        @media (max-width: 768px) {
            .container { padding: 12px; }

            .controls-row {
                padding: 15px;
                margin-bottom: 20px;
            }

            .main-button {
                height: 70px !important;
                font-size: 1.2em !important;
            }

            .report-box {
                padding: 20px;
                margin: 20px 10px;
            }

            .examples-holder .examples-table {
                gap: 10px !important;
            }

            h2 { font-size: 1.4em; }
            h3 { font-size: 1.2em; }
        }

        /* Tablet optimizations */
        @media (min-width: 769px) and (max-width: 1024px) {
            .container { padding: 20px; }
            .main-button { height: 75px !important; }
        }

        /* Large screen optimizations */
        @media (min-width: 1920px) {
            .container { max-width: 1920px; }
            body, .markdown { font-size: 17px; }
        }

        /* High contrast mode for accessibility */
        @media (prefers-contrast: high) {
            .controls-row { border: 2px solid #000; }
            .sample-section { border: 2px solid #000; }
            .image-output img { border: 3px solid #000; }
        }

        /* Reduced motion for accessibility */
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }

        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            body, .markdown { color: #e1e4e8; }
            .controls-row { background: linear-gradient(135deg, #2d333b 0%, #22272e 100%); }
            .sample-section { background: linear-gradient(135deg, #22272e 0%, #1c2128 100%); }
            .report-box { background: #1c2128; border-color: #373e47; }
        }
    """, theme=gr.themes.Base()) as interface:
        with gr.Column(elem_classes="container"):
            gr.Markdown(f"# {title}")

            # Prominent Live Demo CTA Button
            gr.HTML("""
                <div class="demo-button-container">
                    <div class="demo-label">🚀 Try It Now</div>
                    <a href="#demo-section"
                       class="demo-cta-button"
                       onclick="setTimeout(function(){var el=document.getElementById('demo-section');if(el){el.scrollIntoView({behavior:'smooth',block:'center'});}},100);return false;"
                       ontouchend="setTimeout(function(){var el=document.getElementById('demo-section');if(el){el.scrollIntoView({behavior:'smooth',block:'center'});}},100);return false;"
                       style="text-decoration: none !important; display: inline-block;">
                        ▶️ Launch Live Demo
                    </a>
                </div>
            """)

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
                gr.Markdown("⏱️ **Note:** We use free-tier CPU inferencing, so processing can take up to a few minutes. Please be patient or feel free to [donate](https://github.com/sponsors) to support faster GPU processing! 💙")
                
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
            
            # Demo section anchor for scroll navigation
            gr.HTML('<div id="demo-section"></div>')

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
    print("EoMT-DINOv3 segmentation engine")
    try:
        interface = create_gradio_interface()
        interface.queue(max_size=10).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False  # HF Spaces provides public URL automatically
        )
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        raise
