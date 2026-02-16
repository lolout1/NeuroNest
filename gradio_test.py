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
from xai_analyzer import XAIAnalyzer

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
        self.xai_analyzer = None
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
        if seg_success:
            self.xai_analyzer = XAIAnalyzer(
                eomt_model=self.segmenter.model,
                eomt_processor=self.segmenter.processor,
                blackspot_predictor=(
                    self.blackspot_detector.predictor
                    if self.blackspot_detector else None
                ),
            )
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

    # ------------------------------------------------------------------
    # XAI wrapper with progress tracking
    # ------------------------------------------------------------------
    def xai_wrapper(image_path, method, layer, head_choice, target_class, progress=gr.Progress(track_tqdm=True)):
        """Run XAI analysis with Gradio progress bar. Returns 7 visualizations + report."""
        if image_path is None:
            return [None] * 7 + ["Upload an image and click **Run XAI Analysis** to begin."]
        if app.xai_analyzer is None:
            return [None] * 7 + ["XAI analyzer not initialized. Check server logs."]

        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                return [None] * 7 + ["Could not load image."]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            layer_idx = int(layer)
            head_idx = None if head_choice == "Mean (all heads)" else int(head_choice.split()[-1])
            class_id = None if target_class == "Auto (dominant)" else int(target_class.split(":")[0])

            outputs = [None] * 7
            report_text = ""

            if method == "Full Suite":
                progress(0, desc="Starting XAI Full Suite analysis...")
                results = app.xai_analyzer.run_full_analysis(
                    image_rgb, layer=layer_idx, head=head_idx,
                    target_class_id=class_id,
                    progress_callback=progress,
                )
                key_order = ["attention", "rollout", "gradcam", "entropy", "pca", "saliency", "chefer"]
                for i, key in enumerate(key_order):
                    r = results.get(key, {})
                    vis = r.get("visualization")
                    if vis is not None:
                        outputs[i] = prepare_display_image(vis)

                # Build comprehensive report
                rpt = results.get("report", {}).get("report", "")
                parts = [rpt] if rpt else []
                parts.append("\n---\n## Individual Method Results\n")
                ok_count = 0
                for key in key_order:
                    r = results.get(key, {})
                    rtext = r.get("report", "")
                    if rtext:
                        parts.append(f"\n{rtext}")
                        if "Error" not in rtext:
                            ok_count += 1
                parts.insert(1, f"\n**{ok_count}/7 methods completed successfully.**\n")
                report_text = "\n".join(parts)

                app.xai_analyzer.cleanup_fp32()

            else:
                method_map = {
                    "Self-Attention": ("self_attention_maps", {"layer": layer_idx, "head": head_idx}, 0),
                    "Attention Rollout": ("attention_rollout", {}, 1),
                    "GradCAM": ("gradcam_segmentation", {"target_class_id": class_id}, 2),
                    "Predictive Entropy": ("predictive_entropy", {}, 3),
                    "Feature PCA": ("feature_pca", {"layer": layer_idx}, 4),
                    "Class Saliency": ("class_saliency", {"target_class_id": class_id}, 5),
                    "Chefer Relevancy": ("chefer_relevancy", {"target_class_id": class_id}, 6),
                }
                if method in method_map:
                    func_name, kwargs, idx = method_map[method]
                    progress(0.2, desc=f"Running {method}...")
                    func = getattr(app.xai_analyzer, func_name)
                    result = func(image_rgb, **kwargs)
                    vis = result.get("visualization")
                    if vis is not None:
                        outputs[idx] = prepare_display_image(vis)
                    report_text = result.get("report", "")
                    progress(1.0, desc=f"{method} complete")

                    if method in ("GradCAM", "Class Saliency", "Chefer Relevancy"):
                        app.xai_analyzer.cleanup_fp32()

            return outputs + [report_text]

        except Exception as e:
            logger.error(f"XAI analysis error: {e}")
            import traceback
            traceback.print_exc()
            return [None] * 7 + [f"**Error**: {str(e)}\n\nOther methods may still work — try running them individually."]

    title = "NeuroNest"

    with gr.Blocks(css="""
        /* ========== Global ========== */
        .container { max-width: 1400px; margin: auto; padding: 0 16px; }

        /* ========== Header ========== */
        .hero {
            text-align: center; padding: 28px 16px 12px;
            background: linear-gradient(135deg, #eef2ff 0%, #faf5ff 50%, #ecfdf5 100%);
            border-radius: 16px; margin-bottom: 20px;
            border: 1px solid #e0e7ff;
        }
        .hero h1 { font-size: 2.2em; margin: 0 0 4px; font-weight: 800; color: #1e1b4b; letter-spacing: -0.5px; }
        .hero-sub { color: #4b5563; font-size: 1em; margin: 0 0 14px; line-height: 1.6; max-width: 800px; display: inline-block; }
        .metrics-row {
            display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;
            margin: 12px 0 8px;
        }
        .metric {
            display: inline-flex; flex-direction: column; align-items: center;
            padding: 10px 18px; border-radius: 12px; min-width: 100px;
            background: white; border: 1px solid #e5e7eb;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .metric-val { font-size: 1.35em; font-weight: 800; color: #4f46e5; line-height: 1.2; }
        .metric-label { font-size: 0.7em; color: #6b7280; font-weight: 500; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.5px; }
        .badge-row { text-align: center; margin: 8px 0 4px; }
        .badge {
            display: inline-block; padding: 3px 10px; margin: 2px 3px;
            border-radius: 16px; font-size: 0.72em; font-weight: 600;
            background: #f5f3ff; color: #5b21b6; border: 1px solid #ddd6fe;
        }

        /* ========== Sidebar Tabs ========== */
        .sidebar-tabs > .tab-nav {
            display: flex; flex-direction: column !important;
            min-width: 200px; gap: 2px; padding: 8px;
            background: #f9fafb; border-radius: 12px; border: 1px solid #e5e7eb;
        }
        .sidebar-tabs > .tab-nav button {
            text-align: left !important; padding: 10px 16px !important;
            border-radius: 8px !important; font-weight: 500 !important;
            font-size: 0.92em !important; border: none !important;
            transition: all 0.15s ease !important;
        }
        .sidebar-tabs > .tab-nav button:hover { background: #eef2ff !important; }
        .sidebar-tabs > .tab-nav button.selected {
            background: #4f46e5 !important; color: white !important;
            box-shadow: 0 2px 8px rgba(79,70,229,0.25) !important;
        }
        .sidebar-tabs { display: flex !important; flex-direction: row !important; gap: 20px; }
        .sidebar-tabs > .tabitem { flex: 1; min-width: 0; }

        @media (max-width: 768px) {
            .sidebar-tabs { flex-direction: column !important; }
            .sidebar-tabs > .tab-nav { flex-direction: row !important; min-width: unset; overflow-x: auto; }
            .sidebar-tabs > .tab-nav button { white-space: nowrap; }
            .metrics-row { gap: 6px; }
            .metric { padding: 6px 10px; min-width: 70px; }
            .metric-val { font-size: 1.1em; }
        }

        /* ========== Buttons ========== */
        .main-button {
            height: 52px !important; font-size: 1.05em !important; font-weight: 600 !important;
            width: 100% !important; border-radius: 10px !important;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
            border: none !important; color: white !important;
            box-shadow: 0 2px 8px rgba(79,70,229,0.3) !important;
            transition: all 0.2s ease !important;
        }
        .main-button:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(79,70,229,0.4) !important; }

        /* ========== Sections ========== */
        .sample-section {
            padding: 16px; margin-bottom: 16px;
            background: #fafbfc; border-radius: 12px; border: 1px solid #e5e7eb;
        }
        .controls-row {
            padding: 16px; margin-bottom: 16px;
            background: #f9fafb; border-radius: 12px; border: 1px solid #e5e7eb;
        }
        .report-box {
            max-width: 100%; margin: 16px 0; padding: 20px;
            background: #fff; border-radius: 12px; border: 1px solid #e5e7eb;
            line-height: 1.7;
        }
        .info-card {
            padding: 20px; background: #fafbfc; border-radius: 12px;
            border: 1px solid #e5e7eb; line-height: 1.7; margin: 8px 0;
        }

        /* ========== Image outputs ========== */
        .image-output { margin: 8px 0; }
        .image-output img {
            width: 100%; max-width: 100%; border-radius: 10px;
            border: 1px solid #e5e7eb; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }

        /* ========== XAI ========== */
        .xai-controls {
            padding: 16px; background: #f9fafb;
            border-radius: 12px; border: 1px solid #e5e7eb; margin-bottom: 16px;
        }
        .xai-panel { min-height: 180px; }
        .xai-panel img {
            border-radius: 10px; border: 1px solid #e5e7eb;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: transform 0.15s ease;
        }
        .xai-panel img:hover { transform: scale(1.015); box-shadow: 0 4px 16px rgba(0,0,0,0.14); }
        .xai-report {
            max-width: 100%; margin: 16px 0; padding: 20px;
            background: #f9fafb; border-radius: 12px; border: 1px solid #e5e7eb; line-height: 1.7;
        }
        .xai-report table { width: 100%; border-collapse: collapse; margin: 8px 0; }
        .xai-report table th, .xai-report table td { padding: 6px 10px; border: 1px solid #e5e7eb; text-align: left; }
        .xai-report table th { background: #f3f4f6; font-weight: 600; }

        /* ========== Examples ========== */
        .examples-holder img {
            border-radius: 8px; cursor: pointer; transition: all 0.2s;
            border: 2px solid transparent; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }
        .examples-holder img:hover { border-color: #4f46e5; box-shadow: 0 4px 12px rgba(79,70,229,0.2); }

        /* ========== Dark mode ========== */
        @media (prefers-color-scheme: dark) {
            .hero { background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #064e3b 100%); border-color: #4338ca; }
            .hero h1 { color: #e0e7ff; }
            .hero-sub { color: #c7d2fe; }
            .metric { background: #1f2937; border-color: #374151; }
            .metric-val { color: #a5b4fc; }
            .metric-label { color: #9ca3af; }
            .badge { background: #312e81; color: #c7d2fe; border-color: #4338ca; }
            .sidebar-tabs > .tab-nav { background: #1f2937; border-color: #374151; }
            .sidebar-tabs > .tab-nav button:hover { background: #312e81 !important; }
            .sample-section, .controls-row, .xai-controls, .info-card { background: #1f2937; border-color: #374151; }
            .report-box, .xai-report { background: #1f2937; border-color: #374151; }
            .xai-report table th { background: #374151; }
        }

        /* ========== Accessibility ========== */
        @media (prefers-contrast: high) {
            .sample-section, .controls-row, .xai-controls, .info-card { border: 2px solid currentColor; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { animation: none !important; transition: none !important; }
        }
    """, theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    )) as interface:
        with gr.Column(elem_classes="container"):
            # ---- Hero Header ----
            gr.HTML("""
            <div class="hero">
                <h1>NeuroNest</h1>
                <p class="hero-sub">
                    Production ML system for Alzheimer's care environment analysis.
                    Multi-model ensemble combining semantic segmentation, blackspot detection,
                    WCAG contrast analysis, and 7 Explainable AI visualization methods.
                </p>
                <div class="metrics-row">
                    <div class="metric"><span class="metric-val">98%</span><span class="metric-label">Detection Precision</span></div>
                    <div class="metric"><span class="metric-val">150</span><span class="metric-label">ADE20K Classes</span></div>
                    <div class="metric"><span class="metric-val">7</span><span class="metric-label">XAI Methods</span></div>
                    <div class="metric"><span class="metric-val">2-3x</span><span class="metric-label">INT8 Speedup</span></div>
                    <div class="metric"><span class="metric-val">WCAG</span><span class="metric-label">2.1 Level AA</span></div>
                </div>
                <div class="badge-row">
                    <span class="badge">EoMT-DINOv3 ViT-Large</span>
                    <span class="badge">Mask R-CNN R50-FPN</span>
                    <span class="badge">PyTorch 2.5</span>
                    <span class="badge">HuggingFace Transformers</span>
                    <span class="badge">Detectron2</span>
                    <span class="badge">Gradio 5.x</span>
                </div>
            </div>
            """)

            # ---- Sidebar Navigation Tabs ----
            with gr.Tabs(elem_classes="sidebar-tabs"):
                # ============================================================
                # TAB: Project Overview
                # ============================================================
                with gr.TabItem("Project Overview"):
                    gr.Markdown("""## Problem & Motivation

Alzheimer's patients experience **visual-perceptual deficits** that make dark floor areas appear as voids
and low-contrast objects invisible. This leads to falls, reduced mobility, and decreased independence.

## Solution Architecture

Multi-model ensemble pipeline that analyzes room environments for visual hazards:

| Stage | Model | Output |
|-------|-------|--------|
| 1. Semantic Segmentation | EoMT-DINOv3-Large (ViT backbone, 512x512, 24 layers) | 150-class pixel-level scene parsing |
| 2. Floor Isolation | Region filtering (80% overlap threshold) | Floor-only surface mask |
| 3. Blackspot Detection | Mask R-CNN R50-FPN (custom trained on 15,000+ samples) | Instance-level dark area detection |
| 4. Contrast Analysis | Vectorized boundary detection + WCAG 2.1 | Adjacent object-pair contrast ratios |
| 5. Explainable AI | 7 methods: attention, gradient, output-based | Model interpretability visualizations |

## Key Technical Achievements

- **98% precision** on blackspot detection via fine-tuned Mask R-CNN with active learning
- **Dynamic INT8 quantization**: ~2-3x CPU inference speedup via `torch.quantization.quantize_dynamic`
- **Vectorized analysis**: 50-200x faster contrast computation using numpy C-level operations
- **Concurrent pipeline**: Blackspot + contrast analysis execute in parallel (ThreadPoolExecutor)
- **7 XAI methods**: Self-attention, rollout, GradCAM, entropy, PCA, saliency, Chefer relevancy
- **CPU-optimized**: Runs on HuggingFace Spaces free tier (2 vCPU, 16 GB RAM)
- **Hook-based attention capture**: Forward hooks bypass SDPA limitations for XAI

## Optimization Pipeline
```
Image -> Resize (512x512) -> EoMT-DINOv3 (INT8) -> Semantic Mask
                                                   -> Floor Prior
Semantic Mask + Floor Prior -> Mask R-CNN -> Blackspot Instances
Semantic Mask -> Boundary Detection -> WCAG Contrast Ratios
EoMT Features -> 7 XAI Methods -> Interpretability Visualizations
```
""")

                # ============================================================
                # TAB: Team & Research
                # ============================================================
                with gr.TabItem("Research & Team"):
                    gr.Markdown("""## Research Context

**Texas State University** — C.A.D.S (Computer-Aided Design for Safety) Research Initiative

This project addresses a critical gap in assistive technology for cognitive health.
Environments designed with proper visual contrast and hazard-free flooring can significantly
improve quality of life and reduce fall risk for individuals with Alzheimer's disease.

## Team

| Role | Name | Contribution |
|------|------|-------------|
| **ML Engineer** | Abheek Pradhan | Model architecture, training pipeline, INT8 optimization, XAI system, deployment |
| **Frontend Developer** | Samuel Chutter | Mobile app (React Native), UI/UX design |
| **Frontend Developer** | Priyanka Karki | Web interface, data visualization |
| **Faculty Advisor** | Dr. Nadim Adi | Interior design expertise, clinical requirements |
| **Faculty Advisor** | Dr. Greg Lakomski | Computer science guidance, research methodology |

## Technology Stack

| Category | Technologies |
|----------|-------------|
| **ML Frameworks** | PyTorch 2.5, Detectron2, HuggingFace Transformers |
| **Models** | EoMT-DINOv3-Large, Mask R-CNN R50-FPN |
| **XAI Libraries** | grad-cam, captum (custom hook-based implementation) |
| **Image Processing** | OpenCV, Pillow, scikit-image, scipy |
| **Deployment** | Docker, HuggingFace Spaces, Gradio 5.x |
| **Standards** | WCAG 2.1 Level AA, ADE20K (150 classes) |

## Links

- [GitHub Repository](https://github.com/lolout1/NeuroNest)
- [HuggingFace Space](https://huggingface.co/spaces/lolout1/txstNeuroNest)
""")

                # ============================================================
                # TAB: Environment Analysis (Main Demo)
                # ============================================================
                with gr.TabItem("Live Demo"):
                    gr.Markdown("Upload a room image to detect visual hazards for Alzheimer's patients.")

                    with gr.Row(visible=False):
                        image_input = gr.Image(label="Upload", type="filepath")

                    if sample_images_available:
                        with gr.Column(elem_classes="sample-section"):
                            gr.Markdown("**Try a sample image** or upload your own below. Processing takes ~1-2 min on CPU.")
                            gr.Examples(
                                examples=SAMPLE_IMAGES,
                                inputs=image_input,
                                label="",
                                examples_per_page=3,
                            )

                    with gr.Row():
                        with gr.Column(scale=3):
                            image_input_display = gr.Image(
                                label="Upload Room Image",
                                type="filepath",
                                height=400,
                            )
                            image_input.change(fn=lambda x: x, inputs=image_input, outputs=image_input_display)
                        with gr.Column(scale=1):
                            analyze_button = gr.Button(
                                "Analyze Environment",
                                variant="primary",
                                elem_classes="main-button",
                                size="lg",
                            )
                            with gr.Accordion("Settings", open=False):
                                enable_blackspot = gr.Checkbox(
                                    value=blackspot_ok,
                                    label="Blackspot Detection",
                                    interactive=blackspot_ok,
                                )
                                blackspot_threshold = gr.Slider(
                                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                                    label="Blackspot Sensitivity",
                                    visible=blackspot_ok,
                                )
                                enable_contrast = gr.Checkbox(value=True, label="Contrast Analysis")
                                contrast_threshold = gr.Slider(
                                    minimum=3.0, maximum=7.0, value=4.5, step=0.1,
                                    label="WCAG Contrast Threshold",
                                    info="4.5:1 = WCAG AA, 7:1 = AAA",
                                )

                    gr.Markdown("### Results")
                    with gr.Tabs():
                        with gr.TabItem("Segmentation"):
                            seg_display = gr.Image(label="150-class ADE20K semantic segmentation", interactive=False, elem_classes="image-output")
                        with gr.TabItem("Blackspot Detection"):
                            if blackspot_ok:
                                blackspot_display = gr.Image(label="Floor blackspot instances with confidence scores", interactive=False, elem_classes="image-output")
                            else:
                                blackspot_display = gr.Image(visible=False)
                                gr.Markdown("Blackspot detection model not available in this deployment.")
                        with gr.TabItem("Contrast Analysis"):
                            contrast_display = gr.Image(label="WCAG 2.1 contrast ratio analysis with severity overlay", interactive=False, elem_classes="image-output")
                        with gr.TabItem("Full Report"):
                            analysis_report = gr.Markdown(
                                value="Upload an image and click **Analyze Environment** to begin.",
                                elem_classes="report-box",
                            )

                    analyze_button.click(
                        fn=analyze_wrapper,
                        inputs=[image_input_display, blackspot_threshold, contrast_threshold, enable_blackspot, enable_contrast],
                        outputs=[seg_display, blackspot_display, contrast_display, analysis_report],
                    )

                # ============================================================
                # TAB 2: Explainable AI
                # ============================================================
                # ============================================================
                # TAB: Explainable AI
                # ============================================================
                with gr.TabItem("Model Interpretability"):
                    gr.Markdown("""**7 research-grade XAI methods** to interpret the DINOv3-EoMT Vision Transformer's decisions.
Each visualization includes colorbars, quantitative metrics, and method citations.

| Category | Methods | Insight |
|----------|---------|---------|
| **Attention** | Self-Attention, Rollout | Where the model focuses spatially |
| **Gradient** | GradCAM, Saliency, Chefer | What drives specific class predictions |
| **Output** | Entropy, Feature PCA | Confidence levels and learned representations |
""")

                    # Hidden input for sample image Examples
                    with gr.Row(visible=False):
                        xai_hidden_input = gr.Image(
                            label="XAI Image",
                            type="filepath",
                        )

                    if sample_images_available:
                        with gr.Column(elem_classes="sample-section"):
                            gr.Markdown("### Sample Images")
                            gr.Examples(
                                examples=SAMPLE_IMAGES,
                                inputs=xai_hidden_input,
                                label="",
                                examples_per_page=3,
                            )

                    with gr.Row(elem_classes="xai-controls"):
                        with gr.Column(scale=2):
                            xai_image_input = gr.Image(
                                label="Upload Image",
                                type="filepath",
                                height=350,
                            )
                            xai_hidden_input.change(
                                fn=lambda x: x,
                                inputs=xai_hidden_input,
                                outputs=xai_image_input,
                            )
                        with gr.Column(scale=1):
                            xai_method = gr.Radio(
                                choices=[
                                    "Full Suite",
                                    "Self-Attention",
                                    "Attention Rollout",
                                    "GradCAM",
                                    "Predictive Entropy",
                                    "Feature PCA",
                                    "Class Saliency",
                                    "Chefer Relevancy",
                                ],
                                value="Full Suite",
                                label="XAI Method",
                            )
                            xai_layer = gr.Slider(
                                minimum=0,
                                maximum=23,
                                value=19,
                                step=1,
                                label="Transformer Layer (0=early features, 19=last encoder, 23=last decoder)",
                                info="Affects Self-Attention and Feature PCA. Default: last encoder layer.",
                            )
                            xai_head = gr.Dropdown(
                                choices=["Mean (all heads)"] + [f"Head {i}" for i in range(16)],
                                value="Mean (all heads)",
                                label="Attention Head",
                                info="Affects Self-Attention only. Different heads learn different patterns.",
                            )
                            xai_class = gr.Dropdown(
                                choices=["Auto (dominant)"] + [
                                    f"{i}: {ADE20K_NAMES[i].split(',')[0]}"
                                    for i in range(len(ADE20K_NAMES))
                                ],
                                value="Auto (dominant)",
                                label="Target Class (for gradient methods)",
                                info="Affects GradCAM, Saliency, Chefer. Auto picks the largest class.",
                            )
                            xai_button = gr.Button(
                                "Run XAI Analysis",
                                variant="primary",
                                elem_classes="main-button",
                                size="lg",
                            )
                            gr.Markdown(
                                "<small>Full Suite: ~2-5 min on CPU. "
                                "Individual methods: 5-60s each.</small>"
                            )

                    # Visualization grid with category grouping
                    gr.Markdown("### Attention-Based Methods")
                    gr.Markdown("*Where does the model look? These methods visualize spatial attention patterns.*")
                    with gr.Row(equal_height=True):
                        xai_attn = gr.Image(label="1. Self-Attention Map", interactive=False, elem_classes="xai-panel")
                        xai_rollout = gr.Image(label="2. Attention Rollout", interactive=False, elem_classes="xai-panel")

                    gr.Markdown("### Gradient-Based Methods")
                    gr.Markdown("*What drives class predictions? These use backpropagation to attribute importance.*")
                    with gr.Row(equal_height=True):
                        xai_gradcam = gr.Image(label="3. GradCAM", interactive=False, elem_classes="xai-panel")
                        xai_saliency = gr.Image(label="6. Class Saliency", interactive=False, elem_classes="xai-panel")
                        xai_chefer = gr.Image(label="7. Chefer Relevancy", interactive=False, elem_classes="xai-panel")

                    gr.Markdown("### Output & Feature Analysis")
                    gr.Markdown("*How confident is the model? What features has it learned?*")
                    with gr.Row(equal_height=True):
                        xai_entropy = gr.Image(label="4. Predictive Entropy", interactive=False, elem_classes="xai-panel")
                        xai_pca = gr.Image(label="5. Feature PCA", interactive=False, elem_classes="xai-panel")

                    gr.Markdown("### Comprehensive Analysis Report")
                    xai_report = gr.Markdown(
                        value=(
                            "Upload an image and click **Run XAI Analysis** to generate visualizations.\n\n"
                            "Each visualization includes:\n"
                            "- **Title bar** with method name and parameters\n"
                            "- **Colorbar** showing the value scale (high/low)\n"
                            "- **Info panel** with description and quantitative metrics\n"
                        ),
                        elem_classes="xai-report",
                    )

                    xai_button.click(
                        fn=xai_wrapper,
                        inputs=[xai_image_input, xai_method, xai_layer, xai_head, xai_class],
                        outputs=[
                            xai_attn, xai_rollout, xai_gradcam, xai_entropy,
                            xai_pca, xai_saliency, xai_chefer,
                            xai_report,
                        ],
                    )

                # ============================================================
                # TAB: Technical Details
                # ============================================================
                with gr.TabItem("Technical Details"):
                    gr.Markdown("""## Model Architecture

### EoMT-DINOv3-Large (Semantic Segmentation)
| Property | Value |
|----------|-------|
| HuggingFace Model | `tue-mps/ade20k_semantic_eomt_large_512` |
| Backbone | DINOv2 ViT-Large (24 layers, 16 heads, 1024 hidden dim) |
| Architecture | EoMT (Efficient open-vocabulary Mask Transformer) |
| Input Resolution | 512 x 512 |
| Patch Size | 14 x 14 |
| Classes | 150 (ADE20K) |
| Encoder | 20 layers (patch tokens + prefix) |
| Decoder | 4 layers (+ 100 query tokens) |
| Quantization | Dynamic INT8 on nn.Linear layers |

### Mask R-CNN R50-FPN (Blackspot Detection)
| Property | Value |
|----------|-------|
| Framework | Detectron2 |
| Backbone | ResNet-50 + Feature Pyramid Network |
| Classes | 2 (blackspot, background) |
| Training Data | 15,000+ samples with active learning |
| Precision | 98% on custom floor blackspot dataset |
| Floor Filter | 80% mask overlap + 70% pixel threshold |

## XAI Methods Detail

| # | Method | Type | Needs FP32 | Colormap | Reference |
|---|--------|------|-----------|----------|-----------|
| 1 | Self-Attention Maps | Attention | No | INFERNO | Vaswani et al. 2017 |
| 2 | Attention Rollout | Attention | No | INFERNO | Abnar & Zuidema 2020 |
| 3 | GradCAM | Gradient | Yes | JET | Selvaraju et al. 2017 |
| 4 | Predictive Entropy | Output | No | MAGMA | Shannon 1948 |
| 5 | Feature PCA | Hidden State | No | RGB | SVD projection |
| 6 | Class Saliency | Gradient | Yes | HOT | Simonyan et al. 2014 |
| 7 | Chefer Relevancy | Attn x Grad | Yes | INFERNO | Chefer et al. 2021 |

## Deployment
- **Platform**: HuggingFace Spaces (Docker SDK)
- **Hardware**: CPU free tier (2 vCPU, 16 GB RAM)
- **Docker Base**: python:3.10-slim (Debian Trixie)
- **Key Dependencies**: PyTorch 2.5.1, Transformers, Detectron2, Gradio 5.x
""")
    return interface

if __name__ == "__main__":
    print(f"Starting NeuroNest on {DEVICE}")
    print("EoMT-DINOv3 segmentation engine")
    try:
        interface = create_gradio_interface()
        interface.queue(
            max_size=10,
            default_concurrency_limit=1,
        ).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,  # HF Spaces provides public URL automatically
        )
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        raise
