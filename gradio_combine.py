import torch
import numpy as np
from PIL import Image
import cv2
import imutils
import os
import sys
import time
import colorsys
from scipy import ndimage
import gradio as gr
from huggingface_hub import hf_hub_download

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor as DetectronPredictor
from detectron2 import model_zoo

# OneFormer imports
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
)
from demo.defaults import DefaultPredictor as OneFormerPredictor
from demo.visualizer import Visualizer, ColorMode

# NeuroNest contrast detection imports
from utils.contrast_detector import ContrastDetector
from utils.luminance_contrast import LuminanceContrastDetector
from utils.hue_contrast import HueContrastDetector
from utils.saturation_contrast import SaturationContrastDetector
from utils.combined_contrast import CombinedContrastDetector

# Set threads for CPU optimization
torch.set_num_threads(4)

########################################
# GLOBAL CONFIGURATIONS
########################################

# OneFormer configurations
KEY_DICT = {"ADE20K (150 classes)": "ade20k"}

SWIN_CFG_DICT = {
    "ade20k": "configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml",
}

SWIN_MODEL_DICT = {
    "ade20k": hf_hub_download(
        repo_id="shi-labs/oneformer_ade20k_swin_large",
        filename="250_16_swin_l_oneformer_ade20k_160k.pth"
    )
}

DINAT_CFG_DICT = {
    "ade20k": "configs/ade20k/oneformer_dinat_large_IN21k_384_bs16_160k.yaml",
}

DINAT_MODEL_DICT = {
    "ade20k": hf_hub_download(
        repo_id="shi-labs/oneformer_ade20k_dinat_large",
        filename="250_16_dinat_l_oneformer_ade20k_160k.pth"
    )
}

MODEL_DICT = {"DiNAT-L": DINAT_MODEL_DICT, "Swin-L": SWIN_MODEL_DICT}
CFG_DICT = {"DiNAT-L": DINAT_CFG_DICT, "Swin-L": SWIN_CFG_DICT}
WIDTH_DICT = {"ade20k": 640}

# Contrast detector mapping
CONTRAST_DETECTORS = {
    "Luminance (WCAG)": LuminanceContrastDetector(),
    "Hue": HueContrastDetector(),
    "Saturation": SaturationContrastDetector(),
    "Combined": CombinedContrastDetector()
}

# Device configuration
cpu_device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model storage
ONEFORMER_PREDICTORS = {
    "DiNAT-L": {"ADE20K (150 classes)": None},
    "Swin-L": {"ADE20K (150 classes)": None}
}

ONEFORMER_METADATA = {
    "DiNAT-L": {"ADE20K (150 classes)": None},
    "Swin-L": {"ADE20K (150 classes)": None}
}

########################################
# MASK R-CNN SETUP AND FUNCTIONS
########################################

def load_maskrcnn_model(weights_path, device="cuda", threshold=0.5):
    """Load Mask R-CNN model for blackspot detection"""
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # [Floors, blackspot]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = weights_path
    # Fix: Convert torch.device to string
    cfg.MODEL.DEVICE = str(device) if isinstance(device, torch.device) else device
    return DetectronPredictor(cfg)
def postprocess_blackspot_masks(im, instances, show_floor=True, show_blackspot=True):
    """Extract floor and blackspot masks from Mask R-CNN predictions"""
    height, width = im.shape[:2]
    pred_classes = instances.pred_classes.cpu().numpy()
    pred_masks = instances.pred_masks.cpu().numpy()
    
    combined_floor_mask = np.zeros((height, width), dtype=bool)
    final_blackspot = np.zeros((height, width), dtype=bool)
    
    for cls_id, mask in zip(pred_classes, pred_masks):
        if cls_id == 0 and show_floor:  # Floor class
            combined_floor_mask |= mask
        elif cls_id == 1 and show_blackspot:  # Blackspot class
            final_blackspot |= mask
    
    return combined_floor_mask.astype(np.uint8), final_blackspot.astype(np.uint8)

########################################
# ONEFORMER SETUP AND FUNCTIONS
########################################

def setup_oneformer_modules():
    """Initialize OneFormer models"""
    for dataset in ["ADE20K (150 classes)"]:
        for backbone in ["DiNAT-L", "Swin-L"]:
            cfg = setup_oneformer_cfg(dataset, backbone)
            metadata = MetadataCatalog.get(
                cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
            )
            ONEFORMER_PREDICTORS[backbone][dataset] = OneFormerPredictor(cfg)
            ONEFORMER_METADATA[backbone][dataset] = metadata

def setup_oneformer_cfg(dataset, backbone):
    """Setup OneFormer configuration"""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_oneformer_config(cfg)
    add_dinat_config(cfg)
    dataset = KEY_DICT[dataset]
    cfg_path = CFG_DICT[backbone][dataset]
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.MODEL.WEIGHTS = MODEL_DICT[backbone][dataset]
    cfg.freeze()
    return cfg

def semantic_run(img, predictor, metadata):
    """Run OneFormer semantic segmentation"""
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "semantic")
    out = visualizer.draw_sem_seg(
        predictions["sem_seg"].argmax(dim=0).to(cpu_device), alpha=0.5
    )
    return out, predictions["sem_seg"].argmax(dim=0).to(cpu_device).numpy()

########################################
# INTEGRATED ANALYSIS FUNCTION
########################################

def integrated_analysis(image_path, 
                       # Blackspot detection parameters
                       blackspot_threshold, show_floor, show_blackspot,
                       # Contrast detection parameters
                       enable_contrast, backbone, contrast_method, contrast_threshold):
    """
    Perform integrated analysis with both blackspot detection and contrast analysis
    """
    # Read the image
    im = cv2.imread(image_path)
    if im is None:
        return "Error: could not read image!", None, None, None
    
    # Resize for OneFormer if contrast analysis is enabled
    if enable_contrast:
        width = WIDTH_DICT["ade20k"]
        im_resized = imutils.resize(im, width=width)
    else:
        im_resized = im
    
    # Part 1: Blackspot Detection with Mask R-CNN
    blackspot_text = []
    blackspot_viz = None
    
    if show_floor or show_blackspot:
        weights_path = "./output_floor_blackspot/model_0004999.pth"
        maskrcnn_predictor = load_maskrcnn_model(weights_path, device, blackspot_threshold)
        
        # Run blackspot detection
        outputs = maskrcnn_predictor(im)
        instances = outputs["instances"]
        
        # Post-process masks
        floor_mask, blackspot_mask = postprocess_blackspot_masks(im, instances, show_floor, show_blackspot)
        
        # Create visualization
        blackspot_overlay = im.copy()
        overlay = np.zeros_like(im)
        
        if show_floor:
            overlay[floor_mask > 0] = (0, 255, 0)  # Green for floor
        if show_blackspot:
            overlay[blackspot_mask > 0] = (0, 0, 255)  # Red for blackspot
        
        blackspot_overlay = cv2.addWeighted(im, 1.0, overlay, 0.5, 0)
        blackspot_viz = Image.fromarray(cv2.cvtColor(blackspot_overlay, cv2.COLOR_BGR2RGB))
        
        # Calculate statistics
        blackspot_area = int(blackspot_mask.sum())
        floor_area = int(floor_mask.sum())
        
        blackspot_text.append(f"### Blackspot Detection Results")
        blackspot_text.append(f"**Threshold:** {blackspot_threshold:.2f}")
        
        if show_floor:
            blackspot_text.append(f"**Floor area:** {floor_area} pixels")
        if show_blackspot:
            blackspot_text.append(f"**Blackspot area:** {blackspot_area} pixels")
            if floor_area > 0 and show_floor:
                percentage = (blackspot_area / floor_area) * 100
                blackspot_text.append(f"**Blackspot coverage:** {percentage:.2f}% of floor area")
    
    # Part 2: Contrast Analysis with OneFormer
    segmentation_viz = None
    contrast_viz = None
    contrast_text = []
    
    if enable_contrast:
        dataset = "ADE20K (150 classes)"
        predictor = ONEFORMER_PREDICTORS[backbone][dataset]
        metadata = ONEFORMER_METADATA[backbone][dataset]
        
        # Get segmentation
        out, seg_mask = semantic_run(im_resized, predictor, metadata)
        segmentation_viz = Image.fromarray(out.get_image())
        
        # Analyze contrast
        img_rgb = cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB)
        detector = CONTRAST_DETECTORS[contrast_method]
        contrast_image, problem_areas, stats = detector.analyze(
            img_rgb, seg_mask, contrast_threshold
        )
        
        contrast_viz = Image.fromarray(contrast_image)
        
        # Create stats text
        contrast_text.append(f"### Contrast Analysis Results")
        contrast_text.append(f"**Method:** {contrast_method}")
        contrast_text.append(f"**Threshold:** {contrast_threshold:.2f}")
        contrast_text.append(f"**Problem Areas:** {stats['problem_count']}")
        
        if 'min_contrast' in stats:
            contrast_text.append(f"**Min Contrast:** {stats['min_contrast']:.2f}")
        if 'max_contrast' in stats:
            contrast_text.append(f"**Max Contrast:** {stats['max_contrast']:.2f}")
        if 'average_contrast' in stats:
            contrast_text.append(f"**Average Contrast:** {stats['average_contrast']:.2f}")
    
    # Combine results
    combined_text = []
    if blackspot_text:
        combined_text.extend(blackspot_text)
    if contrast_text:
        if blackspot_text:
            combined_text.append("\n")
        combined_text.extend(contrast_text)
    
    return "\n".join(combined_text), blackspot_viz, segmentation_viz, contrast_viz

########################################
# GRADIO INTERFACE
########################################

# Initialize models
print("Initializing OneFormer models...")
setup_oneformer_modules()

title = "NeuroNest: Integrated Blackspot & Contrast Detection"
description = """
This integrated system combines:
1. **Blackspot Detection**: Uses Mask R-CNN to detect blackspots on floors
2. **Contrast Analysis**: Uses OneFormer segmentation to analyze contrast between objects

Both analyses help identify potential accessibility issues for individuals with Alzheimer's disease.
"""

# Create the Gradio interface
with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input image
            image_input = gr.Image(label="Input Image", type="filepath")
            
            # Blackspot detection controls
            with gr.Accordion("Blackspot Detection Settings", open=True):
                blackspot_threshold = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="Blackspot Detection Threshold"
                )
                with gr.Row():
                    show_floor = gr.Checkbox(value=True, label="Show Floor")
                    show_blackspot = gr.Checkbox(value=True, label="Show Blackspots")
            
            # Contrast analysis controls
            with gr.Accordion("Contrast Analysis Settings", open=True):
                enable_contrast = gr.Checkbox(value=True, label="Enable Contrast Analysis")
                backbone = gr.Radio(
                    choices=["Swin-L", "DiNAT-L"], 
                    value="Swin-L", 
                    label="OneFormer Backbone"
                )
                contrast_method = gr.Radio(
                    choices=["Luminance (WCAG)", "Hue", "Saturation", "Combined"],
                    value="Luminance (WCAG)",
                    label="Contrast Detection Method"
                )
                contrast_threshold = gr.Slider(
                    minimum=1.0, maximum=10.0, value=4.5, step=0.1,
                    label="Contrast Threshold"
                )
            
            analyze_btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column(scale=2):
            # Output displays
            with gr.Tabs():
                with gr.Tab("Analysis Report"):
                    analysis_text = gr.Textbox(label="Analysis Results", lines=10)
                
                with gr.Tab("Blackspot Detection"):
                    blackspot_output = gr.Image(label="Blackspot Visualization")
                
                with gr.Tab("Segmentation"):
                    segmentation_output = gr.Image(label="OneFormer Segmentation")
                
                with gr.Tab("Contrast Analysis"):
                    contrast_output = gr.Image(label="Contrast Visualization")
    
    # Connect the interface
    analyze_btn.click(
        fn=integrated_analysis,
        inputs=[
            image_input,
            blackspot_threshold, show_floor, show_blackspot,
            enable_contrast, backbone, contrast_method, contrast_threshold
        ],
        outputs=[
            analysis_text, blackspot_output, segmentation_output, contrast_output
        ]
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["examples/indoor_room.jpg", 0.5, True, True, True, "Swin-L", "Luminance (WCAG)", 4.5],
            ["examples/living_room.jpg", 0.7, True, True, True, "DiNAT-L", "Combined", 3.0],
        ],
        inputs=[
            image_input,
            blackspot_threshold, show_floor, show_blackspot,
            enable_contrast, backbone, contrast_method, contrast_threshold
        ]
    )

if __name__ == "__main__":
    print(f"Launching integrated NeuroNest app on device: {device}")
    demo.queue().launch(server_name="0.0.0.0", share=True)
