import torch
import numpy as np
from PIL import Image
import cv2
import imutils
import os
import sys
import time
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from scipy import ndimage
import colorsys
import math

torch.set_num_threads(16)
torch.set_num_interop_threads(16)

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
)

from demo.defaults import DefaultPredictor
from demo.visualizer import Visualizer, ColorMode

import gradio as gr
from huggingface_hub import hf_hub_download

# NeuroNest specific imports
from utils.contrast_detector import ContrastDetector
from utils.luminance_contrast import LuminanceContrastDetector
from utils.hue_contrast import HueContrastDetector
from utils.saturation_contrast import SaturationContrastDetector
from utils.combined_contrast import CombinedContrastDetector

KEY_DICT = {
    "ADE20K (150 classes)": "ade20k",
}

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

cpu_device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

PREDICTORS = {
    "DiNAT-L": {"ADE20K (150 classes)": None},
    "Swin-L": {"ADE20K (150 classes)": None}
}

METADATA = {
    "DiNAT-L": {"ADE20K (150 classes)": None},
    "Swin-L": {"ADE20K (150 classes)": None}
}

# Contrast detector mapping
CONTRAST_DETECTORS = {
    "Luminance (WCAG)": LuminanceContrastDetector(),
    "Hue": HueContrastDetector(),
    "Saturation": SaturationContrastDetector(),
    "Combined": CombinedContrastDetector()
}

def setup_modules():
    for dataset in ["ADE20K (150 classes)"]:
        for backbone in ["DiNAT-L", "Swin-L"]:
            cfg = setup_cfg(dataset, backbone)
            metadata = MetadataCatalog.get(
                cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
            )
            PREDICTORS[backbone][dataset] = DefaultPredictor(cfg)
            METADATA[backbone][dataset] = metadata

def setup_cfg(dataset, backbone):
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
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "semantic")
    out = visualizer.draw_sem_seg(
        predictions["sem_seg"].argmax(dim=0).to(cpu_device), alpha=0.5
    )
    return out, predictions["sem_seg"].argmax(dim=0).to(cpu_device).numpy()

def analyze_contrast(image, segmentation, contrast_method, threshold):
    """Analyze contrast between segments using selected method"""
    detector = CONTRAST_DETECTORS[contrast_method]
    
    # Perform contrast analysis
    contrast_image, problem_areas, stats = detector.analyze(
        image, segmentation, threshold
    )
    
    return contrast_image, problem_areas, stats

def segment_and_analyze_contrast(path, backbone, contrast_method, threshold):
    """Main function to segment and analyze contrast"""
    dataset = "ADE20K (150 classes)"
    predictor = PREDICTORS[backbone][dataset]
    metadata = METADATA[backbone][dataset]
    
    # Read and resize image
    img = cv2.imread(path)
    if img is None:
        return None, None, "Error: Could not load image"
    
    width = WIDTH_DICT[KEY_DICT[dataset]]
    img = imutils.resize(img, width=width)
    
    # Get segmentation
    out, seg_mask = semantic_run(img, predictor, metadata)
    out_img = Image.fromarray(out.get_image())
    
    # Analyze contrast
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    contrast_img, problem_areas, stats = analyze_contrast(
        img_rgb, seg_mask, contrast_method, threshold
    )
    
    # Create stats text
    stats_text = f"### Contrast Analysis Results\n\n"
    stats_text += f"**Method:** {contrast_method}\n"
    stats_text += f"**Threshold:** {threshold:.2f}\n"
    stats_text += f"**Problem Areas:** {stats['problem_count']}\n"
    
    if 'min_contrast' in stats:
        stats_text += f"**Min Contrast:** {stats['min_contrast']:.2f}\n"
    if 'max_contrast' in stats:
        stats_text += f"**Max Contrast:** {stats['max_contrast']:.2f}\n"
    if 'average_contrast' in stats:
        stats_text += f"**Average Contrast:** {stats['average_contrast']:.2f}\n"
    
    # Convert contrast image to PIL
    contrast_pil = Image.fromarray(contrast_img)
    
    return out_img, contrast_pil, stats_text

# Initialize models
setup_modules()

# Gradio Interface
title = "<h1 style='text-align: center'>NeuroNest: Abheek Pradhan - Contrast Model</h1>"
description = "<p style='font-size: 16px; margin: 5px; font-weight: w600; text-align: center'> "\
              "<a href='https://github.com/lolout1/sam2Contrast' target='_blank'>Github Repo</a></p>" \
              "<p style='text-align: center; margin: 5px; font-size: 14px; font-weight: w300;'>" \
              "I am developing NeuroNest, a contrast detection system designed to identify areas with insufficient contrast " \
              "for individuals with Alzheimer's disease. This tool leverages OneFormer's state-of-the-art segmentation " \
              "capabilities trained on ADE20K dataset to detect indoor objects like floors, furniture, walls, and ceilings. " \
              "By analyzing contrast ratios between adjacent segments, NeuroNest flags potential visual accessibility issues " \
              "that may trigger confusion or disorientation in elderly individuals with cognitive impairments.</p>" \
              "<p style='text-align: center; font-size: 14px; margin: 5px; font-weight: w300;'>" \
              "[Note: When running on my Linux cluster, please request a GPU node for optimal performance. " \
              "On login nodes, CUDA may not be available.]</p>"

gradio_inputs = [
    gr.Image(label="Input Image", type="filepath"),
    gr.Radio(choices=["Swin-L", "DiNAT-L"], value="Swin-L", label="Backbone"),
    gr.Radio(
        choices=["Luminance (WCAG)", "Hue", "Saturation", "Combined"],
        value="Luminance (WCAG)",
        label="Contrast Detection Method"
    ),
    gr.Slider(
        minimum=1.0,
        maximum=10.0,
        value=4.5,
        step=0.1,
        label="Contrast Threshold (Lower = More Strict)"
    )
]

gradio_outputs = [
    gr.Image(type="pil", label="Segmentation Result"),
    gr.Image(type="pil", label="Contrast Analysis"),
    gr.Markdown(label="Analysis Results")
]

examples = [
    ["examples/indoor_room.jpg", "Swin-L", "Luminance (WCAG)", 4.5],
    ["examples/living_room.jpg", "DiNAT-L", "Combined", 3.0],
]

iface = gr.Interface(
    fn=segment_and_analyze_contrast,
    inputs=gradio_inputs,
    outputs=gradio_outputs,
    examples_per_page=5,
    allow_flagging="never",
    examples=examples,
    title=title,
    description=description
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", share=True)
