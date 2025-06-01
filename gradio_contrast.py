import torch
import os
import sys
import time
import numpy as np
from PIL import Image
import cv2
import imutils
import colorsys
from scipy import ndimage

# Set CUDA device explicitly at the start
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Use first GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU available, using CPU")

print("Installed the dependencies!")

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog

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

# Force unbuffered output for SLURM logs
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Set environment variables for better GPU performance
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Contrast Detection Classes
class ContrastDetector:
    """Base class for contrast detection between segments"""
    
    @staticmethod
    def calculate_luminance_contrast(color1, color2):
        """Calculate WCAG luminance contrast ratio"""
        def get_relative_luminance(rgb):
            r, g, b = [val/255.0 for val in rgb]
            r = r/12.92 if r <= 0.03928 else ((r + 0.055)/1.055) ** 2.4
            g = g/12.92 if g <= 0.03928 else ((g + 0.055)/1.055) ** 2.4
            b = b/12.92 if b <= 0.03928 else ((b + 0.055)/1.055) ** 2.4
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        lum1 = get_relative_luminance(color1)
        lum2 = get_relative_luminance(color2)
        
        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)
        
        return (lighter + 0.05) / (darker + 0.05)
    
    @staticmethod
    def calculate_hue_contrast(color1, color2):
        """Calculate hue difference between two colors"""
        hsv1 = colorsys.rgb_to_hsv(color1[0]/255.0, color1[1]/255.0, color1[2]/255.0)
        hsv2 = colorsys.rgb_to_hsv(color2[0]/255.0, color2[1]/255.0, color2[2]/255.0)
        
        hue_diff = abs(hsv1[0] - hsv2[0])
        if hue_diff > 0.5:
            hue_diff = 1 - hue_diff
        
        return hue_diff * 2
    
    @staticmethod
    def calculate_saturation_contrast(color1, color2):
        """Calculate saturation difference between two colors"""
        hsv1 = colorsys.rgb_to_hsv(color1[0]/255.0, color1[1]/255.0, color1[2]/255.0)
        hsv2 = colorsys.rgb_to_hsv(color2[0]/255.0, color2[1]/255.0, color2[2]/255.0)
        
        return abs(hsv1[1] - hsv2[1])
    
    @staticmethod
    def analyze_contrast(image, segmentation, method="luminance", threshold=4.5):
        """Analyze contrast between adjacent segments"""
        unique_segments = np.unique(segmentation)
        h, w = segmentation.shape
        contrast_mask = np.zeros((h, w), dtype=bool)
        problem_areas = []
        
        # Calculate average colors for each segment
        segment_colors = {}
        for seg_id in unique_segments:
            mask = segmentation == seg_id
            if np.any(mask):
                segment_colors[seg_id] = np.mean(image[mask], axis=0).astype(int)
        
        # Check contrast between adjacent segments
        for i in range(h):
            for j in range(w):
                current_seg = segmentation[i, j]
                
                # Check 4-connected neighbors
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        neighbor_seg = segmentation[ni, nj]
                        
                        if current_seg != neighbor_seg:
                            color1 = segment_colors[current_seg]
                            color2 = segment_colors[neighbor_seg]
                            
                            if method == "luminance":
                                contrast = ContrastDetector.calculate_luminance_contrast(color1, color2)
                            elif method == "hue":
                                contrast = ContrastDetector.calculate_hue_contrast(color1, color2)
                                threshold = 0.3  # Adjust threshold for hue
                            elif method == "saturation":
                                contrast = ContrastDetector.calculate_saturation_contrast(color1, color2)
                                threshold = 0.3  # Adjust threshold for saturation
                            
                            if contrast < threshold:
                                contrast_mask[i, j] = True
                                problem_areas.append((current_seg, neighbor_seg, contrast))
        
        return contrast_mask, problem_areas, segment_colors

# Rest of your code remains the same until setup_cfg function
KEY_DICT = {"Cityscapes (19 classes)": "cityscapes",
            "COCO (133 classes)": "coco",
            "ADE20K (150 classes)": "ade20k",}

SWIN_CFG_DICT = {"cityscapes": "configs/cityscapes/oneformer_swin_large_IN21k_384_bs16_90k.yaml",
            "coco": "configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml",
            "ade20k": "configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml",}

SWIN_MODEL_DICT = {"cityscapes": hf_hub_download(repo_id="shi-labs/oneformer_cityscapes_swin_large",
                                            filename="250_16_swin_l_oneformer_cityscapes_90k.pth"),
              "coco": hf_hub_download(repo_id="shi-labs/oneformer_coco_swin_large",
                                            filename="150_16_swin_l_oneformer_coco_100ep.pth"),
              "ade20k": hf_hub_download(repo_id="shi-labs/oneformer_ade20k_swin_large",
                                            filename="250_16_swin_l_oneformer_ade20k_160k.pth")
            }

DINAT_CFG_DICT = {"cityscapes": "configs/cityscapes/oneformer_dinat_large_bs16_90k.yaml",
            "coco": "configs/coco/oneformer_dinat_large_bs16_100ep.yaml",
            "ade20k": "configs/ade20k/oneformer_dinat_large_IN21k_384_bs16_160k.yaml",}

DINAT_MODEL_DICT = {"cityscapes": hf_hub_download(repo_id="shi-labs/oneformer_cityscapes_dinat_large",
                                            filename="250_16_dinat_l_oneformer_cityscapes_90k.pth"),
              "coco": hf_hub_download(repo_id="shi-labs/oneformer_coco_dinat_large",
                                            filename="150_16_dinat_l_oneformer_coco_100ep.pth"),
              "ade20k": hf_hub_download(repo_id="shi-labs/oneformer_ade20k_dinat_large",
                                            filename="250_16_dinat_l_oneformer_ade20k_160k.pth")
            }

MODEL_DICT = {"DiNAT-L": DINAT_MODEL_DICT,
        "Swin-L": SWIN_MODEL_DICT }

CFG_DICT = {"DiNAT-L": DINAT_CFG_DICT,
        "Swin-L": SWIN_CFG_DICT }

WIDTH_DICT = {"cityscapes": 512,
              "coco": 512,
              "ade20k": 640}

# Modified to ensure CUDA device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using device: {device}")
else:
    device = torch.device("cpu")
    print(f"WARNING: Using CPU device")

cpu_device = torch.device("cpu")

PREDICTORS = {
    "DiNAT-L": {
        "Cityscapes (19 classes)": None,
        "COCO (133 classes)": None,
        "ADE20K (150 classes)": None
    },
    "Swin-L": {
        "Cityscapes (19 classes)": None,
        "COCO (133 classes)": None,
        "ADE20K (150 classes)": None
    }
}

METADATA = {
    "DiNAT-L": {
        "Cityscapes (19 classes)": None,
        "COCO (133 classes)": None,
        "ADE20K (150 classes)": None
    },
    "Swin-L": {
        "Cityscapes (19 classes)": None,
        "COCO (133 classes)": None,
        "ADE20K (150 classes)": None
    }
}

def setup_modules():
    print("Setting up modules...")
    for dataset in ["Cityscapes (19 classes)", "COCO (133 classes)", "ADE20K (150 classes)"]:
        for backbone in ["DiNAT-L", "Swin-L"]:
            print(f"Loading {backbone} for {dataset}...")
            cfg = setup_cfg(dataset, backbone)
            metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
            )
            if 'cityscapes_fine_sem_seg_val' in cfg.DATASETS.TEST_PANOPTIC[0]:
                from cityscapesscripts.helpers.labels import labels
                stuff_colors = [k.color for k in labels if k.trainId != 255]
                metadata = metadata.set(stuff_colors=stuff_colors)

            # Create predictor with explicit device
            predictor = DefaultPredictor(cfg)
            predictor.model.to(device)

            PREDICTORS[backbone][dataset] = predictor
            METADATA[backbone][dataset] = metadata
            print(f"✓ Loaded {backbone} for {dataset}")
    print("All modules setup complete!")

def setup_cfg(dataset, backbone):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_oneformer_config(cfg)
    add_dinat_config(cfg)
    dataset = KEY_DICT[dataset]
    cfg_path = CFG_DICT[backbone][dataset]
    cfg.merge_from_file(cfg_path)

    # Explicitly set device to CUDA if available
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cuda:0'
        print(f"Config set to use CUDA device")
    else:
        cfg.MODEL.DEVICE = 'cpu'
        print(f"Config set to use CPU device")

    cfg.MODEL.WEIGHTS = MODEL_DICT[backbone][dataset]
    cfg.freeze()
    return cfg

# Rest of your functions remain the same
def panoptic_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "panoptic")
    panoptic_seg, segments_info = predictions["panoptic_seg"]
    out = visualizer.draw_panoptic_seg_predictions(
        panoptic_seg.to(cpu_device), segments_info, alpha=0.5
    )
    visualizer_map = Visualizer(img[:, :, ::-1], is_img=False, metadata=metadata, instance_mode=ColorMode.IMAGE)
    out_map = visualizer_map.draw_panoptic_seg_predictions(
        panoptic_seg.to(cpu_device), segments_info, alpha=1, is_text=False
    )
    return out, out_map, predictions

def instance_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "instance")
    instances = predictions["instances"].to(cpu_device)
    out = visualizer.draw_instance_predictions(predictions=instances, alpha=0.5)
    visualizer_map = Visualizer(img[:, :, ::-1], is_img=False, metadata=metadata, instance_mode=ColorMode.IMAGE)
    out_map = visualizer_map.draw_instance_predictions(predictions=instances, alpha=1, is_text=False)
    return out, out_map, predictions

def semantic_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "semantic")
    out = visualizer.draw_sem_seg(
        predictions["sem_seg"].argmax(dim=0).to(cpu_device), alpha=0.5
    )
    visualizer_map = Visualizer(img[:, :, ::-1], is_img=False, metadata=metadata, instance_mode=ColorMode.IMAGE)
    out_map = visualizer_map.draw_sem_seg(
        predictions["sem_seg"].argmax(dim=0).to(cpu_device), alpha=1, is_text=False
    )
    return out, out_map, predictions

TASK_INFER = {"the task is panoptic": panoptic_run, "the task is instance": instance_run, "the task is semantic": semantic_run}

def create_contrast_visualization(img, contrast_mask, problem_areas, segment_colors):
    """Create visualization of contrast issues"""
    # Copy original image
    contrast_viz = img.copy()
    
    # Highlight low contrast boundaries
    boundary_color = (255, 0, 0)  # Red for problem areas
    contrast_viz[contrast_mask] = boundary_color
    
    # Add information overlay
    info_text = f"Low contrast areas detected: {len(problem_areas)}"
    cv2.putText(contrast_viz, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return contrast_viz

def segment_and_analyze(path, task, dataset, backbone, enable_contrast, contrast_method, contrast_threshold):
    # Get predictions and segmentation visualization
    predictor = PREDICTORS[backbone][dataset]
    metadata = METADATA[backbone][dataset]
    img = cv2.imread(path)
    width = WIDTH_DICT[KEY_DICT[dataset]]
    img = imutils.resize(img, width=width)
    
    out, out_map, predictions = TASK_INFER[task](img, predictor, metadata)
    out_img = Image.fromarray(out.get_image())
    out_map_img = Image.fromarray(out_map.get_image())
    
    if not enable_contrast:
        return out_img, out_map_img, None, None
    
    # Extract segmentation mask from predictions
    if task == "the task is semantic":
        seg_mask = predictions["sem_seg"].argmax(dim=0).cpu().numpy()
    elif task == "the task is panoptic":
        seg_mask, _ = predictions["panoptic_seg"]
        seg_mask = seg_mask.cpu().numpy()
    elif task == "the task is instance":
        # For instance segmentation, create a mask from instances
        instances = predictions["instances"].to("cpu")
        seg_mask = np.zeros(img.shape[:2], dtype=np.int32)
        for i, mask in enumerate(instances.pred_masks):
            seg_mask[mask] = i + 1
    
    # Analyze contrast
    contrast_mask, problem_areas, segment_colors = ContrastDetector.analyze_contrast(
        img, seg_mask, method=contrast_method, threshold=contrast_threshold
    )
    
    # Create contrast visualization
    contrast_viz = create_contrast_visualization(img, contrast_mask, problem_areas, segment_colors)
    contrast_viz_img = Image.fromarray(contrast_viz[:, :, ::-1])  # Convert BGR to RGB
    
    # Generate analysis report
    report = f"### Contrast Analysis Report\n\n"
    report += f"**Method:** {contrast_method.capitalize()}\n"
    report += f"**Threshold:** {contrast_threshold}\n"
    report += f"**Total segments:** {len(segment_colors)}\n"
    report += f"**Low contrast boundaries found:** {len(problem_areas)}\n\n"
    
    if problem_areas:
        report += "**Problem Areas:**\n"
        for i, (seg1, seg2, contrast_value) in enumerate(problem_areas[:10]):  # Show first 10
            report += f"- Segments {seg1} and {seg2}: Contrast ratio = {contrast_value:.2f}\n"
        if len(problem_areas) > 10:
            report += f"... and {len(problem_areas) - 10} more\n"
    
    return out_img, out_map_img, contrast_viz_img, report

title = "<h1 style='text-align: center'>OneFormer:DIEGO MENTORIA MILIONÁRIA - APP 1</h1>"
description = "<p style='font-size: 14px; margin: 5px; font-weight: w300; text-align: center'> <a href='https://github.com/lolout1/sam2Contrast' style='text-decoration:none' target='_blank'>NeuroNest Contrast Model</a></p>" \
            + "<p style='font-size: 16px; margin: 5px; font-weight: w600; text-align: center'> <a href='https://praeclarumjj3.github.io/oneformer/' target='_blank'>Project Page</a> | <a href='https://arxiv.org/abs/2211.06220' target='_blank'>ArXiv Paper</a> | <a href='https://github.com/SHI-Labs/OneFormer' target='_blank'>Github Repo</a></p>" \
            + "<p style='text-align: center; margin: 5px; font-size: 14px; font-weight: w300;'>  \
                This model leverages the OneFormer architecture to perform comprehensive image segmentation and labeling across multiple tasks. The system can identify and segment various objects, structures, and regions within images with high accuracy. It supports semantic, instance, and panoptic segmentation modes, enabling detailed analysis of indoor and outdoor environments. The model excels at distinguishing between different classes of objects, from common everyday items to complex urban structures, making it particularly useful for environmental analysis and scene understanding applications.\
                </p>" \
            + "<p style='text-align: center; font-size: 14px; margin: 5px; font-weight: w300;'> [Note: Inference on CPU may take upto 2 minutes. On a single RTX A6000 GPU, OneFormer is able to inference at more than 15 FPS.]</p>"

# Main execution with error handling
if __name__ == "__main__":
    try:
        print("Starting setup...")
        setup_modules()

        print("Creating Gradio interface...")
        with gr.Blocks(title="OneFormer with Contrast Detection") as iface:
            gr.Markdown(title)
            gr.Markdown(description)
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(label="Input Image", type="filepath")
                    task = gr.Radio(
                        choices=["the task is panoptic", "the task is instance", "the task is semantic"], 
                        value="the task is panoptic", 
                        label="Task Token Input"
                    )
                    dataset = gr.Radio(
                        choices=["COCO (133 classes)", "Cityscapes (19 classes)", "ADE20K (150 classes)"], 
                        value="COCO (133 classes)", 
                        label="Model"
                    )
                    backbone = gr.Radio(
                        choices=["DiNAT-L", "Swin-L"], 
                        value="DiNAT-L", 
                        label="Backbone"
                    )
                    
                    with gr.Accordion("Contrast Detection Options", open=False):
                        enable_contrast = gr.Checkbox(
                            label="Enable Contrast Detection", 
                            value=False
                        )
                        contrast_method = gr.Radio(
                            choices=["luminance", "hue", "saturation"], 
                            value="luminance", 
                            label="Contrast Method"
                        )
                        contrast_threshold = gr.Slider(
                            minimum=1.0, 
                            maximum=10.0, 
                            value=4.5, 
                            step=0.1, 
                            label="Contrast Threshold (WCAG AA is 4.5)"
                        )
                    
                    submit_btn = gr.Button("Analyze", variant="primary")
                
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("Segmentation Results"):
                            seg_output = gr.Image(type="pil", label="Segmentation Overlay")
                            seg_map = gr.Image(type="pil", label="Segmentation Map")
                        
                        with gr.TabItem("Contrast Analysis"):
                            contrast_viz = gr.Image(type="pil", label="Contrast Visualization")
                            contrast_report = gr.Markdown(label="Contrast Analysis Report")
            
            examples = [
                ["examples/coco.jpeg", "the task is panoptic", "COCO (133 classes)", "DiNAT-L", False, "luminance", 4.5],
                ["examples/cityscapes.png", "the task is panoptic", "Cityscapes (19 classes)", "DiNAT-L", False, "luminance", 4.5],
                ["examples/ade20k.jpeg", "the task is panoptic", "ADE20K (150 classes)", "DiNAT-L", False, "luminance", 4.5]
            ]
            
            gr.Examples(
                examples=examples,
                inputs=[input_image, task, dataset, backbone, enable_contrast, contrast_method, contrast_threshold],
                outputs=[seg_output, seg_map, contrast_viz, contrast_report],
                fn=segment_and_analyze,
                cache_examples=False
            )
            
            submit_btn.click(
                fn=segment_and_analyze,
                inputs=[input_image, task, dataset, backbone, enable_contrast, contrast_method, contrast_threshold],
                outputs=[seg_output, seg_map, contrast_viz, contrast_report]
            )

        print("Launching Gradio app...")
        iface.launch(server_name="0.0.0.0", server_port=7860, share=True)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
