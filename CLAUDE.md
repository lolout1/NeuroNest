# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a OneFormer-based universal image segmentation application with integrated contrast detection capabilities. It supports panoptic, instance, and semantic segmentation using transformer-based models (Swin-L and DiNAT-L backbones).

## Essential Commands

### Setup and Installation
```bash
# Download model checkpoints from HuggingFace
python setup.py --download

# Build deformable attention CUDA operations (required)
bash deform_setup.sh

# Install NATTEN if needed
cd NATTEN && pip install -e .
```

### Running the Application
```bash
# Main application launcher
bash t.sh

# Or run directly
python gradio_test.py
```

### Testing
```bash
# Run NATTEN tests
cd NATTEN && python -m unittest discover -v -s ./tests
```

## Architecture Overview

### Core Components
1. **OneFormer Model** (`oneformer/`): Universal segmentation transformer implementation
   - `oneformer_model.py`: Main model class
   - `modeling/`: Model components including backbones, decoders, and criterion
   - `data/`: Dataset mappers and tokenizers for text queries

2. **Gradio Applications** (multiple entry points):
   - `gradio_test.py`: Main combined interface with pre/post-processing logic
   - `gradio_app.py`: Basic segmentation interface
   - `gradio_contrast.py`: Contrast detection focused interface

3. **Contrast Detection** (`utils/`): Multiple contrast analysis algorithms
   - `improved_contrast_analyzer.py`: Main analyzer combining all methods
   - Individual analyzers: luminance, saturation, hue contrast detection

4. **Model Configurations** (`configs/`): Pre-defined configs for:
   - ADE20K (150 classes)
   - Cityscapes (19 classes)
   - COCO (133 classes)

### Key Implementation Details
- Models are loaded from HuggingFace hub (shi-labs organization)
- Supports both CPU and GPU inference (auto-detection)
- Custom CUDA kernels for multi-scale deformable attention
- NATTEN library provides neighborhood attention operations

### Model Checkpoints
The application uses pre-trained models from:
- `shi-labs/oneformer_cityscapes_swin_large`
- `shi-labs/oneformer_coco_swin_large`
- `shi-labs/oneformer_ade20k_swin_large`
- DiNAT variants also available

### Important Notes
- Always run `deform_setup.sh` after environment setup for CUDA operations
- The application expects model checkpoints to be downloaded before running
- Gradio interface runs on port 7860 by default
- GPU memory requirements vary by model (Swin-L models need ~12GB)