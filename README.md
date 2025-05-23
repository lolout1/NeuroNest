# NeuroNest: AI Framework for Alzheimer's-Friendly Environment Assessment

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**NeuroNest** is a comprehensive AI framework for automated assessment of indoor environments to identify potential hazards for individuals with Alzheimer's disease and dementia. The system combines state-of-the-art computer vision techniques to analyze spatial contrast, detect blackspots, and provide actionable insights for creating safer, more accessible living spaces.

## Features

- **🎯 Semantic Segmentation**: Identifies rooms, furniture, walls, floors, and objects using OneFormer
- **⚫ Blackspot Detection**: Locates dangerous dark areas on floors that may be misperceived as holes
- **🎨 Contrast Analysis**: Evaluates WCAG-compliant color contrast for visual accessibility
- **📊 Risk Assessment**: Provides comprehensive environmental safety scoring
- **🖥️ Interactive Interface**: User-friendly Gradio web interface for easy analysis
- **📈 Detailed Reporting**: Generates actionable recommendations for environment improvements

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Citation](#citation)

## Architecture Overview

NeuroNest integrates three main analysis modules:

### 1. Semantic Segmentation Module (`oneformer/`)
- **Framework**: OneFormer with Swin-Large transformer backbone
- **Dataset**: Trained on ADE20K (150 indoor object categories)
- **Function**: Identifies and segments objects, furniture, and architectural elements
- **Output**: Dense semantic predictions for scene understanding

### 2. Blackspot Detection Module (`blackspot/`)
- **Framework**: Custom Mask R-CNN implementation
- **Purpose**: Detects dark floor patterns that may cause spatial disorientation
- **Features**: Multiple visualization modes and confidence scoring
- **Integration**: Uses semantic floor masks as contextual priors

### 3. Contrast Analysis Module (`contrast/`)
- **Standard**: WCAG 2.1 compliant contrast ratio calculations
- **Innovation**: Semantic-aware priority weighting for safety-critical object relationships
- **Analysis**: Evaluates color contrast between adjacent objects and surfaces
- **Focus**: Prioritizes floor-furniture, wall-door, and stair-floor relationships

```
Input Image → OneFormer Segmentation → Multi-Modal Analysis → Risk Assessment
     ↓              ↓                        ↓                    ↓
   RGB Image    Semantic Masks         Floor Extraction      Safety Score
                                      Blackspot Detection    Recommendations
                                      Contrast Analysis      Visualizations
```

## Installation

### System Requirements

**Minimum:**
- OS: Ubuntu 18.04+ / macOS 10.15+ / Windows 10+
- RAM: 16GB system memory
- Storage: 25GB available space
- Python: 3.8+

**Recommended:**
- OS: Ubuntu 20.04 LTS
- RAM: 32GB system memory
- GPU: NVIDIA GPU with 8GB+ VRAM
- Storage: NVMe SSD with 50GB+ space

### Automated Installation

```bash
#!/bin/bash
# Quick installation script

# Clone repository
git clone https://github.com/yourusername/NeuroNest.git
cd NeuroNest

# Create conda environment
conda create -n neuronest python=3.8 -y
conda activate neuronest

# Install PyTorch (adjust for your CUDA version)
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install dependencies
pip install -r requirements.txt

# Install Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Manual Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/NeuroNest.git
cd NeuroNest
```

2. **Set up Python environment**
```bash
conda create -n neuronest python=3.8
conda activate neuronest
```

3. **Install PyTorch**
```bash
# For CUDA systems
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge

# For CPU-only systems
conda install pytorch==1.10.1 torchvision==0.11.2 cpuonly -c pytorch
```

4. **Install core dependencies**
```bash
pip install -r requirements.txt
```

5. **Install Detectron2**
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
```

6. **Compile CUDA extensions (if applicable)**
```bash
cd oneformer/modeling/pixel_decoder/ops
bash make.sh
cd ../../../..
```

### Dependencies

```txt
# requirements.txt
torch>=1.10.1
torchvision>=0.11.2
numpy>=1.21.0
opencv-python>=4.5.0
gradio>=3.0.0
huggingface-hub>=0.10.0
Pillow>=8.0.0
matplotlib>=3.5.0
scipy>=1.7.0
scikit-learn>=1.0.0
```

## Usage

### Quick Start

1. **Start the Gradio interface**
```bash
conda activate neuronest
python gradio_test.py
```

2. **Open your browser** to the displayed URL (typically `http://localhost:7860`)

3. **Upload an image** of an indoor environment

4. **Configure analysis settings**:
   - Enable/disable blackspot detection
   - Adjust detection thresholds
   - Select visualization styles
   - Set contrast analysis parameters

5. **Click "Analyze Environment"** to get comprehensive results

### Python API Usage

```python
from core import NeuroNestApp
from config import AnalysisConfig

# Initialize the framework
app = NeuroNestApp()
success = app.initialize()

if success:
    # Analyze a single image
    results = app.analyze_image(
        image_path="path/to/room_image.jpg",
        blackspot_threshold=0.5,
        contrast_threshold=4.5,
        enable_blackspot=True,
        enable_contrast=True
    )
    
    # Access results
    if "error" not in results:
        print(f"Segmentation completed: {results['segmentation'] is not None}")
        print(f"Blackspot coverage: {results['blackspot']['coverage_percentage']:.2f}%")
        print(f"Contrast issues found: {len(results['contrast']['critical_issues'])}")
    else:
        print(f"Analysis failed: {results['error']}")
```

### Command Line Interface

```bash
# Basic analysis
python -m neuronest analyze --input image.jpg --output results/

# Batch processing
python -m neuronest batch --input-dir images/ --output-dir results/

# Configuration validation
python -m neuronest validate-config --config config/analysis.yaml
```

## API Reference

### Gradio Web API

When running the Gradio interface, you can interact with it programmatically using the Gradio client:

#### Python Client

```bash
pip install gradio_client
```

```python
from gradio_client import Client, handle_file

# Connect to your Gradio instance
client = Client("http://localhost:7860/")

# Analyze an image
result = client.predict(
    image_path=handle_file('path/to/your/image.jpg'),
    blackspot_threshold=0.5,
    contrast_threshold=4.5,
    enable_blackspot=True,
    enable_contrast=True,
    blackspot_view_type="High Contrast",
    api_name="/analyze_wrapper"
)

# Results contain:
# [0] Semantic segmentation visualization
# [1] Blackspot analysis visualization  
# [2] Pure blackspot segmentation
# [3] Contrast issues visualization
# [4] Analysis report (markdown text)
print("Analysis completed!")
print("Report:", result[4])
```

#### JavaScript Client

```bash
npm install @gradio/client
```

```javascript
import { Client } from "@gradio/client";

const client = await Client.connect("http://localhost:7860/");

// Load image file
const response = await fetch("path/to/your/image.jpg");
const imageFile = await response.blob();

// Analyze image
const result = await client.predict("/analyze_wrapper", { 
    image_path: imageFile,
    blackspot_threshold: 0.5,
    contrast_threshold: 4.5,
    enable_blackspot: true,
    enable_contrast: true,
    blackspot_view_type: "High Contrast"
});

console.log("Analysis results:", result.data);
```

#### API Parameters

**Input Parameters:**
- `image_path` (required): Image file to analyze
- `blackspot_threshold` (float, default: 0.5): Detection sensitivity (0.1-0.9)
- `contrast_threshold` (float, default: 4.5): WCAG contrast threshold (1.0-10.0)
- `enable_blackspot` (bool, default: true): Enable blackspot detection
- `enable_contrast` (bool, default: true): Enable contrast analysis  
- `blackspot_view_type` (string, default: "High Contrast"): Visualization style
  - Options: "High Contrast", "Segmentation Only", "Blackspots Only", "Side by Side", "Annotated"

**Output (tuple of 5 elements):**
1. Semantic segmentation visualization (image path)
2. Blackspot analysis visualization (image path)
3. Pure blackspot segmentation (image path)
4. Contrast issues visualization (image path)
5. Analysis report (markdown text)

### Core Python API

#### NeuroNestApp Class

```python
from core import NeuroNestApp

app = NeuroNestApp()

# Initialize all components
oneformer_ok, blackspot_ok = app.initialize(
    blackspot_model_path="./models/blackspot_model.pth"
)

# Analyze image
results = app.analyze_image(
    image_path="room.jpg",
    blackspot_threshold=0.5,
    contrast_threshold=4.5,
    enable_blackspot=True,
    enable_contrast=True
)
```

#### Analysis Results Structure

```python
{
    'original_image': np.ndarray,  # Original input image
    'segmentation': {
        'visualization': np.ndarray,  # Segmentation overlay
        'mask': np.ndarray           # Raw segmentation mask
    },
    'blackspot': {
        'visualization': np.ndarray,     # Main blackspot view
        'floor_mask': np.ndarray,        # Detected floor areas
        'blackspot_mask': np.ndarray,    # Detected blackspots
        'coverage_percentage': float,     # % of floor covered by blackspots
        'num_detections': int,           # Number of blackspot instances
        'enhanced_views': {              # Multiple visualization options
            'high_contrast_overlay': np.ndarray,
            'segmentation_view': np.ndarray,
            'blackspot_only': np.ndarray,
            'side_by_side': np.ndarray,
            'annotated_view': np.ndarray
        }
    },
    'contrast': {
        'critical_issues': list,      # High-priority contrast problems
        'high_issues': list,          # Medium-priority issues
        'medium_issues': list,        # Lower-priority issues
        'visualization': np.ndarray,  # Contrast issues overlay
        'statistics': dict           # Summary statistics
    },
    'statistics': dict               # Combined analysis statistics
}
```

## Configuration

### Environment Variables

```bash
# .env
export NEURONEST_HOME=/path/to/neuronest
export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME=$NEURONEST_HOME/models
export DETECTRON2_DATASETS=$NEURONEST_HOME/datasets
```

### Configuration Files

#### System Configuration (`config/constants.py`)

```python
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
```

#### Analysis Configuration

```python
# Contrast analysis priority matrix
PRIORITY_RELATIONSHIPS = {
    ('floor', 'furniture'): ('critical', 'Furniture must be clearly visible against floor'),
    ('floor', 'stairs'): ('critical', 'Stairs must have clear contrast with floor'),
    ('floor', 'door'): ('high', 'Door should be easily distinguishable from floor'),
    ('wall', 'furniture'): ('high', 'Furniture should stand out from walls'),
    ('wall', 'door'): ('high', 'Doors should be clearly visible on walls'),
}
```

## Project Structure

```
NeuroNest/
├── README.md
├── requirements.txt
├── gradio_test.py              # Main application entry point
├── config/                     # Configuration files
│   ├── __init__.py
│   ├── constants.py           # Global constants and class mappings
│   └── device_config.py       # Hardware configuration
├── core/                      # Main application logic
│   ├── __init__.py
│   └── app.py                 # NeuroNestApp main class
├── oneformer/                 # Semantic segmentation module
│   ├── __init__.py
│   ├── manager.py             # OneFormer model management
│   └── README.md
├── blackspot/                 # Blackspot detection module
│   ├── __init__.py
│   └── detector.py            # Blackspot detection with Mask R-CNN
├── contrast/                  # Contrast analysis module
│   ├── __init__.py
│   └── analyzer.py            # WCAG-compliant contrast analysis
├── interface/                 # User interface components
│   └── gradio_ui.py          # Gradio web interface
├── utils/                     # Utility functions
│   ├── __init__.py
│   └── helpers.py            # Analysis report generation
├── configs/                   # Model configuration files
│   └── ade20k/               # OneFormer ADE20K configs
├── examples/                  # Example images (optional)
└── output/                   # Analysis results output
```

## Troubleshooting

### Common Issues

#### CUDA/GPU Issues
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Fix CUDA path issues
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### Memory Issues
```python
# Enable memory optimization
import torch
torch.cuda.empty_cache()

# Use mixed precision
torch.backends.cuda.matmul.allow_tf32 = True
```

#### OneFormer Installation Issues
```bash
# If OneFormer import fails, install manually
git clone https://github.com/SHI-Labs/OneFormer.git
cd OneFormer
pip install -e .
```

### Error Messages

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce batch size or use CPU mode |
| `OneFormer not available` | Install OneFormer package manually |
| `Model not found` | Check model paths in configuration |
| `Gradio server failed` | Check port availability (7860) |

## Contributing

We welcome contributions! Please see our contribution guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone for development
git clone https://github.com/yourusername/NeuroNest.git
cd NeuroNest

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
isort .
```

## Research Applications

NeuroNest has applications in:

- **Healthcare**: Environmental assessments for dementia care facilities
- **Architecture**: Accessibility-focused building design
- **Smart Homes**: Automated environment monitoring for aging-in-place
- **Research**: Studies on spatial cognition and environmental factors

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use NeuroNest in your research, please cite:

```bibtex
@software{neuronest2024,
  title={NeuroNest: AI Framework for Alzheimer's-Friendly Environment Assessment},
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/NeuroNest}
}
```

## Acknowledgments

- OneFormer team for the universal segmentation framework
- Detectron2 team for the object detection foundation
- ADE20K dataset creators for semantic segmentation data
- Gradio team for the intuitive web interface framework

## Contact

For questions, issues, or collaboration opportunities:

- Email: [sww35@txstate.edu]
- Issues: [GitHub Issues](https://github.com/lolout1/NeuroNest/issues)
- Discussions: [GitHub Discussions](https://github.com/lolout1/NeuroNest/discussions)

---


