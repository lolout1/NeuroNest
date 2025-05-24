---
title: NeuroNest Alzheimer Environment Analysis
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: mit
---

# 🧠 NeuroNest: Alzheimer's Environment Analysis

Advanced AI system for creating dementia-friendly environments through comprehensive visual analysis.

## Features
- 🎯 Object Segmentation with OneFormer
- ⚫ Blackspot Detection (floor hazards)
- 🎨 Ultra-Sensitive Contrast Analysis
- 📊 Evidence-Based Safety Recommendations

## Model Information
- **Segmentation**: OneFormer (ADE20K trained)
- **Blackspot Detection**: Custom trained MaskRCNN + Color analysis
- **Contrast Analysis**: WCAG 2.0 + Alzheimer's optimized (7:1 ratio)

## Usage
Upload a room image to receive comprehensive analysis including:
- Object identification and segmentation
- Dangerous dark floor areas (blackspots)
- Low contrast issues between objects
- Actionable safety recommendations

## Technical Details
- Optimized for CPU deployment
- Processes images at 512px for efficiency
- Multi-method validation for accuracy
