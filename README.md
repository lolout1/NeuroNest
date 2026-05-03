---
title: TxstNeuroNest
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker 
app_file: app.py
pinned: false
---

# NeuroNest OneFormer

## 🚀 Live Demo

**Try it now:** [https://huggingface.co/spaces/lolout1/txstNeuroNest](https://huggingface.co/spaces/lolout1/txstNeuroNest?logs=container)

Experience NeuroNest's AI-powered room analysis in action. Upload any room image to detect accessibility hazards, analyze floor coverage, and get real-time contrast measurements—no installation required.

---

## About

NeuroNest analyzes indoor environments for safety hazards relevant to Alzheimer's and dementia care. The pipeline combines:

- **Semantic segmentation** (EoMT-DINOv3-Large, 150-class ADE20K) for scene parsing
- **Blackspot detection** (Mask R-CNN R50-FPN, fine-tuned) for dark floor regions perceived as voids
- **WCAG 2.1 contrast analysis** for low-contrast surface boundaries
- **Sign & clock placement analysis** (ADA-compliant centroid heights, 48–60 inches) via monocular metric depth estimation
- **Explainable AI suite** (7 visualization methods) for model interpretability

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
