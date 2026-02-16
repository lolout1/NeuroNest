import numpy as np
import cv2

METHOD_INFO = {
    "attention": {
        "title": "Self-Attention Map",
        "desc": "Where the model focuses within a single transformer layer",
        "detail": "CLS token attention over spatial patches",
        "cmap": cv2.COLORMAP_INFERNO,
    },
    "rollout": {
        "title": "Attention Rollout",
        "desc": "Cumulative attention flow through all encoder layers",
        "detail": "Abnar & Zuidema (2020) — aggregated multi-layer focus",
        "cmap": cv2.COLORMAP_INFERNO,
    },
    "gradcam": {
        "title": "GradCAM",
        "desc": "Gradient-weighted activation — regions driving class prediction",
        "detail": "Selvaraju et al. (2017) — class-discriminative localization",
        "cmap": cv2.COLORMAP_JET,
    },
    "entropy": {
        "title": "Predictive Entropy",
        "desc": "Per-pixel uncertainty — bright = uncertain, dark = confident",
        "detail": "Shannon entropy over 150-class posterior distribution",
        "cmap": cv2.COLORMAP_MAGMA,
    },
    "pca": {
        "title": "Feature PCA",
        "desc": "Hidden state structure — similar colors = similar features",
        "detail": "SVD projection of 1024-dim features to RGB",
        "cmap": None,
    },
    "integrated_gradients": {
        "title": "Integrated Gradients",
        "desc": "Principled pixel-level attribution via path integral from baseline to input",
        "detail": "Sundararajan et al. (2017) — axiom-satisfying attribution (completeness + sensitivity)",
        "cmap": cv2.COLORMAP_HOT,
    },
    "chefer": {
        "title": "Chefer Relevancy",
        "desc": "Attention x gradient propagation — relevance to prediction",
        "detail": "Chefer et al. (2021) — transformer-specific attribution",
        "cmap": cv2.COLORMAP_INFERNO,
    },
}


def draw_colorbar(image, cmap, lo_label="Low", hi_label="High", width=30):
    h, w = image.shape[:2]
    bar_h = h - 80
    gradient = np.linspace(0, 255, bar_h).astype(np.uint8)[::-1].reshape(-1, 1)
    gradient = np.tile(gradient, (1, width))
    bar_rgb = cv2.applyColorMap(gradient, cmap)
    bar_rgb = cv2.cvtColor(bar_rgb, cv2.COLOR_BGR2RGB)

    panel = np.full((h, width + 60, 3), 30, dtype=np.uint8)
    y_off = 40
    panel[y_off : y_off + bar_h, 10 : 10 + width] = bar_rgb

    cv2.putText(
        panel, hi_label, (8, y_off - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA,
    )
    cv2.putText(
        panel, lo_label, (8, y_off + bar_h + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA,
    )
    return np.hstack([image, panel])


def draw_info_panel(image, method_key, extra_text=""):
    info = METHOD_INFO.get(method_key, {})
    desc = info.get("desc", "")
    detail = info.get("detail", "")
    h, w = image.shape[:2]
    bar_h = 50
    bar = np.full((bar_h, w, 3), 25, dtype=np.uint8)
    cv2.putText(
        bar, desc, (10, 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA,
    )
    bottom_text = detail
    if extra_text:
        bottom_text = f"{extra_text}  |  {detail}" if detail else extra_text
    cv2.putText(
        bar, bottom_text, (10, 38),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA,
    )
    return np.vstack([image, bar])
