"""
Explainable AI (XAI) analyzer for NeuroNest.

7 visualization methods for EoMT-DINOv3 semantic segmentation:
1. Self-Attention Maps   2. Attention Rollout   3. GradCAM
4. Predictive Entropy    5. Feature PCA         6. Integrated Gradients
7. Chefer Relevancy

CPU-compatible. Memory-managed for HF Spaces (16 GB).
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import gc
import logging
import os
import time
import traceback
from contextlib import contextmanager
from typing import Dict, Optional, Tuple
from PIL import Image

from ade20k_classes import ADE20K_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hook classes
# ---------------------------------------------------------------------------

class AttentionCaptureHook:
    """Captures attention weights from EomtAttention.forward() output[1]."""
    def __init__(self):
        self.weights = None
        self._h = None

    def _fn(self, module, args, output):
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            self.weights = output[1].detach().cpu()

    def register(self, module):
        self._h = module.register_forward_hook(self._fn)
        return self

    def remove(self):
        if self._h:
            self._h.remove()
            self._h = None


class HiddenStateCaptureHook:
    """Captures hidden states (first element of layer output)."""
    def __init__(self):
        self.state = None
        self._h = None

    def _fn(self, module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        self.state = out.detach().cpu()

    def register(self, module):
        self._h = module.register_forward_hook(self._fn)
        return self

    def remove(self):
        if self._h:
            self._h.remove()
            self._h = None


class ActivationGradientHook:
    """Captures forward activations AND backward gradients."""
    def __init__(self):
        self.activation = None
        self.gradient = None
        self._fh = None
        self._bh = None

    def register(self, module):
        self._fh = module.register_forward_hook(
            lambda m, i, o: setattr(self, 'activation', o[0] if isinstance(o, tuple) else o)
        )
        self._bh = module.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradient', go[0])
        )
        return self

    def remove(self):
        if self._fh:
            self._fh.remove()
        if self._bh:
            self._bh.remove()
        self._fh = self._bh = None


# ---------------------------------------------------------------------------
# Context manager: force eager attention
# ---------------------------------------------------------------------------

@contextmanager
def _eager_attention(model):
    """Temporarily switch to eager attention so attention weights are returned.
    SDPA is default (fast) but returns None for attn_weights."""
    orig = getattr(model.config, "_attn_implementation", "sdpa")
    model.config._attn_implementation = "eager"
    try:
        yield
    finally:
        model.config._attn_implementation = orig


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

# Method descriptions shown on visualizations
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


def _draw_colorbar(image, cmap, lo_label="Low", hi_label="High", width=30):
    """Draw a vertical colorbar on the right side of the image."""
    h, w = image.shape[:2]
    bar_h = h - 80  # leave room for labels
    gradient = np.linspace(0, 255, bar_h).astype(np.uint8)
    gradient = gradient[::-1].reshape(-1, 1)  # top=high, bottom=low
    gradient = np.tile(gradient, (1, width))
    bar_rgb = cv2.applyColorMap(gradient, cmap)
    bar_rgb = cv2.cvtColor(bar_rgb, cv2.COLOR_BGR2RGB)

    # Create panel
    panel = np.full((h, width + 60, 3), 30, dtype=np.uint8)
    y_off = 40
    panel[y_off:y_off + bar_h, 10:10 + width] = bar_rgb

    # Labels
    cv2.putText(panel, hi_label, (8, y_off - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, lo_label, (8, y_off + bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    return np.hstack([image, panel])


def _draw_info_panel(image, method_key, extra_text=""):
    """Draw a descriptive info bar at the bottom of the image."""
    info = METHOD_INFO.get(method_key, {})
    desc = info.get("desc", "")
    detail = info.get("detail", "")
    h, w = image.shape[:2]
    bar_h = 50
    bar = np.full((bar_h, w, 3), 25, dtype=np.uint8)
    # Description line
    cv2.putText(bar, desc, (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    # Detail / citation line
    bottom_text = detail
    if extra_text:
        bottom_text = f"{extra_text}  |  {detail}" if detail else extra_text
    cv2.putText(bar, bottom_text, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)
    return np.vstack([image, bar])


# ---------------------------------------------------------------------------
# XAI Analyzer
# ---------------------------------------------------------------------------

class XAIAnalyzer:
    """All methods CPU-safe, memory-managed, dynamically adapt to model config."""

    def __init__(self, eomt_model, eomt_processor, blackspot_predictor=None):
        self.model = eomt_model
        self.processor = eomt_processor
        self._model_fp32 = None

        # --- Probe architecture dynamically ---
        cfg = self.model.config
        self._num_layers = len(self.model.layers)
        self._num_heads = getattr(cfg, "num_attention_heads", 16)
        self._patch_size = getattr(cfg, "patch_size", 16)
        self._num_queries = getattr(cfg, "num_queries", 100)
        self._num_prefix = getattr(
            getattr(self.model, "embeddings", None), "num_prefix_tokens", 5
        )

        # Encoder/decoder split
        n_dec = getattr(cfg, "num_decoder_layers", getattr(cfg, "decoder_layers", 4))
        self._n_encoder = self._num_layers - n_dec

        logger.info(
            f"[XAI] {self._num_layers} layers ({self._n_encoder}enc+{n_dec}dec), "
            f"{self._num_heads} heads, patch={self._patch_size}, "
            f"prefix={self._num_prefix}, queries={self._num_queries}, "
            f"attn={getattr(cfg, '_attn_implementation', '?')}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray):
        h, w = image.shape[:2]
        inputs = self.processor(images=Image.fromarray(image), return_tensors="pt")
        return inputs, h, w

    def _patch_grid(self, inputs: dict) -> Tuple[int, int]:
        _, _, h, w = inputs["pixel_values"].shape
        return h // self._patch_size, w // self._patch_size

    def _infer_grid(self, seq_len: int, is_decoder: bool = False) -> Tuple[int, int]:
        """Infer (gh, gw) from attention sequence length."""
        n = seq_len - self._num_prefix
        if is_decoder:
            n -= self._num_queries
        n = max(n, 1)
        side = int(round(np.sqrt(n)))
        return side, side

    def _cls_to_patches(self, attn_2d, gh: int, gw: int, is_decoder: bool = False):
        """Extract CLS->patch attention row, reshape to spatial. Returns (map, gh, gw)."""
        seq = attn_2d.shape[-1]
        n_pre = self._num_prefix
        n_expected = gh * gw
        n_avail = seq - n_pre - (self._num_queries if is_decoder else 0)

        if n_avail != n_expected:
            gh, gw = self._infer_grid(seq, is_decoder)
            n_expected = gh * gw
            logger.debug(f"[XAI] Grid inferred: {gh}x{gw} from seq_len={seq}")

        row = attn_2d[0, n_pre:n_pre + n_expected]
        if isinstance(row, torch.Tensor):
            row = row.numpy()
        return row.reshape(gh, gw), gh, gw

    # Floor-related ADE20K class IDs — prefer these as XAI target
    _FLOOR_IDS = {3, 4, 13, 28, 78}  # floor, tree(?), path, carpet, mat

    def _dominant_class(self, seg_mask):
        """Pick best target class. Prefers floor classes if present (>1% of image),
        otherwise falls back to the largest class by area."""
        cls, cnt = np.unique(seg_mask, return_counts=True)
        total = seg_mask.size
        # Check for floor classes first
        for fid in self._FLOOR_IDS:
            if fid in cls:
                idx = list(cls).index(fid)
                if cnt[idx] / total > 0.01:  # at least 1% of image
                    return int(fid)
        # Fallback: largest class by area
        return int(cls[np.argmax(cnt)])

    def _cname(self, cid):
        return ADE20K_NAMES[cid].split(",")[0].strip() if 0 <= cid < len(ADE20K_NAMES) else f"class_{cid}"

    def _seg_from_outputs(self, outputs, h, w):
        return self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(h, w)]
        )[0].cpu().numpy().astype(np.uint8)

    @staticmethod
    def _blend(heatmap, image, cmap=cv2.COLORMAP_INFERNO, alpha=0.5):
        h, w = image.shape[:2]
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        lo, hi = heatmap.min(), heatmap.max()
        norm = (heatmap - lo) / (hi - lo + 1e-8)
        cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
        return cv2.addWeighted(image, 1 - alpha, cv2.cvtColor(cm, cv2.COLOR_BGR2RGB), alpha, 0)

    def _annotate(self, image, method_key, title_text, extra=""):
        """Add title bar, colorbar (if applicable), and info panel."""
        h, w = image.shape[:2]
        info = METHOD_INFO.get(method_key, {})

        # Title bar
        bar = np.full((44, w, 3), 30, dtype=np.uint8)
        cv2.putText(bar, title_text, (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        vis = np.vstack([bar, image])

        # Colorbar for heatmap methods
        cmap = info.get("cmap")
        if cmap is not None:
            vis = _draw_colorbar(vis, cmap)

        # Info panel
        vis = _draw_info_panel(vis, method_key, extra)
        return vis

    def _error_image(self, image, method_key, error_msg):
        """Generate a clear error visualization instead of returning raw image."""
        h, w = image.shape[:2]
        overlay = image.copy()
        # Dark overlay
        overlay = (overlay * 0.3).astype(np.uint8)
        # Error text
        info = METHOD_INFO.get(method_key, {})
        title = info.get("title", method_key)
        cv2.putText(overlay, f"{title}: FAILED", (20, h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 80), 2, cv2.LINE_AA)
        # Wrap error message
        max_chars = max(1, (w - 40) // 10)
        lines = [error_msg[i:i+max_chars] for i in range(0, min(len(error_msg), max_chars * 3), max_chars)]
        for i, line in enumerate(lines):
            cv2.putText(overlay, line, (20, h // 2 + 15 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return overlay

    def _fallback(self, image, msg, method_key=""):
        return {"visualization": self._error_image(image, method_key, msg),
                "report": f"Error ({method_key}): {msg}"}

    # ------------------------------------------------------------------
    # 1. Self-Attention Maps
    # ------------------------------------------------------------------

    def self_attention_maps(self, image, layer=-1, head=None):
        t0 = time.perf_counter()
        # Default to last ENCODER layer for clean patch-only attention
        idx = layer if layer >= 0 else self._n_encoder - 1
        idx = min(idx, self._num_layers - 1)
        is_dec = idx >= self._n_encoder

        inputs, oh, ow = self._preprocess(image)
        gh, gw = self._patch_grid(inputs)

        hook = AttentionCaptureHook().register(self.model.layers[idx].attention)
        try:
            with _eager_attention(self.model), torch.no_grad():
                self.model(**inputs)
        finally:
            hook.remove()

        if hook.weights is None:
            return self._fallback(image, "Attention weights not captured (SDPA fallback)", "attention")

        attn = hook.weights[0]  # (heads, seq, seq)
        if head is not None and 0 <= head < attn.shape[0]:
            a = attn[head]
            hl = f"Head {head}"
        else:
            a = attn.mean(dim=0)
            hl = "Mean"

        spatial, gh, gw = self._cls_to_patches(a, gh, gw, is_dec)

        # Compute attention statistics
        attn_max = float(spatial.max())
        attn_mean = float(spatial.mean())
        focus_pct = float((spatial > spatial.mean() + spatial.std()).mean() * 100)

        blended = self._blend(spatial, image)
        extra = f"Layer {idx}/{self._num_layers-1} | {hl} | Focus: {focus_pct:.0f}% above mean"
        vis = self._annotate(blended, "attention", f"Self-Attention | Layer {idx} | {hl}", extra)

        elapsed = time.perf_counter() - t0
        del hook.weights
        gc.collect()
        return {
            "visualization": vis,
            "report": (
                f"**Self-Attention** (Layer {idx}, {hl}): "
                f"Grid {gh}x{gw}, peak={attn_max:.4f}, mean={attn_mean:.4f}, "
                f"focused area={focus_pct:.1f}%. "
                f"{'Decoder' if is_dec else 'Encoder'} layer. {elapsed:.1f}s"
            ),
        }

    # ------------------------------------------------------------------
    # 2. Attention Rollout (encoder layers only)
    # ------------------------------------------------------------------

    def attention_rollout(self, image, head_fusion="mean", discard_ratio=0.1):
        t0 = time.perf_counter()
        inputs, oh, ow = self._preprocess(image)
        gh, gw = self._patch_grid(inputs)

        n_enc = self._n_encoder
        hooks = [AttentionCaptureHook().register(self.model.layers[i].attention) for i in range(n_enc)]

        try:
            with _eager_attention(self.model), torch.no_grad():
                self.model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        rollout = None
        captured = 0
        for h in hooks:
            if h.weights is None:
                continue
            captured += 1
            attn = h.weights[0]
            if head_fusion == "max":
                fused = attn.max(dim=0).values
            elif head_fusion == "min":
                fused = attn.min(dim=0).values
            else:
                fused = attn.mean(dim=0)

            fused = fused + torch.eye(fused.shape[0])
            fused = fused / fused.sum(dim=-1, keepdim=True)

            if rollout is None:
                rollout = fused
            else:
                if rollout.shape != fused.shape:
                    logger.warning(f"[XAI] Rollout shape mismatch: {rollout.shape} vs {fused.shape}, stopping")
                    break
                rollout = rollout @ fused
            h.weights = None

        del hooks
        gc.collect()

        if rollout is None:
            return self._fallback(image, f"Rollout failed (0/{n_enc} layers captured)", "rollout")

        spatial, gh, gw = self._cls_to_patches(rollout, gh, gw, is_decoder=False)
        if discard_ratio > 0:
            thr = np.percentile(spatial, discard_ratio * 100)
            spatial = np.where(spatial > thr, spatial, 0)

        focus_pct = float((spatial > spatial.mean()).mean() * 100)
        blended = self._blend(spatial, image)
        extra = f"{captured}/{n_enc} layers | {head_fusion} fusion | Focused: {focus_pct:.0f}%"
        vis = self._annotate(blended, "rollout", f"Attention Rollout | {captured} Layers", extra)

        elapsed = time.perf_counter() - t0
        del rollout
        gc.collect()
        return {
            "visualization": vis,
            "report": (
                f"**Attention Rollout**: {captured}/{n_enc} encoder layers, "
                f"{head_fusion} head fusion, discard bottom {discard_ratio*100:.0f}%. "
                f"Focused area={focus_pct:.1f}%. {elapsed:.1f}s"
            ),
        }

    # ------------------------------------------------------------------
    # 3. Predictive Entropy
    # ------------------------------------------------------------------

    def predictive_entropy(self, image):
        t0 = time.perf_counter()
        inputs, oh, ow = self._preprocess(image)

        with torch.no_grad():
            outputs = self.model(**inputs)

        mp = torch.sigmoid(outputs.masks_queries_logits[0])
        cp = F.softmax(outputs.class_queries_logits[0][:, :-1], dim=-1)
        nc = cp.shape[-1]

        pp = torch.einsum("qhw,qc->chw", mp, cp)
        pp = pp / (pp.sum(dim=0, keepdim=True) + 1e-10)
        ent = -(pp * torch.log(pp + 1e-10)).sum(dim=0)
        ent_norm = (ent / np.log(nc)).numpy()
        ent_map = cv2.resize(ent_norm, (ow, oh), interpolation=cv2.INTER_CUBIC)

        me = float(ent_map.mean())
        hp = float((ent_map > 0.5).mean() * 100)
        conf_label = "HIGH" if me < 0.15 else "MODERATE" if me < 0.35 else "LOW"

        blended = self._blend(ent_map, image, cv2.COLORMAP_MAGMA, 0.55)
        extra = f"Mean H={me:.3f} | High-unc={hp:.1f}% | Confidence: {conf_label}"
        vis = self._annotate(blended, "entropy", "Predictive Entropy", extra)

        elapsed = time.perf_counter() - t0
        del outputs, mp, cp, pp, ent
        gc.collect()
        return {
            "visualization": vis,
            "entropy_map": ent_map,
            "report": (
                f"**Predictive Entropy**: Mean={me:.3f}, high-uncertainty={hp:.1f}%, "
                f"confidence={conf_label}. Bright regions = class boundaries or ambiguity. {elapsed:.1f}s"
            ),
        }

    # ------------------------------------------------------------------
    # 4. Feature PCA
    # ------------------------------------------------------------------

    def feature_pca(self, image, layer=-1):
        t0 = time.perf_counter()
        idx = layer if layer >= 0 else self._n_encoder - 1
        idx = min(idx, self._num_layers - 1)

        inputs, oh, ow = self._preprocess(image)
        gh, gw = self._patch_grid(inputs)

        hook = HiddenStateCaptureHook().register(self.model.layers[idx])
        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            hook.remove()

        if hook.state is None:
            return self._fallback(image, "Hidden state not captured", "pca")

        hidden = hook.state[0].numpy()  # (seq_len, hidden_dim)
        n_pre = self._num_prefix
        seq_len = hidden.shape[0]

        is_dec = idx >= self._n_encoder
        n_patches_avail = seq_len - n_pre - (self._num_queries if is_dec else 0)
        side = int(round(np.sqrt(max(n_patches_avail, 1))))
        n_use = side * side
        gh, gw = side, side

        feats = hidden[n_pre:n_pre + n_use]
        centered = feats - feats.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(centered, full_matrices=False)
        comps = U[:, :3] * S[:3]

        for c in range(3):
            lo, hi = comps[:, c].min(), comps[:, c].max()
            comps[:, c] = (comps[:, c] - lo) / (hi - lo + 1e-8) * 255

        pca_img = cv2.resize(comps.reshape(gh, gw, 3).astype(np.uint8), (ow, oh), interpolation=cv2.INTER_CUBIC)

        tv = (S ** 2).sum()
        v3 = (S[:3] ** 2) / tv * 100

        # PCA doesn't use _blend (it's already RGB), so annotate manually
        title_bar = np.full((44, ow, 3), 30, dtype=np.uint8)
        cv2.putText(title_bar, f"Feature PCA | Layer {idx}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        vis = np.vstack([title_bar, pca_img])

        # PCA legend panel
        legend_h = 60
        legend = np.full((legend_h, ow, 3), 25, dtype=np.uint8)
        cv2.putText(legend, f"PC1(R)={v3[0]:.1f}%  PC2(G)={v3[1]:.1f}%  PC3(B)={v3[2]:.1f}%", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(legend, "Similar colors = similar learned representations | SVD of 1024-dim features",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)
        # Color swatches
        sw = 18
        cv2.rectangle(legend, (ow - 3*(sw+5) - 10, 5), (ow - 2*(sw+5) - 10, 5+sw), (255, 0, 0), -1)
        cv2.rectangle(legend, (ow - 2*(sw+5) - 10, 5), (ow - 1*(sw+5) - 10, 5+sw), (0, 255, 0), -1)
        cv2.rectangle(legend, (ow - 1*(sw+5) - 10, 5), (ow - 10, 5+sw), (0, 0, 255), -1)
        vis = np.vstack([vis, legend])

        elapsed = time.perf_counter() - t0
        del hook.state
        gc.collect()
        return {
            "visualization": vis,
            "report": (
                f"**Feature PCA** (Layer {idx}): "
                f"PC1={v3[0]:.1f}%, PC2={v3[1]:.1f}%, PC3={v3[2]:.1f}% variance explained. "
                f"Regions with similar colors share similar learned feature representations. {elapsed:.1f}s"
            ),
        }

    # ------------------------------------------------------------------
    # FP32 model for gradient methods
    # ------------------------------------------------------------------

    def _get_fp32(self):
        if self._model_fp32 is not None:
            return self._model_fp32
        logger.info("[XAI] Loading FP32 model for gradient methods...")
        from transformers import AutoModelForUniversalSegmentation
        mid = getattr(self.model.config, "_name_or_path", "tue-mps/ade20k_semantic_eomt_large_512")
        self._model_fp32 = AutoModelForUniversalSegmentation.from_pretrained(mid)
        self._model_fp32.eval()
        self._model_fp32.config._attn_implementation = "eager"
        logger.info("[XAI] FP32 loaded (eager attn)")
        return self._model_fp32

    def cleanup_fp32(self):
        if self._model_fp32 is not None:
            del self._model_fp32
            self._model_fp32 = None
            gc.collect()
            logger.info("[XAI] FP32 released")

    def _fp32_forward(self, inputs, need_grad_pv=False):
        """Run FP32 forward pass with all inputs. Optionally enable grad on pixel_values.

        Returns (outputs, pixel_values_tensor).
        pixel_values_tensor will have requires_grad=True if need_grad_pv is set.
        """
        fp32 = self._get_fp32()
        fwd = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if need_grad_pv:
            fwd["pixel_values"] = fwd["pixel_values"].requires_grad_(True)
        outputs = fp32(**fwd)
        return outputs, fwd.get("pixel_values")

    # ------------------------------------------------------------------
    # 5. GradCAM
    # ------------------------------------------------------------------

    def gradcam_segmentation(self, image, target_class_id=None):
        t0 = time.perf_counter()
        fp32 = self._get_fp32()
        inputs, oh, ow = self._preprocess(image)
        gh, gw = self._patch_grid(inputs)

        target_layer_idx = self._n_encoder - 1
        hook = ActivationGradientHook().register(fp32.layers[target_layer_idx].mlp)

        try:
            outputs = fp32(**inputs)
            seg = self._seg_from_outputs(outputs, oh, ow)
            if target_class_id is None:
                target_class_id = self._dominant_class(seg)
            cn = self._cname(target_class_id)

            ml = outputs.masks_queries_logits[0]
            cl = outputs.class_queries_logits[0]
            preds = cl[:, :-1].argmax(dim=-1)
            qs = (preds == target_class_id).nonzero(as_tuple=True)[0]
            if len(qs) == 0:
                qs = cl[:, target_class_id].argmax(dim=0, keepdim=True)

            ml[qs].sum().backward()

            if hook.activation is not None and hook.gradient is not None:
                act = hook.activation
                grad = hook.gradient
                w = grad.mean(dim=-1, keepdim=True)
                cam = F.relu((act * w).sum(dim=-1))
                cam_np = cam[0].detach().cpu().numpy()

                n_pre = self._num_prefix
                seq = cam_np.shape[0]
                n_avail = seq - n_pre
                side = int(round(np.sqrt(max(n_avail, 1))))
                n_use = side * side
                spatial = cam_np[n_pre:n_pre + n_use].reshape(side, side)
                blended = self._blend(spatial, image, cv2.COLORMAP_JET)
                cam_max = float(spatial.max())
                hot_pct = float((spatial > spatial.mean() + spatial.std()).mean() * 100)
            else:
                blended = image.copy()
                cn = "N/A (no gradient)"
                cam_max = 0
                hot_pct = 0
        finally:
            hook.remove()
            fp32.zero_grad()

        extra = f"Target: {cn} (ID {target_class_id}) | Peak={cam_max:.3f} | Hot area={hot_pct:.0f}%"
        vis = self._annotate(blended, "gradcam", f"GradCAM | {cn}", extra)

        elapsed = time.perf_counter() - t0
        gc.collect()
        return {
            "visualization": vis,
            "target_class": target_class_id,
            "report": (
                f"**GradCAM** (target: '{cn}', ID {target_class_id}): "
                f"Encoder layer {target_layer_idx}, peak activation={cam_max:.3f}, "
                f"hot area={hot_pct:.1f}%. Red/yellow = strongest class activation. {elapsed:.1f}s"
            ),
        }

    # ------------------------------------------------------------------
    # 6. Integrated Gradients (replaces vanilla saliency)
    # ------------------------------------------------------------------

    def integrated_gradients(self, image, target_class_id=None, n_steps=8):
        """Sundararajan et al. (2017) — principled attribution via path integral.

        Computes attribution by integrating gradients along the straight-line
        path from a zero baseline to the actual input. Satisfies completeness
        and sensitivity axioms (unlike vanilla saliency).

        Uses batched interpolation: processes all steps in a single forward pass
        through the model, then accumulates gradients. Much faster than per-step.

        Args:
            n_steps: Number of interpolation steps (8 = good quality/speed tradeoff on CPU).
        """
        t0 = time.perf_counter()
        fp32 = self._get_fp32()
        inputs, oh, ow = self._preprocess(image)

        # First pass to determine target class
        with torch.no_grad():
            outputs_ref = fp32(**inputs)
        seg = self._seg_from_outputs(outputs_ref, oh, ow)
        if target_class_id is None:
            target_class_id = self._dominant_class(seg)
        cn = self._cname(target_class_id)
        del outputs_ref
        gc.collect()

        pv = inputs["pixel_values"]  # (1, 3, H, W)
        baseline = torch.zeros_like(pv)  # black baseline

        # Accumulate gradients along interpolation path
        ig_grads = torch.zeros_like(pv)
        for step in range(n_steps + 1):
            alpha = step / n_steps
            interp = (baseline + alpha * (pv - baseline)).detach().requires_grad_(True)

            fwd = {k: v.clone() if isinstance(v, torch.Tensor) else v
                   for k, v in inputs.items()}
            fwd["pixel_values"] = interp

            outputs = fp32(**fwd)
            score = outputs.class_queries_logits[0][:, target_class_id].sum()
            score.backward()

            if interp.grad is not None:
                ig_grads += interp.grad.detach()

            fp32.zero_grad()
            del outputs, score, fwd, interp
            if step % 3 == 0:
                gc.collect()

            logger.debug(f"[XAI] IG step {step}/{n_steps} ({time.perf_counter()-t0:.0f}s)")

        # Riemann sum approximation: IG = (input - baseline) * mean(gradients)
        ig_attr = (pv - baseline) * ig_grads / (n_steps + 1)
        attr_map = ig_attr[0].abs().max(dim=0).values.detach().cpu().numpy()
        attr_map = cv2.resize(attr_map, (ow, oh), interpolation=cv2.INTER_CUBIC)

        attr_max = float(attr_map.max())
        attr_sum = float(ig_attr.sum())
        significant_pct = float((attr_map > attr_map.mean() + attr_map.std()).mean() * 100)

        blended = self._blend(attr_map, image, cv2.COLORMAP_HOT)
        extra = f"Target: {cn} | {n_steps} steps | Significant={significant_pct:.0f}% | Sum={attr_sum:.2f}"
        vis = self._annotate(blended, "integrated_gradients",
                             f"Integrated Gradients | {cn}", extra)

        elapsed = time.perf_counter() - t0
        del ig_grads, ig_attr, baseline
        gc.collect()
        return {
            "visualization": vis,
            "attr_map": attr_map,
            "report": (
                f"**Integrated Gradients** (target: '{cn}', ID {target_class_id}): "
                f"{n_steps} steps, peak={attr_max:.4f}, "
                f"significant area={significant_pct:.1f}%, attribution sum={attr_sum:.2f}. "
                f"Satisfies completeness axiom (Sundararajan 2017). "
                f"Bright = pixels most influencing '{cn}' prediction. {elapsed:.1f}s"
            ),
        }

    # ------------------------------------------------------------------
    # 7. Chefer Relevancy
    # ------------------------------------------------------------------

    def chefer_relevancy(self, image, target_class_id=None):
        t0 = time.perf_counter()
        fp32 = self._get_fp32()
        inputs, oh, ow = self._preprocess(image)
        gh, gw = self._patch_grid(inputs)

        layer_data = []
        handles = []
        for i in range(self._n_encoder):
            data = {"attn": None, "grad": None}

            def mk_fwd(d):
                def fn(m, a, o):
                    if isinstance(o, tuple) and len(o) >= 2 and o[1] is not None:
                        d["attn"] = o[1]
                return fn

            def mk_bwd(d):
                def fn(m, gi, go):
                    if len(go) >= 2 and go[1] is not None:
                        d["grad"] = go[1]
                return fn

            mod = fp32.layers[i].attention
            handles.append(mod.register_forward_hook(mk_fwd(data)))
            handles.append(mod.register_full_backward_hook(mk_bwd(data)))
            layer_data.append(data)

        try:
            outputs = fp32(**inputs)
            seg = self._seg_from_outputs(outputs, oh, ow)
            if target_class_id is None:
                target_class_id = self._dominant_class(seg)
            cn = self._cname(target_class_id)

            outputs.class_queries_logits[0][:, target_class_id].sum().backward()

            first_attn = next((d["attn"] for d in layer_data if d["attn"] is not None), None)
            if first_attn is None:
                return self._fallback(image, "No attention captured for Chefer", "chefer")

            seq = first_attn.shape[-1]
            R = torch.eye(seq)
            layers_used = 0

            for data in layer_data:
                attn = data.get("attn")
                if attn is None:
                    continue
                layers_used += 1
                am = attn[0].detach().cpu().mean(dim=0)
                grad = data.get("grad")
                if grad is not None:
                    gm = grad[0].detach().cpu().mean(dim=0)
                    rel = torch.clamp(am * gm, min=0)
                else:
                    rel = am

                s = min(rel.shape[0], seq)
                rel = rel[:s, :s] + torch.eye(s)
                rel = rel / (rel.sum(dim=-1, keepdim=True) + 1e-10)
                R[:s, :s] = R[:s, :s] + R[:s, :s] @ rel

            spatial, gh, gw = self._cls_to_patches(R, gh, gw, is_decoder=False)
            blended = self._blend(spatial, image)

            # Threshold overlay — green tint on relevant regions
            thr = spatial.mean()
            mask = cv2.resize((spatial > thr).astype(np.float32), (ow, oh), interpolation=cv2.INTER_NEAREST)
            ov = blended.copy()
            ov[mask > 0.5] = (ov[mask > 0.5] * 0.7 + np.array([0, 80, 0]) * 0.3).astype(np.uint8)
            relevance_pct = float((spatial > thr).mean() * 100)

        finally:
            for h in handles:
                h.remove()
            fp32.zero_grad()

        extra = f"Target: {cn} | {layers_used} layers | Relevant area={relevance_pct:.0f}% (green tint)"
        vis = self._annotate(ov, "chefer", f"Chefer Relevancy | {cn}", extra)

        elapsed = time.perf_counter() - t0
        gc.collect()
        return {
            "visualization": vis,
            "report": (
                f"**Chefer Relevancy** (target: '{cn}', ID {target_class_id}): "
                f"{layers_used}/{self._n_encoder} encoder layers, "
                f"relevant area={relevance_pct:.1f}%. "
                f"Green overlay = model-relevant regions. {elapsed:.1f}s"
            ),
        }

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_xai_report(self, image, seg_mask=None, method_results=None):
        """Generate a comprehensive cross-method XAI analysis report.

        When method_results is provided (from run_full_analysis), produces
        deeper cross-method correlation analysis and practical insights.
        """
        t0 = time.perf_counter()

        if seg_mask is None:
            inputs, oh, ow = self._preprocess(image)
            with torch.no_grad():
                outputs = self.model(**inputs)
            seg_mask = self._seg_from_outputs(outputs, oh, ow)
            del outputs
            gc.collect()

        oh, ow = seg_mask.shape[:2]
        cls, cnt = np.unique(seg_mask, return_counts=True)
        total = seg_mask.size
        info = sorted(zip(cls, cnt), key=lambda x: -x[1])

        # --- Entropy analysis ---
        ent_result = None
        emap = None
        me, hp = 0, 0
        if method_results and "entropy" in method_results:
            ent_result = method_results["entropy"]
            emap = ent_result.get("entropy_map")
        if emap is None:
            ent_result = self.predictive_entropy(image)
            emap = ent_result.get("entropy_map")

        if emap is not None:
            me = float(emap.mean())
            hp = float((emap > 0.5).mean() * 100)

        # --- Boundary uncertainty analysis ---
        boundary_info = []
        if emap is not None:
            try:
                from skimage.segmentation import find_boundaries
                for cid, _ in info[:8]:
                    bd = find_boundaries(seg_mask == cid, mode="thick")
                    if bd.any():
                        be = float(emap[bd].mean())
                        interior = (seg_mask == cid) & ~bd
                        ie = float(emap[interior].mean()) if interior.any() else 0
                        boundary_info.append({
                            "name": self._cname(cid),
                            "id": int(cid),
                            "boundary_entropy": be,
                            "interior_entropy": ie,
                            "area_pct": float(cnt[list(cls).index(cid)] / total * 100),
                            "confidence_ratio": ie / (be + 1e-8),
                        })
            except ImportError:
                pass

        lines = [
            "# Comprehensive XAI Analysis Report\n",
        ]

        # --- 1. Scene Understanding ---
        lines += [
            "## 1. Scene Composition & Object Distribution",
            f"The model identified **{len(cls)} semantic categories** in this scene, "
            f"processed through {self._n_encoder} encoder layers and {self._num_layers - self._n_encoder} decoder layers.\n",
        ]
        for cid, c in info[:10]:
            p = c / total * 100
            bar = "\u2588" * int(p / 4) + "\u2591" * max(0, 25 - int(p / 4))
            lines.append(f"- **{self._cname(cid)}** (ID {cid}): {p:.1f}% `{bar}`")

        if len(info) > 10:
            rest_pct = sum(c for _, c in info[10:]) / total * 100
            lines.append(f"- *{len(info)-10} more categories*: {rest_pct:.1f}% combined")

        # --- 2. Model Confidence ---
        lines += [
            "\n## 2. Prediction Confidence Analysis",
            f"- **Mean entropy**: {me:.3f} (range 0-1, lower = more confident)",
            f"- **High-uncertainty pixels**: {hp:.1f}% of image",
        ]
        if me < 0.10:
            lines.append("- Assessment: **Very high confidence** — the model strongly recognizes all scene elements")
        elif me < 0.20:
            lines.append("- Assessment: **High confidence** — clear semantic boundaries with minor ambiguity")
        elif me < 0.35:
            lines.append("- Assessment: **Moderate confidence** — some object boundaries or textures cause uncertainty")
        else:
            lines.append("- Assessment: **Low confidence** — significant regions where the model is uncertain")

        # --- 3. Boundary Analysis ---
        if boundary_info:
            lines += ["\n## 3. Boundary Clarity & Transition Zones"]
            uncertain = [b for b in boundary_info if b["boundary_entropy"] > 0.25]
            clear = [b for b in boundary_info if b["boundary_entropy"] <= 0.25]

            if uncertain:
                lines.append("\n**Challenging boundaries** (high entropy at transitions):")
                for b in sorted(uncertain, key=lambda x: -x["boundary_entropy"])[:5]:
                    lines.append(
                        f"- **{b['name']}** boundaries: entropy={b['boundary_entropy']:.3f} "
                        f"(interior={b['interior_entropy']:.3f}, ratio={b['confidence_ratio']:.2f})"
                    )
                lines.append(
                    "\n> *High boundary entropy suggests the model struggles to delineate these objects "
                    "from their surroundings — a pattern also observed in human visual perception "
                    "for Alzheimer's patients with low-contrast environments.*"
                )
            if clear:
                lines.append("\n**Well-defined boundaries** (model is confident):")
                for b in sorted(clear, key=lambda x: x["boundary_entropy"])[:3]:
                    lines.append(
                        f"- **{b['name']}**: boundary entropy={b['boundary_entropy']:.3f} "
                        f"(strong semantic contrast with neighbors)"
                    )

        # --- 4. Cross-Method Insights ---
        lines += ["\n## 4. Cross-Method Interpretation"]

        # Attention insights
        has_attn = method_results and "attention" in method_results and "Error" not in method_results["attention"].get("report", "")
        has_rollout = method_results and "rollout" in method_results and "Error" not in method_results["rollout"].get("report", "")
        has_gradcam = method_results and "gradcam" in method_results and "Error" not in method_results["gradcam"].get("report", "")
        has_ig = method_results and "integrated_gradients" in method_results and "Error" not in method_results["integrated_gradients"].get("report", "")
        has_chefer = method_results and "chefer" in method_results and "Error" not in method_results["chefer"].get("report", "")
        has_pca = method_results and "pca" in method_results and "Error" not in method_results["pca"].get("report", "")

        if has_attn and has_rollout:
            lines += [
                "\n### Attention Analysis (Self-Attention + Rollout)",
                "- **Self-Attention** shows where a *single layer* focuses — useful for understanding "
                "what features are processed at different network depths",
                "- **Attention Rollout** aggregates across *all 20 encoder layers*, revealing the "
                "model's overall spatial priority. The cumulative attention reveals which scene "
                "regions are most \"important\" to the final representation",
                "- If rollout focuses on object centers while self-attention spreads across edges, "
                "the model has learned strong object-level representations",
            ]

        if has_gradcam and has_ig:
            lines += [
                "\n### Gradient Attribution (GradCAM + Integrated Gradients)",
                "- **GradCAM** highlights the *internal feature map* regions that activate for a class "
                "— operates in the model's latent space (coarse spatial resolution)",
                "- **Integrated Gradients** attributes importance to *individual input pixels* "
                "via a principled path integral — operates in pixel space (full resolution)",
                "- Agreement between both methods indicates robust class localization. "
                "Disagreement may reveal that the model uses contextual cues (e.g., detecting "
                "'floor' partly from seeing 'furniture' nearby)",
            ]
        elif has_ig:
            lines += [
                "\n### Gradient Attribution (Integrated Gradients)",
                "- Attributions satisfying the *completeness axiom*: total attribution equals the "
                "difference between the model's prediction on the input vs. a black baseline",
                "- Bright regions show which pixels most influence the target class prediction",
            ]

        if has_chefer:
            lines += [
                "\n### Transformer-Specific Attribution (Chefer Relevancy)",
                "- Combines attention patterns *with* gradient flow across all layers — the most "
                "theoretically grounded method for Vision Transformers",
                "- Green overlay indicates regions the model deems \"relevant\" to its prediction",
                "- Complements GradCAM by capturing token-level interactions specific to "
                "the self-attention architecture",
            ]

        if has_pca:
            lines += [
                "\n### Learned Feature Structure (Feature PCA)",
                "- Projects 1024-dimensional hidden states to 3 principal components (RGB channels)",
                "- Regions with **similar colors** share similar learned representations — "
                "revealing how the model internally groups scene elements",
                "- If floor and furniture have distinct colors, the model has learned to differentiate "
                "them at the feature level (critical for accurate blackspot detection)",
            ]

        # --- 5. Accessibility Implications ---
        lines += ["\n## 5. Implications for Accessibility Analysis"]
        if boundary_info:
            floor_related = [b for b in boundary_info
                             if any(f in b["name"].lower() for f in ["floor", "carpet", "rug", "mat"])]
            if floor_related:
                lines.append("\n**Floor boundary analysis:**")
                for b in floor_related:
                    if b["boundary_entropy"] > 0.3:
                        lines.append(
                            f"- **{b['name']}** has uncertain boundaries (entropy={b['boundary_entropy']:.3f}) — "
                            f"this suggests low visual contrast that may also challenge Alzheimer's patients"
                        )
                    else:
                        lines.append(
                            f"- **{b['name']}** has clear boundaries (entropy={b['boundary_entropy']:.3f}) — "
                            f"model can easily distinguish this surface from surroundings"
                        )

        lines += [
            "\n> *The XAI visualizations above reveal how DINOv3-EoMT processes scene geometry. "
            "Regions where the model shows high uncertainty or diffuse attention correlate with "
            "areas that may present visual navigation challenges for individuals with "
            "Alzheimer's-related perceptual deficits.*",
        ]

        # --- 6. Method Reference ---
        lines += [
            "\n## 6. Method Reference",
            "| Method | Type | Key Property | Reference |",
            "|--------|------|-------------|-----------|",
            "| Self-Attention | Attention | Single-layer spatial focus | Vaswani et al. (2017) |",
            "| Attention Rollout | Attention | Multi-layer cumulative focus | Abnar & Zuidema (2020) |",
            "| GradCAM | Gradient | Class-discriminative localization | Selvaraju et al. (2017) |",
            "| Predictive Entropy | Output | Per-pixel classification uncertainty | Shannon (1948) |",
            "| Feature PCA | Hidden State | Learned representation structure | — |",
            "| Integrated Gradients | Gradient | Axiom-satisfying pixel attribution | Sundararajan et al. (2017) |",
            "| Chefer Relevancy | Attn x Grad | ViT-specific relevance propagation | Chefer et al. (2021) |",
            f"\n**Architecture**: DINOv3-EoMT-Large — {self._num_layers} layers "
            f"({self._n_encoder} encoder + {self._num_layers - self._n_encoder} decoder), "
            f"{self._num_heads} heads, {self._patch_size}px patches, 150 ADE20K classes",
        ]

        elapsed = time.perf_counter() - t0
        vis = ent_result["visualization"] if ent_result else None
        return {"visualization": vis, "report": "\n".join(lines)}

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run_full_analysis(self, image, layer=-1, head=None, target_class_id=None,
                          progress_callback=None):
        """Run all 7 methods. Each method is isolated — one failure won't block others.

        Args:
            progress_callback: Optional callable(fraction, description) for progress updates.
        """
        logger.info("[XAI] Running full analysis suite...")
        t0 = time.perf_counter()
        results = {}

        def _progress(frac, msg):
            if progress_callback:
                try:
                    progress_callback(frac, desc=msg)
                except Exception:
                    pass
            logger.info(f"[XAI] [{frac*100:.0f}%] {msg}")

        methods = [
            ("attention", self.self_attention_maps, {"layer": layer, "head": head}),
            ("rollout",   self.attention_rollout,   {}),
            ("entropy",   self.predictive_entropy,  {}),
            ("pca",       self.feature_pca,         {"layer": layer}),
            ("gradcam",   self.gradcam_segmentation, {"target_class_id": target_class_id}),
            ("integrated_gradients", self.integrated_gradients, {"target_class_id": target_class_id}),
            ("chefer",    self.chefer_relevancy,      {"target_class_id": target_class_id}),
        ]

        for i, (key, fn, kw) in enumerate(methods):
            _progress(i / len(methods), f"Running {key}...")
            try:
                results[key] = fn(image, **kw)
            except Exception as e:
                logger.error(f"[XAI] {key} failed: {e}\n{traceback.format_exc()}")
                results[key] = self._fallback(image, str(e), key)
            gc.collect()

        _progress(0.9, "Generating report...")
        try:
            results["report"] = self.generate_xai_report(image, method_results=results)
        except Exception as e:
            logger.error(f"[XAI] report failed: {e}")
            results["report"] = {"visualization": None, "report": f"Report generation failed: {e}"}

        elapsed = time.perf_counter() - t0
        _progress(1.0, f"Complete ({elapsed:.0f}s)")
        logger.info(f"[XAI] Full analysis done in {elapsed:.1f}s")

        # Count successes
        ok = sum(1 for k in ["attention", "rollout", "entropy", "pca", "gradcam", "integrated_gradients", "chefer"]
                 if k in results and "Error" not in results[k].get("report", ""))
        logger.info(f"[XAI] {ok}/7 methods succeeded")

        return results

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key_from_file(image_path: str) -> str:
        """Generate a stable cache key by hashing image FILE CONTENT.

        This ensures the same image always maps to the same cache key,
        regardless of the file path (Gradio temp paths, sample paths, etc.).
        """
        import hashlib
        h = hashlib.md5()
        try:
            with open(image_path, "rb") as f:
                # Read first 256KB — enough to uniquely identify any image
                h.update(f.read(256 * 1024))
        except (OSError, IOError):
            # Fallback to path-based key
            h.update(image_path.encode())
        return h.hexdigest()[:12]

    # Keep np array version for when we only have the image in memory
    @staticmethod
    def _cache_key_from_array(image: np.ndarray) -> str:
        import hashlib
        h = hashlib.md5()
        h.update(image.tobytes()[:256 * 1024])
        return h.hexdigest()[:12]

    _CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xai_cache")
    _METHOD_KEYS = ["attention", "rollout", "gradcam", "entropy",
                    "pca", "integrated_gradients", "chefer"]

    def save_results(self, results: dict, image_path: str = None, image: np.ndarray = None):
        """Save XAI results (visualizations as PNG, reports as text) to disk.

        Uses content-based hashing so the same image always maps to the same cache.
        """
        if image_path:
            key = self._cache_key_from_file(image_path)
        elif image is not None:
            key = self._cache_key_from_array(image)
        else:
            logger.warning("[XAI] save_results: no image_path or image provided")
            return None

        out = os.path.join(self._CACHE_DIR, key)
        os.makedirs(out, exist_ok=True)

        saved = 0
        for method_key in self._METHOD_KEYS:
            r = results.get(method_key, {})
            vis = r.get("visualization")
            if vis is not None:
                try:
                    Image.fromarray(vis).save(os.path.join(out, f"{method_key}.png"), optimize=True)
                    saved += 1
                except Exception as e:
                    logger.warning(f"[XAI] Cache save failed for {method_key}: {e}")
            report = r.get("report", "")
            if report:
                with open(os.path.join(out, f"{method_key}.txt"), "w") as f:
                    f.write(report)

        full_report = results.get("report", {}).get("report", "")
        if full_report:
            with open(os.path.join(out, "full_report.md"), "w") as f:
                f.write(full_report)

        logger.info(f"[XAI] Cached {saved} visualizations to {out} (key={key})")
        return out

    @staticmethod
    def load_cached_from_file(image_path: str):
        """Load cached XAI results by image file content hash."""
        key = XAIAnalyzer._cache_key_from_file(image_path)
        return XAIAnalyzer._load_cache_dir(
            os.path.join(XAIAnalyzer._CACHE_DIR, key)
        )

    @staticmethod
    def load_cached_from_array(image: np.ndarray):
        """Load cached XAI results by image array content hash."""
        key = XAIAnalyzer._cache_key_from_array(image)
        return XAIAnalyzer._load_cache_dir(
            os.path.join(XAIAnalyzer._CACHE_DIR, key)
        )

    @staticmethod
    def _load_cache_dir(cache_path: str):
        """Load cached results from a specific cache directory."""
        if not os.path.isdir(cache_path):
            return None

        results = {}
        for method_key in XAIAnalyzer._METHOD_KEYS:
            r = {}
            png = os.path.join(cache_path, f"{method_key}.png")
            if os.path.exists(png):
                try:
                    r["visualization"] = np.array(Image.open(png).convert("RGB"))
                except Exception as e:
                    logger.warning(f"[XAI] Cache load failed for {method_key}: {e}")
            txt = os.path.join(cache_path, f"{method_key}.txt")
            if os.path.exists(txt):
                with open(txt) as f:
                    r["report"] = f.read()
            if r:
                results[method_key] = r

        rpt = os.path.join(cache_path, "full_report.md")
        if os.path.exists(rpt):
            with open(rpt) as f:
                results["report"] = {"report": f.read()}

        if results:
            logger.info(f"[XAI] Cache hit: {cache_path} ({len(results)-('report' in results)} methods)")
        return results if results else None

    @staticmethod
    def list_cached():
        """List all cached analysis directories."""
        if not os.path.isdir(XAIAnalyzer._CACHE_DIR):
            return []
        return [
            os.path.join(XAIAnalyzer._CACHE_DIR, d)
            for d in os.listdir(XAIAnalyzer._CACHE_DIR)
            if os.path.isdir(os.path.join(XAIAnalyzer._CACHE_DIR, d))
        ]


# ---------------------------------------------------------------------------
# Notebook-style HTML renderer
# ---------------------------------------------------------------------------

# Method display order and metadata for the notebook renderer
_NB_METHODS = [
    {
        "key": "attention",
        "num": 1,
        "category": "Attention-Based",
        "cat_color": "#6366f1",
        "title": "Self-Attention Map",
        "what": "Shows where a single transformer layer focuses its spatial attention.",
        "why": (
            "Each of the 16 attention heads learns different patterns (edges, textures, "
            "object shapes). This map reveals the raw attention distribution before any "
            "aggregation, helping identify what features the network processes at a given depth."
        ),
    },
    {
        "key": "rollout",
        "num": 2,
        "category": "Attention-Based",
        "cat_color": "#6366f1",
        "title": "Attention Rollout",
        "what": "Cumulative attention flow aggregated across all 20 encoder layers.",
        "why": (
            "By multiplying attention matrices layer-by-layer, rollout reveals the model's "
            "overall spatial priority. Bright regions are where information flows to the CLS "
            "token — the model's 'summary' of the scene. This is more stable than single-layer "
            "attention and reflects the full representational pipeline."
        ),
    },
    {
        "key": "gradcam",
        "num": 3,
        "category": "Gradient-Based",
        "cat_color": "#dc2626",
        "title": "GradCAM",
        "what": "Gradient-weighted class activation map — which internal features activate for a class.",
        "why": (
            "GradCAM computes the gradient of the target class score w.r.t. the last encoder "
            "layer's activations, then weights each activation channel by its mean gradient. "
            "Red/yellow regions are where the model's internal representation most strongly "
            "activates for the predicted class. Operates in feature space (coarse resolution)."
        ),
    },
    {
        "key": "entropy",
        "num": 4,
        "category": "Output Analysis",
        "cat_color": "#059669",
        "title": "Predictive Entropy",
        "what": "Per-pixel classification uncertainty — dark = confident, bright = uncertain.",
        "why": (
            "Shannon entropy over the 150-class probability distribution at each pixel. "
            "High entropy at object boundaries reveals where the model struggles to decide "
            "between adjacent categories. This directly correlates with visual transitions that "
            "may challenge individuals with Alzheimer's-related perceptual deficits."
        ),
    },
    {
        "key": "pca",
        "num": 5,
        "category": "Output Analysis",
        "cat_color": "#059669",
        "title": "Feature PCA",
        "what": "Learned feature structure — similar colors indicate similar internal representations.",
        "why": (
            "Projects 1024-dimensional hidden states to 3 principal components mapped to RGB. "
            "Objects with the same color share similar learned features — revealing how the model "
            "internally groups scene elements. Distinct color boundaries between floor and "
            "furniture indicate strong feature-level differentiation (good for blackspot detection)."
        ),
    },
    {
        "key": "integrated_gradients",
        "num": 6,
        "category": "Gradient-Based",
        "cat_color": "#dc2626",
        "title": "Integrated Gradients",
        "what": "Principled pixel-level attribution via path integral from black baseline to input.",
        "why": (
            "Unlike vanilla saliency, Integrated Gradients satisfies the completeness axiom: "
            "attributions sum to the prediction difference. By interpolating from a black baseline "
            "to the actual image in 8 steps, it identifies which specific pixels most influence "
            "the target class prediction. Full input resolution, theoretically sound."
        ),
    },
    {
        "key": "chefer",
        "num": 7,
        "category": "Gradient-Based",
        "cat_color": "#dc2626",
        "title": "Chefer Relevancy",
        "what": "Attention x gradient propagation — transformer-specific relevance attribution.",
        "why": (
            "The most theoretically grounded XAI method for Vision Transformers (Chefer 2021). "
            "Combines attention patterns WITH gradient flow across all encoder layers, propagating "
            "relevance from the classification output back to input tokens. Green overlay shows "
            "regions the model considers 'relevant' to its prediction."
        ),
    },
]


def render_notebook_html(results: dict, original_image: np.ndarray = None) -> str:
    """Render XAI results as scrollable notebook-style HTML.

    Returns HTML string with base64-embedded images, analysis descriptions,
    and the full report in a Jupyter-notebook-style layout.
    """
    import base64
    from io import BytesIO

    def _img_b64(img_array, quality=85):
        if img_array is None:
            return ""
        pil = Image.fromarray(img_array)
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode()

    parts = []

    # --- CSS ---
    parts.append("""<style>
    .nb { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1100px; margin: 0 auto; }
    .nb-cell { margin: 20px 0; padding: 20px; background: #fafbfc; border: 1px solid #e5e7eb; border-radius: 12px; }
    .nb-cell-header { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
    .nb-num { display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; border-radius: 50%; font-size: 0.8em; font-weight: 700; color: white; }
    .nb-title { font-size: 1.15em; font-weight: 700; color: #1f2937; }
    .nb-badge { padding: 2px 10px; border-radius: 12px; font-size: 0.72em; font-weight: 600; color: white; }
    .nb-what { color: #374151; font-size: 0.95em; margin: 8px 0 4px; font-weight: 500; line-height: 1.5; }
    .nb-why { color: #6b7280; font-size: 0.85em; line-height: 1.6; margin: 4px 0 12px; }
    .nb-img { width: 100%; border-radius: 8px; border: 1px solid #d1d5db; margin: 10px 0; }
    .nb-metrics { color: #4b5563; font-size: 0.82em; padding: 8px 12px; background: #f3f4f6; border-radius: 8px; margin-top: 8px; line-height: 1.6; font-family: 'SF Mono', Monaco, monospace; }
    .nb-sep { border: none; border-top: 2px solid #e5e7eb; margin: 28px 0; }
    .nb-section-header { font-size: 1.3em; font-weight: 800; color: #1f2937; margin: 24px 0 8px; display: flex; align-items: center; gap: 8px; }
    .nb-section-desc { color: #6b7280; font-size: 0.9em; margin-bottom: 16px; }
    .nb-report { margin: 24px 0; padding: 24px; background: white; border: 1px solid #e5e7eb; border-radius: 12px; line-height: 1.8; }
    .nb-report h1, .nb-report h2, .nb-report h3 { color: #1f2937; margin-top: 20px; }
    .nb-report table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.9em; }
    .nb-report th, .nb-report td { padding: 8px 12px; border: 1px solid #e5e7eb; text-align: left; }
    .nb-report th { background: #f9fafb; font-weight: 600; }
    .nb-report blockquote { border-left: 3px solid #6366f1; padding: 8px 16px; margin: 12px 0; background: #f5f3ff; border-radius: 0 8px 8px 0; color: #4b5563; }
    .nb-error { color: #dc2626; background: #fef2f2; padding: 12px; border-radius: 8px; border: 1px solid #fecaca; }
    @media (prefers-color-scheme: dark) {
        .nb-cell { background: #1f2937; border-color: #374151; }
        .nb-title { color: #f3f4f6; }
        .nb-what { color: #d1d5db; }
        .nb-why { color: #9ca3af; }
        .nb-metrics { background: #374151; color: #d1d5db; }
        .nb-report { background: #111827; border-color: #374151; }
        .nb-report h1, .nb-report h2, .nb-report h3 { color: #f3f4f6; }
        .nb-report th { background: #374151; }
        .nb-report blockquote { background: #312e81; border-color: #6366f1; color: #c7d2fe; }
        .nb-section-header { color: #f3f4f6; }
        .nb-section-desc { color: #9ca3af; }
    }
    </style>""")

    parts.append('<div class="nb">')

    # --- Original Image ---
    if original_image is not None:
        b64 = _img_b64(original_image)
        parts.append(f"""
        <div class="nb-cell">
            <div class="nb-cell-header">
                <span class="nb-num" style="background:#1f2937;">In</span>
                <span class="nb-title">Input Image</span>
            </div>
            <img class="nb-img" src="data:image/jpeg;base64,{b64}" alt="Input" />
        </div>""")

    # --- Method Cells ---
    current_cat = None
    cat_descs = {
        "Attention-Based": "Where does the model look? These methods visualize spatial attention patterns across transformer layers.",
        "Gradient-Based": "What drives predictions? Gradient-based methods attribute importance to input regions and internal features.",
        "Output Analysis": "How confident is the model? Output and feature analysis reveals prediction certainty and learned representations.",
    }
    for m in _NB_METHODS:
        # Category header
        if m["category"] != current_cat:
            current_cat = m["category"]
            parts.append(f'<hr class="nb-sep">')
            parts.append(f'<div class="nb-section-header"><span style="color:{m["cat_color"]};">&#9679;</span> {current_cat}</div>')
            parts.append(f'<div class="nb-section-desc">{cat_descs.get(current_cat, "")}</div>')

        r = results.get(m["key"], {})
        vis = r.get("visualization")
        report = r.get("report", "")
        is_error = "Error" in report

        parts.append(f"""
        <div class="nb-cell">
            <div class="nb-cell-header">
                <span class="nb-num" style="background:{m['cat_color']};">{m['num']}</span>
                <span class="nb-title">{m['title']}</span>
                <span class="nb-badge" style="background:{m['cat_color']};">{m['category']}</span>
            </div>
            <div class="nb-what">{m['what']}</div>
            <div class="nb-why">{m['why']}</div>
        """)

        if vis is not None:
            b64 = _img_b64(vis)
            parts.append(f'<img class="nb-img" src="data:image/jpeg;base64,{b64}" alt="{m["title"]}" />')
        elif is_error:
            parts.append(f'<div class="nb-error">{report}</div>')
        else:
            parts.append('<div class="nb-error">Not computed</div>')

        if report and not is_error:
            parts.append(f'<div class="nb-metrics">{report}</div>')

        parts.append("</div>")

    # --- Full Report ---
    full_report = results.get("report", {}).get("report", "")
    if full_report:
        import re
        # Convert markdown to basic HTML for the report
        rpt_html = _md_to_html(full_report)
        parts.append(f'<hr class="nb-sep">')
        parts.append(f'<div class="nb-section-header">Comprehensive Analysis Report</div>')
        parts.append(f'<div class="nb-report">{rpt_html}</div>')

    parts.append("</div>")
    return "\n".join(parts)


def _md_to_html(md_text: str) -> str:
    """Minimal markdown-to-HTML converter for the report section."""
    import re
    lines = md_text.split("\n")
    html_lines = []
    in_table = False
    in_list = False
    in_blockquote = False

    for line in lines:
        stripped = line.strip()

        # Close blockquote
        if in_blockquote and not stripped.startswith(">"):
            html_lines.append("</blockquote>")
            in_blockquote = False

        # Close list
        if in_list and not stripped.startswith("- ") and not stripped.startswith("* "):
            html_lines.append("</ul>")
            in_list = False

        # Close table
        if in_table and not stripped.startswith("|"):
            html_lines.append("</tbody></table>")
            in_table = False

        if not stripped:
            html_lines.append("<br>")
            continue

        # Headers
        if stripped.startswith("# "):
            html_lines.append(f"<h1>{stripped[2:]}</h1>")
        elif stripped.startswith("## "):
            html_lines.append(f"<h2>{stripped[3:]}</h2>")
        elif stripped.startswith("### "):
            html_lines.append(f"<h3>{stripped[4:]}</h3>")
        # Blockquote
        elif stripped.startswith("> "):
            if not in_blockquote:
                html_lines.append("<blockquote>")
                in_blockquote = True
            html_lines.append(f"<p>{_inline_md(stripped[2:])}</p>")
        # Table
        elif stripped.startswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if all(set(c) <= set("-: ") for c in cells):
                continue  # separator row
            if not in_table:
                in_table = True
                html_lines.append("<table><thead><tr>")
                for c in cells:
                    html_lines.append(f"<th>{_inline_md(c)}</th>")
                html_lines.append("</tr></thead><tbody>")
            else:
                html_lines.append("<tr>")
                for c in cells:
                    html_lines.append(f"<td>{_inline_md(c)}</td>")
                html_lines.append("</tr>")
        # List
        elif stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                in_list = True
                html_lines.append("<ul>")
            html_lines.append(f"<li>{_inline_md(stripped[2:])}</li>")
        else:
            html_lines.append(f"<p>{_inline_md(stripped)}</p>")

    # Close open tags
    if in_blockquote:
        html_lines.append("</blockquote>")
    if in_list:
        html_lines.append("</ul>")
    if in_table:
        html_lines.append("</tbody></table>")

    return "\n".join(html_lines)


def _inline_md(text: str) -> str:
    """Convert inline markdown (bold, italic, code) to HTML."""
    import re
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    return text
