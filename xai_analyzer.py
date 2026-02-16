"""
Explainable AI (XAI) analyzer for NeuroNest.

7 visualization methods for EoMT-DINOv3 semantic segmentation:
1. Self-Attention Maps   2. Attention Rollout   3. GradCAM
4. Predictive Entropy    5. Feature PCA         6. Class Saliency
7. Chefer Relevancy

CPU-compatible. Memory-managed for HF Spaces (16 GB).
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import gc
import logging
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
    "saliency": {
        "title": "Class Saliency",
        "desc": "Which input pixels most influence the target class",
        "detail": "Simonyan et al. (2014) — input gradient magnitude",
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

    def _dominant_class(self, seg_mask):
        cls, cnt = np.unique(seg_mask, return_counts=True)
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
    # 6. Class Saliency
    # ------------------------------------------------------------------

    def class_saliency(self, image, target_class_id=None):
        t0 = time.perf_counter()
        inputs, oh, ow = self._preprocess(image)

        # Pass ALL inputs (including pixel_mask etc.) with grad enabled on pixel_values
        outputs, pv = self._fp32_forward(inputs, need_grad_pv=True)
        seg = self._seg_from_outputs(outputs, oh, ow)
        if target_class_id is None:
            target_class_id = self._dominant_class(seg)
        cn = self._cname(target_class_id)

        score = outputs.class_queries_logits[0][:, target_class_id].sum()
        score.backward()

        fp32 = self._get_fp32()
        if pv.grad is not None:
            sal = pv.grad[0].abs().max(dim=0).values.detach().cpu().numpy()
            sal = cv2.resize(sal, (ow, oh), interpolation=cv2.INTER_CUBIC)
            sal_max = float(sal.max())
            sensitive_pct = float((sal > sal.mean() + sal.std()).mean() * 100)
            blended = self._blend(sal, image, cv2.COLORMAP_HOT)
        else:
            blended = image.copy()
            sal_max = 0
            sensitive_pct = 0
            logger.warning("[XAI] Saliency: no gradient on pixel_values")

        extra = f"Target: {cn} (ID {target_class_id}) | Peak grad={sal_max:.4f} | Sensitive={sensitive_pct:.0f}%"
        vis = self._annotate(blended, "saliency", f"Saliency | {cn}", extra)

        elapsed = time.perf_counter() - t0
        fp32.zero_grad()
        gc.collect()
        return {
            "visualization": vis,
            "report": (
                f"**Class Saliency** (target: '{cn}', ID {target_class_id}): "
                f"Peak gradient={sal_max:.4f}, sensitive area={sensitive_pct:.1f}%. "
                f"Bright pixels = high influence on class prediction. {elapsed:.1f}s"
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

    def generate_xai_report(self, image, seg_mask=None):
        t0 = time.perf_counter()
        ent_result = self.predictive_entropy(image)

        if seg_mask is None:
            inputs, oh, ow = self._preprocess(image)
            with torch.no_grad():
                outputs = self.model(**inputs)
            seg_mask = self._seg_from_outputs(outputs, oh, ow)
            del outputs
            gc.collect()

        cls, cnt = np.unique(seg_mask, return_counts=True)
        total = seg_mask.size
        info = sorted(zip(cls, cnt), key=lambda x: -x[1])

        emap = ent_result.get("entropy_map")
        me = float(emap.mean()) if emap is not None else 0
        hp = float((emap > 0.5).mean() * 100) if emap is not None else 0

        ub = []
        if emap is not None:
            try:
                from skimage.segmentation import find_boundaries
                for cid, _ in info[:5]:
                    bd = find_boundaries(seg_mask == cid, mode="thick")
                    if bd.any():
                        be = float(emap[bd].mean())
                        if be > 0.3:
                            ub.append((self._cname(cid), be))
            except ImportError:
                pass

        lines = [
            "# XAI Analysis Report\n",
            "## Scene Composition",
            f"**{len(cls)} categories** detected:\n",
        ]
        for cid, c in info[:8]:
            p = c / total * 100
            bar = "\u2588" * int(p / 5) + "\u2591" * max(0, 20 - int(p / 5))
            lines.append(f"- **{self._cname(cid)}**: {p:.1f}% `{bar}`")

        lines += [
            "\n## Model Confidence",
            f"- Mean uncertainty: **{me:.3f}**",
            f"- High-uncertainty pixels: **{hp:.1f}%**",
        ]
        if me < 0.15:
            lines.append("- Overall: **Highly confident** predictions across the scene")
        elif me < 0.35:
            lines.append("- Overall: **Moderate confidence** — some ambiguous boundaries")
        else:
            lines.append("- Overall: **Significant uncertainty** — review predictions carefully")

        if ub:
            lines.append("\n### Uncertain Boundaries")
            for n, e in sorted(ub, key=lambda x: -x[1])[:5]:
                lines.append(f"- **{n}**: boundary entropy={e:.3f}")

        lines += [
            "\n## Method Summary",
            "| Method | Type | What It Shows |",
            "|--------|------|--------------|",
            "| Self-Attention | Attention | Single-layer spatial focus pattern |",
            "| Attention Rollout | Attention | Cumulative focus across all encoder layers |",
            "| GradCAM | Gradient | Regions activating target class prediction |",
            "| Predictive Entropy | Output | Per-pixel classification uncertainty |",
            "| Feature PCA | Hidden State | Learned feature structure (false-color) |",
            "| Class Saliency | Gradient | Input pixels influencing class decision |",
            "| Chefer Relevancy | Attn x Grad | Transformer-specific attribution map |",
            f"\n## Architecture",
            f"- **Model**: DINOv3-EoMT-Large ({self._num_layers} layers, {self._num_heads} heads)",
            f"- **Patch size**: {self._patch_size}px, **Classes**: 150 (ADE20K)",
            f"- **Encoder**: {self._n_encoder} layers, **Decoder**: {self._num_layers - self._n_encoder} layers",
            "\n## References",
            "- Attention Rollout: Abnar & Zuidema (2020)",
            "- GradCAM: Selvaraju et al. (2017)",
            "- Chefer Relevancy: Chefer et al. (2021)",
            "- Saliency Maps: Simonyan et al. (2014)",
        ]
        return {"visualization": ent_result["visualization"], "report": "\n".join(lines)}

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
            ("saliency",  self.class_saliency,       {"target_class_id": target_class_id}),
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
            results["report"] = self.generate_xai_report(image)
        except Exception as e:
            logger.error(f"[XAI] report failed: {e}")
            results["report"] = {"visualization": None, "report": f"Report generation failed: {e}"}

        elapsed = time.perf_counter() - t0
        _progress(1.0, f"Complete ({elapsed:.0f}s)")
        logger.info(f"[XAI] Full analysis done in {elapsed:.1f}s")

        # Count successes
        ok = sum(1 for k in ["attention", "rollout", "entropy", "pca", "gradcam", "saliency", "chefer"]
                 if k in results and "Error" not in results[k].get("report", ""))
        logger.info(f"[XAI] {ok}/7 methods succeeded")

        return results
