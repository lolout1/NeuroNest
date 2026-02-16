"""
Explainable AI (XAI) analyzer for NeuroNest.

Provides 7+ visualization methods for the EoMT-DINOv3 semantic segmentation model:
1. Self-Attention Maps — raw attention from specific layers/heads
2. Attention Rollout — aggregated attention across all layers
3. GradCAM — class-specific activation maps
4. Predictive Entropy — per-pixel uncertainty
5. Feature PCA — hidden state structure visualization
6. Class Saliency — gradient-based input importance
7. Chefer Relevancy — attention × gradient propagation

All methods are CPU-compatible and memory-managed for HF Spaces free tier.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import gc
import logging
import time
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

from ade20k_classes import ADE20K_NAMES

logger = logging.getLogger(__name__)


class AttentionCaptureHook:
    """Forward hook to capture attention weights from EomtAttention modules.

    The EoMT model computes attention weights internally but does not propagate
    them to the final output. This hook intercepts them at the attention layer.
    """

    def __init__(self):
        self.attention_weights = None
        self._handle = None

    def hook_fn(self, module, args, output):
        # EomtAttention.forward returns (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) >= 2:
            self.attention_weights = output[1].detach().cpu()

    def register(self, module):
        self._handle = module.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.attention_weights = None


class HiddenStateCaptureHook:
    """Forward hook to capture hidden states from transformer layers."""

    def __init__(self):
        self.hidden_state = None
        self._handle = None

    def hook_fn(self, module, args, output):
        if isinstance(output, tuple):
            self.hidden_state = output[0].detach().cpu()
        else:
            self.hidden_state = output.detach().cpu()

    def register(self, module):
        self._handle = module.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.hidden_state = None


class XAIAnalyzer:
    """Explainable AI analysis for EoMT-DINOv3 segmentation model.

    All methods operate on CPU and manage memory carefully for HF Spaces.
    """

    def __init__(self, eomt_model, eomt_processor, blackspot_predictor=None):
        self.model = eomt_model
        self.processor = eomt_processor
        self.blackspot_predictor = blackspot_predictor
        self._model_fp32 = None  # lazy-loaded for gradient methods

        # Probe model architecture
        self._num_layers = len(self.model.layers)
        try:
            self._num_prefix_tokens = self.model.embeddings.num_prefix_tokens
        except AttributeError:
            self._num_prefix_tokens = 5  # 1 CLS + 4 register tokens (default)

        try:
            self._num_heads = self.model.config.num_attention_heads
        except AttributeError:
            self._num_heads = 16

        logger.info(
            f"[XAI] Initialized: {self._num_layers} layers, "
            f"{self._num_heads} heads, {self._num_prefix_tokens} prefix tokens"
        )

    def _preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, int, int]:
        """Preprocess image for model input. Returns (inputs_tensor, orig_h, orig_w)."""
        orig_h, orig_w = image.shape[:2]
        pil_image = Image.fromarray(image)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        return inputs, orig_h, orig_w

    def _get_patch_grid_size(self, inputs: dict) -> Tuple[int, int]:
        """Compute patch grid dimensions from input tensor."""
        pixel_values = inputs["pixel_values"]
        _, _, h, w = pixel_values.shape
        patch_size = 14  # DINOv2 ViT-Large patch size
        return h // patch_size, w // patch_size

    @staticmethod
    def _normalize_heatmap(
        heatmap: np.ndarray,
        colormap: int,
        image: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Normalize heatmap to [0,255], apply colormap, blend with image."""
        h, w = image.shape[:2]
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        hmap_min, hmap_max = heatmap.min(), heatmap.max()
        if hmap_max - hmap_min > 1e-8:
            heatmap = (heatmap - hmap_min) / (hmap_max - hmap_min)
        else:
            heatmap = np.zeros_like(heatmap)
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(image, 1 - alpha, colored_rgb, alpha, 0)
        return blended

    @staticmethod
    def _add_title(image: np.ndarray, title: str) -> np.ndarray:
        """Add a title bar above the image."""
        h, w = image.shape[:2]
        banner_h = 40
        canvas = np.full((h + banner_h, w, 3), 30, dtype=np.uint8)
        canvas[banner_h:, :] = image
        cv2.putText(
            canvas, title, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
        return canvas

    # ------------------------------------------------------------------
    # Method 1: Self-Attention Maps
    # ------------------------------------------------------------------
    def self_attention_maps(
        self, image: np.ndarray, layer: int = -1, head: Optional[int] = None
    ) -> Dict:
        """Extract raw attention maps from a specific layer/head.

        Args:
            image: RGB numpy array
            layer: transformer layer index (0-23, or -1 for last)
            head: attention head index (0-15), or None for mean across heads
        """
        t0 = time.perf_counter()
        layer_idx = layer if layer >= 0 else self._num_layers + layer

        inputs, orig_h, orig_w = self._preprocess_image(image)
        grid_h, grid_w = self._get_patch_grid_size(inputs)

        # Hook into the specific layer's attention module
        hook = AttentionCaptureHook()
        hook.register(self.model.layers[layer_idx].attention)

        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            hook.remove()

        if hook.attention_weights is None:
            logger.warning("[XAI] No attention weights captured")
            return {"visualization": image, "report": "Failed to capture attention."}

        # attn_weights shape: (1, num_heads, seq_len, seq_len)
        attn = hook.attention_weights[0]  # (num_heads, seq_len, seq_len)

        # Select head or average
        if head is not None and 0 <= head < attn.shape[0]:
            attn_map = attn[head]  # (seq_len, seq_len)
            head_label = f"Head {head}"
        else:
            attn_map = attn.mean(dim=0)  # (seq_len, seq_len)
            head_label = "Mean"

        # Extract CLS-to-patch attention (row 0, skip prefix tokens)
        n_prefix = self._num_prefix_tokens
        n_patches = grid_h * grid_w
        cls_to_patches = attn_map[0, n_prefix:n_prefix + n_patches].numpy()

        # Reshape to spatial grid
        spatial = cls_to_patches.reshape(grid_h, grid_w)
        vis = self._normalize_heatmap(spatial, cv2.COLORMAP_INFERNO, image, alpha=0.5)
        vis = self._add_title(vis, f"Self-Attention | Layer {layer_idx} | {head_label}")

        del hook.attention_weights, attn, attn_map
        gc.collect()

        elapsed = time.perf_counter() - t0
        return {
            "visualization": vis,
            "report": (
                f"Self-attention from layer {layer_idx}, {head_label}. "
                f"Patch grid: {grid_h}x{grid_w}. Time: {elapsed:.1f}s"
            ),
        }

    # ------------------------------------------------------------------
    # Method 2: Attention Rollout
    # ------------------------------------------------------------------
    def attention_rollout(
        self, image: np.ndarray, head_fusion: str = "mean", discard_ratio: float = 0.1
    ) -> Dict:
        """Aggregate attention across all layers using rollout.

        Processes layers one at a time via hooks to minimize memory.
        """
        t0 = time.perf_counter()
        inputs, orig_h, orig_w = self._preprocess_image(image)
        grid_h, grid_w = self._get_patch_grid_size(inputs)

        # Capture attention from ALL layers one forward pass
        hooks = []
        for i in range(self._num_layers):
            h = AttentionCaptureHook()
            h.register(self.model.layers[i].attention)
            hooks.append(h)

        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        # Process rollout incrementally
        rollout = None
        for i, h in enumerate(hooks):
            if h.attention_weights is None:
                continue
            attn = h.attention_weights[0]  # (num_heads, seq_len, seq_len)

            # Fuse heads
            if head_fusion == "max":
                fused = attn.max(dim=0).values
            elif head_fusion == "min":
                fused = attn.min(dim=0).values
            else:
                fused = attn.mean(dim=0)

            # Add identity (residual) and normalize rows
            seq_len = fused.shape[0]
            identity = torch.eye(seq_len)
            fused = fused + identity
            fused = fused / fused.sum(dim=-1, keepdim=True)

            if rollout is None:
                rollout = fused
            else:
                rollout = rollout @ fused

            # Free memory for this layer
            h.attention_weights = None
            del attn, fused

        del hooks
        gc.collect()

        if rollout is None:
            return {"visualization": image, "report": "Attention rollout failed."}

        # Extract CLS-to-patch attention
        n_prefix = self._num_prefix_tokens
        n_patches = grid_h * grid_w
        cls_rollout = rollout[0, n_prefix:n_prefix + n_patches].numpy()

        # Discard lowest values
        if discard_ratio > 0:
            threshold = np.percentile(cls_rollout, discard_ratio * 100)
            cls_rollout = np.where(cls_rollout > threshold, cls_rollout, 0)

        spatial = cls_rollout.reshape(grid_h, grid_w)
        vis = self._normalize_heatmap(spatial, cv2.COLORMAP_INFERNO, image, alpha=0.5)
        vis = self._add_title(vis, f"Attention Rollout | {self._num_layers} Layers")

        del rollout
        gc.collect()

        elapsed = time.perf_counter() - t0
        return {
            "visualization": vis,
            "report": (
                f"Attention rollout across {self._num_layers} layers "
                f"({head_fusion} fusion, discard {discard_ratio:.0%}). "
                f"Time: {elapsed:.1f}s"
            ),
        }

    # ------------------------------------------------------------------
    # Method 3: Predictive Entropy
    # ------------------------------------------------------------------
    def predictive_entropy(self, image: np.ndarray) -> Dict:
        """Per-pixel prediction uncertainty from softmax distributions."""
        t0 = time.perf_counter()
        inputs, orig_h, orig_w = self._preprocess_image(image)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get per-pixel class probabilities
        # masks_queries_logits: (batch, num_queries, H, W)
        # class_queries_logits: (batch, num_queries, num_classes+1)
        mask_logits = outputs.masks_queries_logits[0]   # (Q, H, W)
        class_logits = outputs.class_queries_logits[0]  # (Q, C)

        # Compute per-pixel class probabilities via mask-weighted averaging
        mask_probs = torch.sigmoid(mask_logits)  # (Q, H, W)
        class_probs = F.softmax(class_logits[:, :-1], dim=-1)  # (Q, C-1) exclude no-object

        # Per-pixel class distribution: sum_q mask_prob_q * class_prob_q
        # Shape: (C, H, W)
        num_classes = class_probs.shape[-1]
        h_feat, w_feat = mask_probs.shape[1:]
        pixel_probs = torch.einsum("qhw,qc->chw", mask_probs, class_probs)

        # Normalize to proper probability distribution per pixel
        pixel_probs = pixel_probs / (pixel_probs.sum(dim=0, keepdim=True) + 1e-10)

        # Entropy: H = -sum(p * log(p))
        entropy = -(pixel_probs * torch.log(pixel_probs + 1e-10)).sum(dim=0)

        # Normalize by max possible entropy: log(num_classes)
        max_entropy = np.log(num_classes)
        entropy_norm = (entropy / max_entropy).numpy()

        # Resize to original
        entropy_map = cv2.resize(entropy_norm, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        vis = self._normalize_heatmap(entropy_map, cv2.COLORMAP_MAGMA, image, alpha=0.55)
        vis = self._add_title(vis, "Predictive Entropy (Uncertainty)")

        # Stats
        mean_ent = float(entropy_map.mean())
        max_ent = float(entropy_map.max())
        high_unc_pct = float((entropy_map > 0.5).mean() * 100)

        del outputs, mask_logits, class_logits, mask_probs, class_probs, pixel_probs, entropy
        gc.collect()

        elapsed = time.perf_counter() - t0
        return {
            "visualization": vis,
            "entropy_map": entropy_map,
            "report": (
                f"Predictive entropy (uncertainty). "
                f"Mean: {mean_ent:.3f}, Max: {max_ent:.3f}. "
                f"High uncertainty (>0.5): {high_unc_pct:.1f}% of pixels. "
                f"Time: {elapsed:.1f}s"
            ),
        }

    # ------------------------------------------------------------------
    # Method 4: Feature PCA
    # ------------------------------------------------------------------
    def feature_pca(self, image: np.ndarray, layer: int = -1) -> Dict:
        """PCA of hidden states — false-color visualization of feature structure."""
        t0 = time.perf_counter()
        layer_idx = layer if layer >= 0 else self._num_layers + layer

        inputs, orig_h, orig_w = self._preprocess_image(image)
        grid_h, grid_w = self._get_patch_grid_size(inputs)

        # Hook hidden state from specific layer
        hook = HiddenStateCaptureHook()
        hook.register(self.model.layers[layer_idx])

        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            hook.remove()

        if hook.hidden_state is None:
            return {"visualization": image, "report": "Failed to capture hidden states."}

        # hidden_state: (1, seq_len, hidden_dim)
        hidden = hook.hidden_state[0].numpy()  # (seq_len, hidden_dim)

        # Extract patch tokens only
        n_prefix = self._num_prefix_tokens
        n_patches = grid_h * grid_w
        patch_features = hidden[n_prefix:n_prefix + n_patches]  # (n_patches, hidden_dim)

        # PCA to 3 components
        centered = patch_features - patch_features.mean(axis=0, keepdims=True)
        # Use SVD for PCA (more numerically stable than covariance)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        components = U[:, :3] * S[:3]  # (n_patches, 3)

        # Normalize each component to [0, 255]
        for c in range(3):
            col = components[:, c]
            cmin, cmax = col.min(), col.max()
            if cmax - cmin > 1e-8:
                components[:, c] = (col - cmin) / (cmax - cmin) * 255
            else:
                components[:, c] = 128

        # Reshape to spatial RGB
        pca_image = components.reshape(grid_h, grid_w, 3).astype(np.uint8)
        pca_upscaled = cv2.resize(pca_image, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        vis = self._add_title(pca_upscaled, f"Feature PCA | Layer {layer_idx}")

        # Variance explained
        total_var = (S ** 2).sum()
        var_explained = (S[:3] ** 2) / total_var * 100

        del hook.hidden_state, hidden, centered, U, S, Vt, components
        gc.collect()

        elapsed = time.perf_counter() - t0
        return {
            "visualization": vis,
            "report": (
                f"Feature PCA (layer {layer_idx}). "
                f"Variance explained: PC1={var_explained[0]:.1f}%, "
                f"PC2={var_explained[1]:.1f}%, PC3={var_explained[2]:.1f}%. "
                f"Time: {elapsed:.1f}s"
            ),
        }

    # ------------------------------------------------------------------
    # Method 5: GradCAM for Segmentation
    # ------------------------------------------------------------------
    def _get_fp32_model(self):
        """Lazy-load FP32 model for gradient-based methods."""
        if self._model_fp32 is not None:
            return self._model_fp32

        logger.info("[XAI] Loading FP32 model for gradient-based XAI methods...")
        from transformers import AutoModelForUniversalSegmentation
        try:
            model_id = self.model.config._name_or_path
        except AttributeError:
            model_id = "tue-mps/ade20k_semantic_eomt_large_512"

        self._model_fp32 = AutoModelForUniversalSegmentation.from_pretrained(model_id)
        self._model_fp32.eval()
        logger.info("[XAI] FP32 model loaded for gradient methods")
        return self._model_fp32

    def gradcam_segmentation(
        self, image: np.ndarray, target_class_id: Optional[int] = None
    ) -> Dict:
        """GradCAM activation map for a target semantic class."""
        t0 = time.perf_counter()
        fp32_model = self._get_fp32_model()

        inputs, orig_h, orig_w = self._preprocess_image(image)
        grid_h, grid_w = self._get_patch_grid_size(inputs)

        # Hook into the last layer's attention for activations and gradients
        layer_idx = self._num_layers - 1
        target_module = fp32_model.layers[layer_idx].mlp

        activations = {}
        gradients = {}

        def forward_hook(module, inp, out):
            if isinstance(out, tuple):
                activations["value"] = out[0]
            else:
                activations["value"] = out

        def backward_hook(module, grad_input, grad_output):
            gradients["value"] = grad_output[0]

        fwd_handle = target_module.register_forward_hook(forward_hook)
        bwd_handle = target_module.register_full_backward_hook(backward_hook)

        try:
            # Forward pass (with gradients)
            outputs = fp32_model(**inputs)

            # Get segmentation to find target class
            seg_maps = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(orig_h, orig_w)]
            )
            seg_mask = seg_maps[0].cpu().numpy().astype(np.uint8)

            if target_class_id is None:
                # Use the most common non-background class
                classes, counts = np.unique(seg_mask, return_counts=True)
                target_class_id = int(classes[np.argmax(counts)])

            class_name = ADE20K_NAMES[target_class_id] if target_class_id < len(ADE20K_NAMES) else f"class_{target_class_id}"

            # Create target: sum of mask logits where seg prediction == target_class
            mask_logits = outputs.masks_queries_logits[0]  # (Q, H, W)
            class_logits = outputs.class_queries_logits[0]  # (Q, C)
            class_preds = class_logits[:, :-1].argmax(dim=-1)  # (Q,)

            # Find queries that predict target class
            target_queries = (class_preds == target_class_id).nonzero(as_tuple=True)[0]

            if len(target_queries) == 0:
                # Fallback: use query with highest probability for target class
                target_queries = class_logits[:, target_class_id].argmax(dim=0, keepdim=True)

            # Score to backprop from
            score = mask_logits[target_queries].sum()
            score.backward()

            # Compute GradCAM from captured activations/gradients
            if "value" in activations and "value" in gradients:
                act = activations["value"]  # (1, seq_len, hidden_dim)
                grad = gradients["value"]   # (1, seq_len, hidden_dim)

                # Global average pooling of gradients → weights
                weights = grad.mean(dim=-1, keepdim=True)  # (1, seq_len, 1)
                cam = (act * weights).sum(dim=-1)  # (1, seq_len)
                cam = F.relu(cam)  # Only positive contributions

                # Extract patch tokens
                n_prefix = self._num_prefix_tokens
                n_patches = grid_h * grid_w
                cam_patches = cam[0, n_prefix:n_prefix + n_patches].detach().cpu().numpy()
                spatial = cam_patches.reshape(grid_h, grid_w)

                vis = self._normalize_heatmap(spatial, cv2.COLORMAP_JET, image, alpha=0.5)
            else:
                vis = image.copy()
                class_name = "N/A"

        finally:
            fwd_handle.remove()
            bwd_handle.remove()
            fp32_model.zero_grad()

        vis = self._add_title(vis, f"GradCAM | {class_name.split(',')[0]}")

        del activations, gradients
        gc.collect()

        elapsed = time.perf_counter() - t0
        return {
            "visualization": vis,
            "target_class": target_class_id,
            "report": (
                f"GradCAM for class '{class_name}' (ID {target_class_id}). "
                f"Time: {elapsed:.1f}s"
            ),
        }

    # ------------------------------------------------------------------
    # Method 6: Class Saliency
    # ------------------------------------------------------------------
    def class_saliency(
        self, image: np.ndarray, target_class_id: Optional[int] = None
    ) -> Dict:
        """Gradient-based saliency: which input pixels matter for a class."""
        t0 = time.perf_counter()
        fp32_model = self._get_fp32_model()

        inputs, orig_h, orig_w = self._preprocess_image(image)
        pixel_values = inputs["pixel_values"].requires_grad_(True)

        outputs = fp32_model(pixel_values=pixel_values)

        # Determine target class from segmentation
        seg_maps = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(orig_h, orig_w)]
        )
        seg_mask = seg_maps[0].cpu().numpy().astype(np.uint8)

        if target_class_id is None:
            classes, counts = np.unique(seg_mask, return_counts=True)
            target_class_id = int(classes[np.argmax(counts)])

        class_name = ADE20K_NAMES[target_class_id] if target_class_id < len(ADE20K_NAMES) else f"class_{target_class_id}"

        # Backprop from target class scores
        class_logits = outputs.class_queries_logits[0]  # (Q, C)
        target_score = class_logits[:, target_class_id].sum()
        target_score.backward()

        # Saliency = max absolute gradient across color channels
        saliency = pixel_values.grad[0].abs().max(dim=0).values.detach().cpu().numpy()

        # Resize to original image dimensions
        saliency_map = cv2.resize(saliency, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        vis = self._normalize_heatmap(saliency_map, cv2.COLORMAP_HOT, image, alpha=0.5)
        vis = self._add_title(vis, f"Saliency | {class_name.split(',')[0]}")

        fp32_model.zero_grad()
        del saliency
        gc.collect()

        elapsed = time.perf_counter() - t0
        return {
            "visualization": vis,
            "report": (
                f"Class saliency for '{class_name}' (ID {target_class_id}). "
                f"Time: {elapsed:.1f}s"
            ),
        }

    # ------------------------------------------------------------------
    # Method 7: Chefer Relevancy
    # ------------------------------------------------------------------
    def chefer_relevancy(
        self, image: np.ndarray, target_class_id: Optional[int] = None
    ) -> Dict:
        """Chefer et al. attention × gradient relevancy propagation for ViT."""
        t0 = time.perf_counter()
        fp32_model = self._get_fp32_model()

        inputs, orig_h, orig_w = self._preprocess_image(image)
        grid_h, grid_w = self._get_patch_grid_size(inputs)

        # Register hooks on ALL attention layers to capture attention + gradients
        attn_hooks = []
        attn_grads = []

        for i in range(self._num_layers):
            attn_module = fp32_model.layers[i].attention
            hook_data = {"attn": None, "grad": None}

            def make_fwd_hook(data):
                def hook_fn(module, args, output):
                    if isinstance(output, tuple) and len(output) >= 2:
                        data["attn"] = output[1]
                return hook_fn

            def make_bwd_hook(data):
                def hook_fn(module, grad_input, grad_output):
                    # Gradient w.r.t. attention weights
                    if len(grad_output) >= 2 and grad_output[1] is not None:
                        data["grad"] = grad_output[1]
                return hook_fn

            fwd_h = attn_module.register_forward_hook(make_fwd_hook(hook_data))
            bwd_h = attn_module.register_full_backward_hook(make_bwd_hook(hook_data))
            attn_hooks.append((fwd_h, bwd_h, hook_data))

        try:
            outputs = fp32_model(**inputs)

            # Determine target and backprop
            seg_maps = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(orig_h, orig_w)]
            )
            seg_mask = seg_maps[0].cpu().numpy().astype(np.uint8)

            if target_class_id is None:
                classes, counts = np.unique(seg_mask, return_counts=True)
                target_class_id = int(classes[np.argmax(counts)])

            class_name = ADE20K_NAMES[target_class_id] if target_class_id < len(ADE20K_NAMES) else f"class_{target_class_id}"

            class_logits = outputs.class_queries_logits[0]
            target_score = class_logits[:, target_class_id].sum()
            target_score.backward()

            # Chefer relevancy propagation
            n_prefix = self._num_prefix_tokens
            n_patches = grid_h * grid_w
            seq_len = n_prefix + n_patches

            # Initialize relevancy as identity
            R = torch.eye(seq_len)

            for _, _, hook_data in attn_hooks:
                attn = hook_data.get("attn")
                grad = hook_data.get("grad")

                if attn is None:
                    continue

                # Mean across heads
                attn_mean = attn[0].detach().cpu().mean(dim=0)  # (seq, seq)

                if grad is not None:
                    grad_mean = grad[0].detach().cpu().mean(dim=0)  # (seq, seq)
                    # Attention × gradient, clamp negatives
                    relevancy = torch.clamp(attn_mean * grad_mean, min=0)
                else:
                    relevancy = attn_mean

                # Truncate to common seq_len if needed
                min_s = min(relevancy.shape[0], seq_len)
                rel_truncated = relevancy[:min_s, :min_s]
                R_crop = R[:min_s, :min_s]

                # Add identity and normalize
                identity = torch.eye(min_s)
                rel_truncated = rel_truncated + identity
                rel_truncated = rel_truncated / (rel_truncated.sum(dim=-1, keepdim=True) + 1e-10)

                R[:min_s, :min_s] = R_crop + R_crop @ rel_truncated

            # Extract CLS-to-patch relevancy
            cls_relevancy = R[0, n_prefix:n_prefix + n_patches].numpy()
            spatial = cls_relevancy.reshape(grid_h, grid_w)

            vis = self._normalize_heatmap(spatial, cv2.COLORMAP_INFERNO, image, alpha=0.5)

            # Add threshold overlay
            threshold = spatial.mean()
            binary_mask = cv2.resize(
                (spatial > threshold).astype(np.float32),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )
            # Subtle green tint on relevant regions
            overlay = vis.copy()
            overlay[binary_mask > 0.5] = (
                overlay[binary_mask > 0.5] * 0.7 + np.array([0, 80, 0]) * 0.3
            ).astype(np.uint8)
            vis = overlay

        finally:
            for fwd_h, bwd_h, _ in attn_hooks:
                fwd_h.remove()
                bwd_h.remove()
            fp32_model.zero_grad()

        vis = self._add_title(vis, f"Chefer Relevancy | {class_name.split(',')[0]}")

        del attn_hooks, R
        gc.collect()

        elapsed = time.perf_counter() - t0
        return {
            "visualization": vis,
            "report": (
                f"Chefer relevancy for '{class_name}' (ID {target_class_id}). "
                f"Propagated through {self._num_layers} layers. "
                f"Time: {elapsed:.1f}s"
            ),
        }

    # ------------------------------------------------------------------
    # Method 8: Natural Language Report
    # ------------------------------------------------------------------
    def generate_xai_report(
        self, image: np.ndarray, seg_mask: Optional[np.ndarray] = None
    ) -> Dict:
        """Generate a structured natural-language XAI report."""
        t0 = time.perf_counter()

        # Run lightweight methods for report data
        entropy_result = self.predictive_entropy(image)
        rollout_result = self.attention_rollout(image)

        # Segmentation analysis
        if seg_mask is None:
            inputs, orig_h, orig_w = self._preprocess_image(image)
            with torch.no_grad():
                outputs = self.model(**inputs)
            seg_maps = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(orig_h, orig_w)]
            )
            seg_mask = seg_maps[0].cpu().numpy().astype(np.uint8)
            del outputs
            gc.collect()

        # Class distribution
        classes, counts = np.unique(seg_mask, return_counts=True)
        total_pixels = seg_mask.size
        class_info = []
        for cls_id, count in sorted(zip(classes, counts), key=lambda x: -x[1]):
            pct = count / total_pixels * 100
            name = ADE20K_NAMES[cls_id] if cls_id < len(ADE20K_NAMES) else f"class_{cls_id}"
            class_info.append((cls_id, name.split(",")[0], pct))

        # Entropy analysis
        entropy_map = entropy_result.get("entropy_map")
        mean_entropy = float(entropy_map.mean()) if entropy_map is not None else 0
        high_unc_pct = float((entropy_map > 0.5).mean() * 100) if entropy_map is not None else 0

        # Find uncertain boundaries
        uncertain_boundaries = []
        if entropy_map is not None:
            high_unc_mask = entropy_map > 0.5
            # Find which class pairs are at uncertain boundaries
            from skimage.segmentation import find_boundaries
            for cls_id, name, pct in class_info[:5]:
                cls_mask = seg_mask == cls_id
                boundary = find_boundaries(cls_mask, mode="thick")
                boundary_entropy = entropy_map[boundary].mean() if boundary.any() else 0
                if boundary_entropy > 0.3:
                    uncertain_boundaries.append((name, boundary_entropy))

        # Build report
        report_lines = [
            "# Explainable AI Analysis Report\n",
            f"*Generated in {time.perf_counter() - t0:.1f}s*\n",
            "## Scene Composition",
            f"The model identified **{len(classes)} object categories**:\n",
        ]
        for cls_id, name, pct in class_info[:8]:
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            report_lines.append(f"- **{name}**: {pct:.1f}% `{bar}`")

        report_lines.extend([
            "\n## Model Confidence",
            f"- **Mean uncertainty**: {mean_entropy:.3f} (0=certain, 1=uncertain)",
            f"- **High uncertainty regions**: {high_unc_pct:.1f}% of image",
        ])

        if mean_entropy < 0.15:
            report_lines.append("- The model is **highly confident** in its predictions")
        elif mean_entropy < 0.35:
            report_lines.append("- The model shows **moderate confidence** overall")
        else:
            report_lines.append("- The model shows **significant uncertainty** — predictions may be unreliable")

        if uncertain_boundaries:
            report_lines.append("\n### Uncertain Boundaries:")
            for name, ent in sorted(uncertain_boundaries, key=lambda x: -x[1])[:5]:
                report_lines.append(f"- {name} boundary: entropy={ent:.3f}")

        report_lines.extend([
            "\n## Attention Analysis",
            "Attention rollout reveals where the model focuses across all transformer layers.",
            f"The model uses a {self._num_layers}-layer, {self._num_heads}-head "
            f"Vision Transformer (DINOv3-EoMT-Large).",
        ])

        report_lines.extend([
            "\n## Methodology",
            "- **Entropy**: Shannon entropy of per-pixel class probability distributions",
            "- **Attention Rollout**: Abnar & Zuidema (2020) — multiplicative propagation of attention",
            "- **GradCAM**: Selvaraju et al. (2017) — gradient-weighted activation mapping",
            "- **Chefer Relevancy**: Chefer et al. (2021) — attention × gradient for ViT",
            "- **Feature PCA**: Principal component analysis of intermediate representations",
            "- **Saliency**: Simonyan et al. (2014) — input gradient magnitude",
        ])

        elapsed = time.perf_counter() - t0
        return {
            "visualization": entropy_result["visualization"],
            "report": "\n".join(report_lines),
        }

    # ------------------------------------------------------------------
    # Orchestrator: Full Suite
    # ------------------------------------------------------------------
    def run_full_analysis(
        self,
        image: np.ndarray,
        layer: int = -1,
        head: Optional[int] = None,
        target_class_id: Optional[int] = None,
    ) -> Dict:
        """Run all XAI methods. Returns dict with all visualizations + report."""
        logger.info("[XAI] Running full analysis suite...")
        t0 = time.perf_counter()

        results = {}

        # Phase 1: No-gradient methods (can run on quantized model)
        # Run sequentially to manage memory on CPU
        logger.info("[XAI] Phase 1: Attention & entropy methods...")
        try:
            results["attention"] = self.self_attention_maps(image, layer=layer, head=head)
        except Exception as e:
            logger.error(f"[XAI] Self-attention failed: {e}")
            results["attention"] = {"visualization": image, "report": f"Error: {e}"}

        try:
            results["rollout"] = self.attention_rollout(image)
        except Exception as e:
            logger.error(f"[XAI] Attention rollout failed: {e}")
            results["rollout"] = {"visualization": image, "report": f"Error: {e}"}

        try:
            results["entropy"] = self.predictive_entropy(image)
        except Exception as e:
            logger.error(f"[XAI] Entropy failed: {e}")
            results["entropy"] = {"visualization": image, "report": f"Error: {e}"}

        try:
            results["pca"] = self.feature_pca(image, layer=layer)
        except Exception as e:
            logger.error(f"[XAI] Feature PCA failed: {e}")
            results["pca"] = {"visualization": image, "report": f"Error: {e}"}

        gc.collect()

        # Phase 2: Gradient methods (need FP32 model, sequential)
        logger.info("[XAI] Phase 2: Gradient-based methods (loading FP32 model)...")
        try:
            results["gradcam"] = self.gradcam_segmentation(
                image, target_class_id=target_class_id
            )
        except Exception as e:
            logger.error(f"[XAI] GradCAM failed: {e}")
            results["gradcam"] = {"visualization": image, "report": f"Error: {e}"}

        try:
            results["saliency"] = self.class_saliency(
                image, target_class_id=target_class_id
            )
        except Exception as e:
            logger.error(f"[XAI] Saliency failed: {e}")
            results["saliency"] = {"visualization": image, "report": f"Error: {e}"}

        try:
            results["chefer"] = self.chefer_relevancy(
                image, target_class_id=target_class_id
            )
        except Exception as e:
            logger.error(f"[XAI] Chefer relevancy failed: {e}")
            results["chefer"] = {"visualization": image, "report": f"Error: {e}"}

        gc.collect()

        # Phase 3: Generate report
        logger.info("[XAI] Phase 3: Generating report...")
        try:
            results["report"] = self.generate_xai_report(image)
        except Exception as e:
            logger.error(f"[XAI] Report generation failed: {e}")
            results["report"] = {"visualization": image, "report": f"Error: {e}"}

        elapsed = time.perf_counter() - t0
        logger.info(f"[XAI] Full analysis completed in {elapsed:.1f}s")

        return results

    def cleanup_fp32(self):
        """Release FP32 model to free memory."""
        if self._model_fp32 is not None:
            del self._model_fp32
            self._model_fp32 = None
            gc.collect()
            logger.info("[XAI] FP32 model released")
