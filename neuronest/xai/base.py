import numpy as np
import cv2
import torch
import gc
import logging
from typing import Tuple
from PIL import Image

from ade20k_classes import ADE20K_NAMES
from .viz import METHOD_INFO, draw_colorbar, draw_info_panel

logger = logging.getLogger(__name__)

_FLOOR_IDS = {3, 4, 13, 28, 78}


class XAIBase:
    """Core XAI infrastructure: model refs, preprocessing, visualization helpers."""

    def __init__(self, eomt_model, eomt_processor, blackspot_predictor=None):
        self.model = eomt_model
        self.processor = eomt_processor
        self._model_fp32 = None

        cfg = self.model.config
        self._num_layers = len(self.model.layers)
        self._num_heads = getattr(cfg, "num_attention_heads", 16)
        self._patch_size = getattr(cfg, "patch_size", 16)
        self._num_queries = getattr(cfg, "num_queries", 100)
        self._num_prefix = getattr(
            getattr(self.model, "embeddings", None), "num_prefix_tokens", 5
        )

        n_dec = getattr(cfg, "num_decoder_layers", getattr(cfg, "decoder_layers", 4))
        self._n_encoder = self._num_layers - n_dec

        logger.info(
            f"[XAI] {self._num_layers} layers ({self._n_encoder}enc+{n_dec}dec), "
            f"{self._num_heads} heads, patch={self._patch_size}, "
            f"prefix={self._num_prefix}, queries={self._num_queries}"
        )

    def _preprocess(self, image: np.ndarray):
        h, w = image.shape[:2]
        inputs = self.processor(images=Image.fromarray(image), return_tensors="pt")
        return inputs, h, w

    def _patch_grid(self, inputs: dict) -> Tuple[int, int]:
        _, _, h, w = inputs["pixel_values"].shape
        return h // self._patch_size, w // self._patch_size

    def _infer_grid(self, seq_len: int, is_decoder: bool = False) -> Tuple[int, int]:
        n = seq_len - self._num_prefix
        if is_decoder:
            n -= self._num_queries
        n = max(n, 1)
        side = int(round(np.sqrt(n)))
        return side, side

    def _cls_to_patches(self, attn_2d, gh: int, gw: int, is_decoder: bool = False):
        seq = attn_2d.shape[-1]
        n_pre = self._num_prefix
        n_expected = gh * gw
        n_avail = seq - n_pre - (self._num_queries if is_decoder else 0)

        if n_avail != n_expected:
            gh, gw = self._infer_grid(seq, is_decoder)
            n_expected = gh * gw

        row = attn_2d[0, n_pre : n_pre + n_expected]
        if isinstance(row, torch.Tensor):
            row = row.numpy()
        return row.reshape(gh, gw), gh, gw

    def _dominant_class(self, seg_mask):
        cls, cnt = np.unique(seg_mask, return_counts=True)
        total = seg_mask.size
        for fid in _FLOOR_IDS:
            if fid in cls:
                idx = list(cls).index(fid)
                if cnt[idx] / total > 0.01:
                    return int(fid)
        return int(cls[np.argmax(cnt)])

    def _cname(self, cid):
        return (
            ADE20K_NAMES[cid].split(",")[0].strip()
            if 0 <= cid < len(ADE20K_NAMES)
            else f"class_{cid}"
        )

    def _seg_from_outputs(self, outputs, h, w):
        return (
            self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(h, w)]
            )[0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

    @staticmethod
    def _blend(heatmap, image, cmap=cv2.COLORMAP_INFERNO, alpha=0.5):
        h, w = image.shape[:2]
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        lo, hi = heatmap.min(), heatmap.max()
        norm = (heatmap - lo) / (hi - lo + 1e-8)
        cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
        return cv2.addWeighted(
            image, 1 - alpha, cv2.cvtColor(cm, cv2.COLOR_BGR2RGB), alpha, 0
        )

    def _annotate(self, image, method_key, title_text, extra=""):
        h, w = image.shape[:2]
        info = METHOD_INFO.get(method_key, {})

        bar = np.full((44, w, 3), 30, dtype=np.uint8)
        cv2.putText(
            bar, title_text, (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA,
        )
        vis = np.vstack([bar, image])

        cmap = info.get("cmap")
        if cmap is not None:
            vis = draw_colorbar(vis, cmap)

        vis = draw_info_panel(vis, method_key, extra)
        return vis

    def _error_image(self, image, method_key, error_msg):
        h, w = image.shape[:2]
        overlay = (image * 0.3).astype(np.uint8)
        info = METHOD_INFO.get(method_key, {})
        title = info.get("title", method_key)
        cv2.putText(
            overlay, f"{title}: FAILED", (20, h // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 80), 2, cv2.LINE_AA,
        )
        max_chars = max(1, (w - 40) // 10)
        lines = [
            error_msg[i : i + max_chars]
            for i in range(0, min(len(error_msg), max_chars * 3), max_chars)
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                overlay, line, (20, h // 2 + 15 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
            )
        return overlay

    def _fallback(self, image, msg, method_key=""):
        return {
            "visualization": self._error_image(image, method_key, msg),
            "report": f"Error ({method_key}): {msg}",
        }

    # FP32 model management for gradient methods

    def _get_fp32(self):
        if self._model_fp32 is not None:
            return self._model_fp32
        logger.info("[XAI] Loading FP32 model for gradient methods...")
        from transformers import AutoModelForUniversalSegmentation

        mid = getattr(
            self.model.config,
            "_name_or_path",
            "tue-mps/ade20k_semantic_eomt_large_512",
        )
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
        fp32 = self._get_fp32()
        fwd = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        if need_grad_pv:
            fwd["pixel_values"] = fwd["pixel_values"].requires_grad_(True)
        outputs = fp32(**fwd)
        return outputs, fwd.get("pixel_values")
