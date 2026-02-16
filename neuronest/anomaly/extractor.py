"""DINOv2-Large feature extractor for room images.

Extracts 1024-dim CLS token embeddings from DINOv2-Large, the same backbone
family used by EoMT-DINOv3 in the main segmentation pipeline. Features are
L2-normalized for stable autoencoder training.

Supports batched extraction, GPU/CPU, caching to disk, and robust image
loading with validation.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

DINOV2_MODEL_ID = "facebook/dinov2-large"
FEATURE_DIM = 1024
MIN_IMAGE_SIZE = 32
MAX_IMAGE_SIZE = 8192


class FeatureExtractor:
    """Extracts DINOv2-Large CLS features from room images.

    Args:
        device: torch device ("cuda", "cpu", or auto-detect).
        cache_dir: Directory for caching extracted features. None disables caching.
        dtype: Inference dtype. float16 for GPU (saves VRAM), float32 for CPU.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._processor = None
        self._model = None

    def _ensure_loaded(self):
        """Lazy-load model on first use to avoid import-time GPU allocation."""
        if self._model is not None:
            return

        from transformers import AutoImageProcessor, AutoModel

        logger.info(f"Loading DINOv2-Large on {self.device} (dtype={self.dtype})")
        self._processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
        self._model = AutoModel.from_pretrained(
            DINOV2_MODEL_ID, torch_dtype=self.dtype,
        ).to(self.device)
        self._model.eval()

        param_count = sum(p.numel() for p in self._model.parameters()) / 1e6
        logger.info(f"DINOv2-Large loaded: {param_count:.0f}M params")

    def _load_image(self, path: Union[str, Path]) -> Optional[Image.Image]:
        """Load and validate an image. Returns None on failure."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Image not found: {path}")
            return None
        if path.stat().st_size == 0:
            logger.warning(f"Empty file: {path}")
            return None

        try:
            img = Image.open(path)
            img.verify()
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            logger.warning(f"Cannot load image {path}: {e}")
            return None

        w, h = img.size
        if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
            logger.warning(f"Image too small ({w}x{h}): {path}")
            return None
        if w > MAX_IMAGE_SIZE or h > MAX_IMAGE_SIZE:
            # Resize proportionally
            scale = MAX_IMAGE_SIZE / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            logger.info(f"Resized oversized image {path} to {img.size}")

        return img

    def _image_hash(self, path: Union[str, Path]) -> str:
        """Stable hash for cache key based on file path and modification time."""
        path = Path(path)
        key = f"{path.resolve()}:{path.stat().st_mtime}:{path.stat().st_size}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _load_cached(self, path: Union[str, Path]) -> Optional[np.ndarray]:
        """Load cached feature vector if available."""
        if self.cache_dir is None:
            return None
        cache_path = self.cache_dir / f"{self._image_hash(path)}.npy"
        if cache_path.exists():
            try:
                feat = np.load(cache_path)
                if feat.shape == (FEATURE_DIM,) and np.isfinite(feat).all():
                    return feat
                logger.warning(f"Invalid cached feature, re-extracting: {cache_path}")
                cache_path.unlink()
            except Exception:
                pass
        return None

    def _save_cached(self, path: Union[str, Path], feature: np.ndarray):
        """Save feature vector to cache."""
        if self.cache_dir is None:
            return
        cache_path = self.cache_dir / f"{self._image_hash(path)}.npy"
        try:
            np.save(cache_path, feature)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    @torch.inference_mode()
    def extract_single(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Extract 1024-dim feature vector from a single image.

        Returns:
            L2-normalized float32 numpy array of shape (1024,), or None on failure.
        """
        cached = self._load_cached(image_path)
        if cached is not None:
            return cached

        img = self._load_image(image_path)
        if img is None:
            return None

        self._ensure_loaded()

        inputs = self._processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        # CLS token is first token in last_hidden_state
        cls_feat = outputs.last_hidden_state[:, 0, :].squeeze(0)

        # L2 normalize for stable training
        feat = cls_feat.float().cpu()
        feat = feat / (feat.norm() + 1e-8)
        feat_np = feat.numpy()

        if not np.isfinite(feat_np).all():
            logger.warning(f"Non-finite features from {image_path}, skipping")
            return None

        self._save_cached(image_path, feat_np)
        return feat_np

    @torch.inference_mode()
    def extract_batch(
        self,
        image_paths: Sequence[Union[str, Path]],
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> tuple:
        """Extract features from multiple images with batching.

        Args:
            image_paths: Sequence of image file paths.
            batch_size: Images per forward pass. 8 fits comfortably on RTX 2080 8GB.
            show_progress: Print progress updates.

        Returns:
            (features, valid_paths, skipped_paths) where features is (N, 1024) float32.
        """
        self._ensure_loaded()

        all_features = []
        valid_paths = []
        skipped_paths = []

        # First pass: load from cache
        uncached_indices = []
        for i, path in enumerate(image_paths):
            cached = self._load_cached(path)
            if cached is not None:
                all_features.append((i, cached))
                valid_paths.append(str(path))
            else:
                uncached_indices.append(i)

        if show_progress and all_features:
            logger.info(f"Loaded {len(all_features)} features from cache")

        # Second pass: extract uncached in batches
        batch_images = []
        batch_indices = []

        def _flush_batch():
            if not batch_images:
                return
            inputs = self._processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}

            outputs = self._model(**inputs)
            cls_feats = outputs.last_hidden_state[:, 0, :].float().cpu()

            for j, feat in enumerate(cls_feats):
                feat = feat / (feat.norm() + 1e-8)
                feat_np = feat.numpy()
                if np.isfinite(feat_np).all():
                    idx = batch_indices[j]
                    all_features.append((idx, feat_np))
                    valid_paths.append(str(image_paths[idx]))
                    self._save_cached(image_paths[idx], feat_np)
                else:
                    skipped_paths.append(str(image_paths[batch_indices[j]]))

            batch_images.clear()
            batch_indices.clear()

        total_uncached = len(uncached_indices)
        for count, i in enumerate(uncached_indices):
            img = self._load_image(image_paths[i])
            if img is None:
                skipped_paths.append(str(image_paths[i]))
                continue

            batch_images.append(img)
            batch_indices.append(i)

            if len(batch_images) >= batch_size:
                _flush_batch()
                if show_progress:
                    done = count + 1
                    logger.info(f"Extracted {done}/{total_uncached} images")

                # Clear GPU cache periodically
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        _flush_batch()

        if not all_features:
            return np.empty((0, FEATURE_DIM), dtype=np.float32), [], list(map(str, image_paths))

        # Sort by original index to maintain order
        all_features.sort(key=lambda x: x[0])
        features = np.stack([f for _, f in all_features], axis=0)

        if show_progress:
            logger.info(
                f"Extraction complete: {len(valid_paths)} succeeded, "
                f"{len(skipped_paths)} skipped"
            )

        return features, valid_paths, skipped_paths

    def unload(self):
        """Free GPU memory by unloading the model."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("DINOv2 model unloaded")
