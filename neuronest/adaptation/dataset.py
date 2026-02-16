"""Indoor image dataset for self-supervised domain adaptation.

Loads images from:
1. ADE20K training split via HuggingFace datasets (filtered to indoor scenes)
2. Local directories of room images (e.g., blackspot dataset)

All images are resized to 224x224 for MAE training and normalized using
ImageNet statistics (matching DINOv2 preprocessing).
"""

import glob
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ADE20K indoor scene keywords — scenes containing these are kept
INDOOR_SCENE_KEYWORDS = {
    "bedroom", "kitchen", "bathroom", "living_room", "dining_room",
    "office", "corridor", "hallway", "lobby", "basement", "attic",
    "closet", "garage", "laundry", "nursery", "playroom", "classroom",
    "library", "hospital", "waiting_room", "hotel", "staircase",
    "elevator", "arcade", "bar", "cafeteria", "restaurant",
    "studio", "warehouse", "dorm", "room", "indoor", "interior",
    "apartment", "house", "home", "building_indoor", "shop",
    "store", "market", "mall", "gym", "spa", "salon",
    "church_indoor", "theater_indoor", "museum_indoor",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _find_local_images(dirs: Sequence[Union[str, Path]]) -> List[str]:
    """Recursively find all image files in a list of directories."""
    images = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            logger.warning(f"Directory not found: {d}")
            continue
        for ext in IMAGE_EXTENSIONS:
            images.extend(str(p) for p in d.rglob(f"*{ext}"))
            images.extend(str(p) for p in d.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def _is_indoor_scene(scene_name: str) -> bool:
    """Check if an ADE20K scene name is indoor."""
    name = scene_name.lower().replace("-", "_").replace(" ", "_")
    # Direct keyword match
    for kw in INDOOR_SCENE_KEYWORDS:
        if kw in name:
            return True
    # ADE20K uses prefixes like "a/abbey" or "b/bedroom"
    parts = name.split("/")
    for part in parts:
        for kw in INDOOR_SCENE_KEYWORDS:
            if kw in part:
                return True
    return False


class IndoorImageDataset(Dataset):
    """Dataset of indoor room images for MAE self-supervised training.

    Combines ADE20K indoor subset with local image directories.

    Args:
        ade20k: Whether to load ADE20K from HuggingFace.
        local_dirs: Additional local directories to include.
        image_size: Target size for all images (square crop).
        max_ade20k: Maximum ADE20K images to use (None = all indoor).
        split: ADE20K split ("train" or "validation").
    """

    def __init__(
        self,
        ade20k: bool = True,
        local_dirs: Optional[Sequence[Union[str, Path]]] = None,
        image_size: int = 224,
        max_ade20k: Optional[int] = None,
        split: str = "train",
    ):
        self.image_size = image_size
        self._images: List = []  # PIL Images or paths
        self._sources: List[str] = []  # "ade20k" or "local"

        # Load ADE20K indoor images
        if ade20k:
            self._load_ade20k(split, max_ade20k)

        # Load local images
        if local_dirs:
            local_paths = _find_local_images(local_dirs)
            for p in local_paths:
                self._images.append(p)
                self._sources.append("local")
            logger.info(f"Added {len(local_paths)} local images")

        logger.info(
            f"IndoorImageDataset: {len(self._images)} total images "
            f"(ADE20K: {self._sources.count('ade20k')}, "
            f"local: {self._sources.count('local')})"
        )

    def _load_ade20k(self, split: str, max_images: Optional[int]):
        """Load indoor images from ADE20K via HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            logger.warning(
                "HuggingFace datasets not installed. "
                "Install with: pip install datasets"
            )
            return

        logger.info(f"Loading ADE20K {split} split from HuggingFace...")
        try:
            ds = load_dataset(
                "scene_parse_150", split=split,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Failed to load ADE20K: {e}")
            logger.info("Trying alternative dataset name...")
            try:
                ds = load_dataset(
                    "huggingface/scene-parse-150", split=split,
                    trust_remote_code=True,
                )
            except Exception as e2:
                logger.error(f"Could not load ADE20K: {e2}")
                return

        logger.info(f"ADE20K loaded: {len(ds)} images in {split} split")

        # Check if scene labels are available for filtering
        has_scene = "scene_category" in ds.column_names if hasattr(ds, "column_names") else False

        count = 0
        for i, sample in enumerate(ds):
            # Filter to indoor if scene labels available
            if has_scene:
                scene = sample.get("scene_category", "")
                if scene and not _is_indoor_scene(str(scene)):
                    continue

            img = sample.get("image")
            if img is None:
                continue

            self._images.append(img)
            self._sources.append("ade20k")
            count += 1

            if max_images and count >= max_images:
                break

        logger.info(
            f"ADE20K: kept {count} images"
            + (" (indoor filtered)" if has_scene else " (all, no scene labels)")
        )

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return preprocessed image tensor (3, H, W) in [0, 1]."""
        item = self._images[idx]

        # Load image
        if isinstance(item, str):
            try:
                img = Image.open(item).convert("RGB")
            except (UnidentifiedImageError, OSError):
                # Return a random valid image on failure
                return self[np.random.randint(len(self))]
        elif isinstance(item, Image.Image):
            img = item.convert("RGB")
        else:
            return self[np.random.randint(len(self))]

        # Resize with center crop to square
        w, h = img.size
        short = min(w, h)
        left = (w - short) // 2
        top = (h - short) // 2
        img = img.crop((left, top, left + short, top + short))
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)

        # To tensor [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

        return tensor

    @property
    def source_distribution(self) -> dict:
        """Count of images by source."""
        from collections import Counter
        return dict(Counter(self._sources))
