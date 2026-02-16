import numpy as np
import cv2
from typing import Tuple

from .config import DISPLAY_MAX_WIDTH, DISPLAY_MAX_HEIGHT


def resize_for_processing(
    image: np.ndarray, target_size: int = 640, max_size: int = 2560
) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = target_size / min(h, w)
    if scale * max(h, w) > max_size:
        scale = max_size / max(h, w)
    new_w = (int(w * scale) // 32) * 32
    new_h = (int(h * scale) // 32) * 32
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return resized, scale


def resize_mask_to_original(
    mask: np.ndarray, original_size: Tuple[int, int]
) -> np.ndarray:
    return cv2.resize(
        mask.astype(np.uint8),
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def prepare_display_image(
    image: np.ndarray,
    max_width: int = DISPLAY_MAX_WIDTH,
    max_height: int = DISPLAY_MAX_HEIGHT,
) -> np.ndarray:
    h, w = image.shape[:2]
    scale = 1.0
    if w > max_width:
        scale = max_width / w
    if h * scale > max_height:
        scale = max_height / h
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return image
