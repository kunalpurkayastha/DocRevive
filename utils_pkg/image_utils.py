"""
Image utility helpers for cropping, pasting, resizing, and skew correction.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ── Conversion ─────────────────────────────────────────────────────────

def ensure_pil_image(img: np.ndarray | Image.Image) -> Image.Image:
    """Convert a NumPy array to a PIL Image if needed."""
    if isinstance(img, np.ndarray):
        if img.max() <= 1.0:
            return Image.fromarray((img * 255).astype(np.uint8))
        return Image.fromarray(img.astype(np.uint8))
    return img


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) → RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert RGB → BGR (OpenCV)."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ── Crop / Paste ───────────────────────────────────────────────────────

def crop_patch(
    image: np.ndarray,
    box: List[int],
    margin: int = 0,
) -> np.ndarray:
    """Safely crop a patch from *image* given ``[x1,y1,x2,y2]`` box.

    Clamps to image boundaries and optionally expands by *margin* pixels.
    """
    h, w = image.shape[:2]
    x1 = max(0, box[0] - margin)
    y1 = max(0, box[1] - margin)
    x2 = min(w, box[2] + margin)
    y2 = min(h, box[3] + margin)
    return image[y1:y2, x1:x2].copy()


def paste_patch_feathered(
    canvas: np.ndarray,
    patch: np.ndarray,
    box: List[int],
    feather_ksize: int = 15,
) -> np.ndarray:
    """Paste *patch* onto *canvas* at *box* with feathered alpha blending.

    Args:
        canvas: Full document image (BGR, modified in-place).
        patch: Patch to paste, resized to match box dimensions.
        box: ``[x1, y1, x2, y2]`` target location.
        feather_ksize: Gaussian kernel size for feathering mask.

    Returns:
        Modified canvas.
    """
    x1, y1, x2, y2 = box
    h_box, w_box = y2 - y1, x2 - x1
    if h_box <= 0 or w_box <= 0:
        return canvas

    # Resize patch to target dimensions
    resized = cv2.resize(patch, (w_box, h_box), interpolation=cv2.INTER_LINEAR)

    # Create feathered mask
    mask = np.ones((h_box, w_box), dtype=np.float32)
    ksize = feather_ksize if feather_ksize % 2 == 1 else feather_ksize + 1
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    mask = mask[:, :, np.newaxis]  # (H, W, 1) for broadcasting

    # Alpha blend
    region = canvas[y1:y2, x1:x2].astype(np.float32)
    blended = resized.astype(np.float32) * mask + region * (1.0 - mask)
    canvas[y1:y2, x1:x2] = blended.astype(np.uint8)
    return canvas


# ── Skew Correction ───────────────────────────────────────────────────

def correct_skew(
    image: np.ndarray, delta: float = 1.0, limit: float = 5.0
) -> Tuple[float, np.ndarray]:
    """Correct document skew using projection profile analysis.

    Returns:
        ``(best_angle, corrected_image)``.
    """

    def _score(arr: np.ndarray, angle: float) -> float:
        h, w = arr.shape[:2]
        centre = (w // 2, h // 2)
        mat = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rotated = cv2.warpAffine(
            arr, mat, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        histogram = np.sum(rotated, axis=1, dtype=float)
        return float(np.sum((histogram[1:] - histogram[:-1]) ** 2))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    angles = np.arange(-limit, limit + delta, delta)
    scores = [_score(thresh, a) for a in angles]
    best_angle = float(angles[int(np.argmax(scores))])

    h, w = image.shape[:2]
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), best_angle, 1.0)
    corrected = cv2.warpAffine(
        image, mat, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return best_angle, corrected


# ── Misc ───────────────────────────────────────────────────────────────

def get_background_color(image: np.ndarray) -> Tuple[int, ...]:
    """Estimate background colour as the median of border pixels."""
    h, w = image.shape[:2]
    border = np.concatenate([
        image[0, :].reshape(-1, image.shape[2]),
        image[h - 1, :].reshape(-1, image.shape[2]),
        image[:, 0].reshape(-1, image.shape[2]),
        image[:, w - 1].reshape(-1, image.shape[2]),
    ])
    return tuple(int(v) for v in np.median(border, axis=0))


def load_image_for_model(
    image_path: str, height: int = 256, width: int = 256
) -> np.ndarray:
    """Load and resize an image for model input (normalised to [0,1])."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.resize(img, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img
