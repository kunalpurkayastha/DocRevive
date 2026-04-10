"""
Debug visualisation helpers.

Used when ``config.debug == True`` to draw annotated images after each
pipeline stage.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ── Colour Palette ─────────────────────────────────────────────────────

_COLOURS = {
    "detection": (0, 255, 0),       # green
    "recognition": (255, 165, 0),   # orange
    "blank": (0, 0, 255),           # red
    "edited": (255, 0, 255),        # magenta
    "inpainted": (0, 255, 255),     # yellow
}


def draw_boxes(
    image: np.ndarray,
    boxes: List[List[int]],
    colour: Tuple[int, int, int] = (0, 255, 0),
    label: Optional[str] = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on an image (BGR, returns a copy).

    Args:
        image: Source image (BGR).
        boxes: List of ``[x1, y1, x2, y2]``.
        colour: BGR colour tuple.
        label: Optional label drawn above each box.
        thickness: Line thickness.

    Returns:
        Annotated copy of the image.
    """
    vis = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, thickness)
        if label:
            txt = f"{label}_{i}"
            font_scale = 0.4
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), colour, -1)
            cv2.putText(
                vis, txt, (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
            )
    return vis


def draw_boxes_with_text(
    image: np.ndarray,
    results: List[dict],
    colour: Tuple[int, int, int] = (255, 165, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Draw recognition results with their text labels.

    Args:
        image: Source image (BGR).
        results: List of dicts with ``box`` and ``text`` keys.
        colour: BGR colour.
        thickness: Line thickness.

    Returns:
        Annotated copy of the image.
    """
    vis = image.copy()
    for r in results:
        box = r["box"]
        text = r.get("text", "")
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, thickness)
        cv2.putText(
            vis, text, (x1, y2 + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1,
        )
    return vis


def save_debug_image(
    image: np.ndarray,
    output_dir: str,
    base_name: str,
    stage: str,
) -> str:
    """Save a debug visualisation image.

    Args:
        image: Annotated image to save.
        output_dir: Output directory.
        base_name: Document base name.
        stage: Pipeline stage name (e.g. ``"detection"``).

    Returns:
        Path to the saved image.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{base_name}_{stage}.png")
    cv2.imwrite(path, image)
    return path
