"""
Bounding-box utility helpers.

All boxes are in ``[x1, y1, x2, y2]`` (top-left, bottom-right) format
unless explicitly noted.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# ── Format Converters ──────────────────────────────────────────────────

def xywh_to_xyxy(box: List[int]) -> List[int]:
    """Convert ``[x, y, w, h]`` → ``[x1, y1, x2, y2]``."""
    x, y, w, h = box
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box: List[int]) -> List[int]:
    """Convert ``[x1, y1, x2, y2]`` → ``[x, y, w, h]``."""
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


# ── IoU / Overlap ──────────────────────────────────────────────────────

def compute_iou(box_a: List[int], box_b: List[int]) -> float:
    """Compute Intersection-over-Union for two ``[x1,y1,x2,y2]`` boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def boxes_horizontally_adjacent(
    box_a: List[int], box_b: List[int], x_gap_tolerance: int
) -> bool:
    """Return *True* if two boxes overlap vertically and their horizontal
    gap is within *x_gap_tolerance* pixels."""
    y_overlap = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    if y_overlap <= 0:
        return False
    x_gap = max(box_b[0] - box_a[2], box_a[0] - box_b[2])
    return x_gap <= x_gap_tolerance


# ── Merging ────────────────────────────────────────────────────────────

def merge_two_boxes(box_a: List[int], box_b: List[int]) -> List[int]:
    """Return the union bounding box of two ``[x1,y1,x2,y2]`` boxes."""
    return [
        min(box_a[0], box_b[0]),
        min(box_a[1], box_b[1]),
        max(box_a[2], box_b[2]),
        max(box_a[3], box_b[3]),
    ]


def merge_overlapping_boxes(
    boxes: List[List[int]],
    iou_threshold: float = 0.0,
    x_gap_tolerance: int = 20,
) -> List[List[int]]:
    """Iteratively merge boxes that overlap (IoU > *iou_threshold*) or are
    horizontally adjacent within *x_gap_tolerance* pixels.

    Args:
        boxes: List of ``[x1, y1, x2, y2]`` boxes.
        iou_threshold: Minimum IoU to trigger a merge.
        x_gap_tolerance: Maximum horizontal pixel gap for adjacency merge.

    Returns:
        Merged list of boxes.
    """
    if not boxes:
        return []

    merged: List[List[int]] = [list(boxes[0])]
    for box in boxes[1:]:
        was_merged = False
        for i, existing in enumerate(merged):
            if (
                compute_iou(existing, box) > iou_threshold
                or boxes_horizontally_adjacent(existing, box, x_gap_tolerance)
            ):
                merged[i] = merge_two_boxes(existing, box)
                was_merged = True
                break
        if not was_merged:
            merged.append(list(box))
    return merged


# ── Sorting ────────────────────────────────────────────────────────────

def sort_boxes_reading_order(boxes: List[List[int]]) -> List[List[int]]:
    """Sort boxes top-to-bottom (by ``y1``), then left-to-right (by ``x1``)."""
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def vertical_center(box: dict) -> float:
    """Return the vertical centre of a normalised box dict."""
    return (box["y1"] + box["y2"]) / 2


# ── Coordinate helpers ─────────────────────────────────────────────────

def clamp_box(
    box: List[int], img_w: int, img_h: int
) -> List[int]:
    """Clamp a box to image boundaries."""
    return [
        max(0, min(box[0], img_w)),
        max(0, min(box[1], img_h)),
        max(0, min(box[2], img_w)),
        max(0, min(box[3], img_h)),
    ]


def is_adjacent(
    blank: Tuple[int, int, int, int],
    candidate: dict,
    img_w: int,
    img_h: int,
    gap_threshold: int = 10,
) -> bool:
    """Check if a candidate OCR box is adjacent to a blank region.

    Args:
        blank: ``(x1, y1, x2, y2)`` of the blank region.
        candidate: Dict with normalised ``x1, y1, x2, y2`` keys.
        img_w: Image width for de-normalisation.
        img_h: Image height for de-normalisation.
        gap_threshold: Maximum pixel gap.

    Returns:
        True if the candidate is adjacent.
    """
    cx1 = int(candidate["x1"] * img_w)
    cy1 = int(candidate["y1"] * img_h)
    cx2 = int(candidate["x2"] * img_w)
    cy2 = int(candidate["y2"] * img_h)

    bx1, by1, bx2, by2 = blank

    # Vertical overlap required
    y_overlap = min(by2, cy2) - max(by1, cy1)
    if y_overlap <= 0:
        return False

    # Horizontal proximity
    left_gap = bx1 - cx2
    right_gap = cx1 - bx2
    return left_gap <= gap_threshold or right_gap <= gap_threshold
