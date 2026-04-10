"""
Post-Processor module.

Handles patch cleaning, seamless integration back into the document,
and inter-line gap filling.  The core ``paste_edited_patches`` method
mirrors ``paste_edited_blanks_trim_no_scale`` from ``e2e_working.py``.
"""

from __future__ import annotations

import glob
import logging
import os
from collections import deque
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from config import Config

logger = logging.getLogger(__name__)


# ── Standalone helpers (used by paste logic) ──────────────────────────

def _get_background_color(pil_img: Image.Image) -> Tuple[int, ...]:
    """Estimate background colour from border pixels of a PIL Image."""
    arr = np.array(pil_img)
    h, w = arr.shape[:2]
    border = np.concatenate([
        arr[0, :].reshape(-1, arr.shape[2]),
        arr[h - 1, :].reshape(-1, arr.shape[2]),
        arr[:, 0].reshape(-1, arr.shape[2]),
        arr[:, w - 1].reshape(-1, arr.shape[2]),
    ])
    med = np.median(border, axis=0).astype(np.uint8)
    return tuple(int(v) for v in med)


def _remove_dark_background(pil_img: Image.Image, threshold: int = 40) -> Image.Image:
    """Remove dark edge backgrounds via flood-fill from corners."""
    arr = np.array(pil_img)
    if arr.ndim < 3:
        return pil_img

    h, w, _ = arr.shape
    mask = np.zeros((h, w), dtype=bool)

    def is_dark(r, g, b):
        return r < threshold and g < threshold and b < threshold

    queue: deque = deque()
    for x, y in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if is_dark(*arr[y, x, :3]):
            queue.append((x, y))
            mask[y, x] = True

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and not mask[ny, nx]:
                if is_dark(*arr[ny, nx, :3]):
                    mask[ny, nx] = True
                    queue.append((nx, ny))

    arr[mask] = 255
    return Image.fromarray(arr)


def _trim_whitespace(pil_img: Image.Image, threshold: int = 240) -> Image.Image:
    """Trim near-white margins from a PIL image."""
    arr = np.array(pil_img)
    if arr.ndim == 3:
        gray = np.mean(arr, axis=2)
    else:
        gray = arr.astype(float)

    non_white = gray < threshold
    rows = np.any(non_white, axis=1)
    cols = np.any(non_white, axis=0)

    if not np.any(rows) or not np.any(cols):
        return pil_img

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return pil_img.crop((cmin, rmin, cmax + 1, rmax + 1))


class PostProcessor:
    """Cleans edited patches and integrates them back into the document."""

    def __init__(self, config: Config) -> None:
        self.config = config

    # ── Inter-Line Gap Filling ─────────────────────────────────────

    def fill_inter_line_gaps(
        self,
        image: np.ndarray,
        line_boxes: List[List[int]],
    ) -> np.ndarray:
        """Fill gaps between consecutive text lines with background colour."""
        if len(line_boxes) < 2:
            return image

        sorted_boxes = sorted(line_boxes, key=lambda b: b[1])
        h, w = image.shape[:2]

        border_strip = 5
        border_pixels = np.concatenate([
            image[:border_strip, :].reshape(-1, image.shape[2]),
            image[-border_strip:, :].reshape(-1, image.shape[2]),
            image[:, :border_strip].reshape(-1, image.shape[2]),
            image[:, -border_strip:].reshape(-1, image.shape[2]),
        ])
        bg_colour = np.median(border_pixels, axis=0).astype(np.uint8)

        threshold = self.config.inter_line_gap_threshold

        for i in range(len(sorted_boxes) - 1):
            curr_bottom = sorted_boxes[i][3]
            next_top = sorted_boxes[i + 1][1]
            gap = next_top - curr_bottom
            if gap <= threshold:
                continue

            x_start = max(0, min(sorted_boxes[i][0], sorted_boxes[i + 1][0]))
            x_end = min(w, max(sorted_boxes[i][2], sorted_boxes[i + 1][2]))
            image[curr_bottom:next_top, x_start:x_end] = bg_colour

        return image

    # ── Main Paste Logic ───────────────────────────────────────────

    def paste_edited_patches(
        self,
        doc_image_path: str,
        blank_bboxes: List[dict],
        edited_dir: str,
        output_dir: str,
        line_bboxes: List[List[int]],
        doc_name: str,
    ) -> str:
        """Paste edited patches back into the document.

        Mirrors ``paste_edited_blanks_trim_no_scale`` from
        ``e2e_working.py``:

        1. Fill each blank bbox with the documents background colour.
        2. For each sub-word patch, apply dark-background removal +
           whitespace trimming.
        3. Centre the trimmed patch inside the sub-box without scaling.

        Args:
            doc_image_path: Path to the corrected document image.
            blank_bboxes: Blank metadata (must include ``sub_boxes``).
            edited_dir: Directory containing GaMuSA output patches.
            output_dir: Where to save the final image.
            line_bboxes: Line bounding boxes for gap cleaning.
            doc_name: Document base name.

        Returns:
            Path to the final restored document.
        """
        if not os.path.exists(doc_image_path):
            logger.error("Document image not found: %s", doc_image_path)
            return doc_image_path

        doc_img = Image.open(doc_image_path).convert("RGB")
        bg_color = _get_background_color(doc_img)
        draw = ImageDraw.Draw(doc_img)
        processed_count = 0

        # Collect all edited patch files (search several locations)
        search_dirs = [
            edited_dir,
            output_dir,
            os.path.join(output_dir, doc_name),
        ]
        all_edited_files: List[str] = []
        for d in search_dirs:
            if os.path.exists(d):
                all_edited_files.extend(
                    glob.glob(os.path.join(d, f"{doc_name}*.png"))
                )

        logger.info("Found %d candidate edited patch files.", len(all_edited_files))

        for blank in blank_bboxes:
            blank_id = blank.get("blank_id", blank.get("id", 0))
            sub_boxes = blank.get("sub_boxes", [])

            if not sub_boxes:
                continue

            # ── Step 1: Fill the entire blank region with bg colour ─
            pre_word = blank.get("pre_word", {})
            post_word = blank.get("post_word", {})

            # Use pre-computed absolute bbox if available (handles
            # start-of-line and end-of-line blanks)
            abs_bbox = blank.get("blank_abs_bbox")
            if abs_bbox:
                bx1, by1, bx2, by2 = abs_bbox
            elif pre_word and post_word:
                img_w, img_h = doc_img.size
                bx1 = int(pre_word.get("x2", 0) * img_w)
                by1 = int(min(pre_word.get("y1", 0), post_word.get("y1", 0)) * img_h)
                bx2 = int(post_word.get("x1", 1) * img_w)
                by2 = int(max(pre_word.get("y2", 1), post_word.get("y2", 1)) * img_h)
            elif pre_word:
                # End-of-line: blank after the last word
                img_w, img_h = doc_img.size
                bx1 = int(pre_word.get("x2", 0) * img_w)
                by1 = int(pre_word.get("y1", 0) * img_h)
                bx2 = bx1 + 100  # fallback width
                by2 = int(pre_word.get("y2", 1) * img_h)
            elif post_word:
                # Start-of-line: blank before the first word
                img_w, img_h = doc_img.size
                bx2 = int(post_word.get("x1", 1) * img_w)
                by1 = int(post_word.get("y1", 0) * img_h)
                bx1 = max(0, bx2 - 100)  # fallback width
                by2 = int(post_word.get("y2", 1) * img_h)
            else:
                continue

            if bx2 > bx1 and by2 > by1:
                draw.rectangle([bx1, by1, bx2, by2], fill=bg_color)

            # ── Step 2: Paste each sub-word patch ──────────────────
            sub_count = 0
            for s_idx, sub_info in enumerate(sub_boxes):
                sx1, sy1, sx2, sy2 = (
                    int(sub_info[0]), int(sub_info[1]),
                    int(sub_info[2]), int(sub_info[3]),
                )

                # Try multiple filename patterns
                candidates = [
                    os.path.join(edited_dir, f"{doc_name}_b{blank_id}_w{s_idx}.png"),
                    os.path.join(output_dir, f"{doc_name}_b{blank_id}_w{s_idx}.png"),
                    os.path.join(edited_dir, f"{doc_name}_blank{blank_id}_word{s_idx}.png"),
                    os.path.join(output_dir, f"{doc_name}_blank{blank_id}_word{s_idx}.png"),
                ]
                # Also check all_edited_files
                for ef in all_edited_files:
                    if f"b{blank_id}_w{s_idx}" in ef or f"blank{blank_id}_word{s_idx}" in ef:
                        candidates.append(ef)

                patch_path = None
                for p in candidates:
                    if os.path.exists(p):
                        patch_path = p
                        break

                if patch_path is None:
                    logger.debug("No edited patch for blank %d sub %d", blank_id, s_idx)
                    continue

                try:
                    edited_img = Image.open(patch_path).convert("RGB")

                    # Clean dark edges, trim white margins
                    cleaned = _remove_dark_background(edited_img, threshold=40)
                    trimmed = _trim_whitespace(cleaned, threshold=240)
                    tw, th = trimmed.size

                    bw = sx2 - sx1
                    bh = sy2 - sy1

                    # Centre inside the sub-box (no scaling)
                    offset_x = sx1 + max(0, (bw - tw) // 2)
                    offset_y = sy1 + max(0, (bh - th) // 2)

                    doc_img.paste(trimmed, (offset_x, offset_y))
                    sub_count += 1
                    logger.debug(
                        "Pasted blank_%d sub_%d at (%d, %d), size %dx%d",
                        blank_id, s_idx, offset_x, offset_y, tw, th,
                    )
                except Exception as exc:
                    logger.error("Failed to paste blank %d sub %d: %s", blank_id, s_idx, exc)

            if sub_count > 0:
                processed_count += 1

        logger.info("Pasted %d/%d blank regions.", processed_count, len(blank_bboxes))

        # ── Inter-line gap cleaning ────────────────────────────────
        if line_bboxes:
            # Convert PIL → numpy, clean, convert back
            arr = np.array(doc_img)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            arr = self.fill_inter_line_gaps(arr, line_bboxes)
            doc_img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

        # ── Save ───────────────────────────────────────────────────
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{doc_name}_restored.png")
        doc_img.save(out_path)
        logger.info("Final document saved: %s", out_path)
        return out_path

    # ── Occlusion Region Background Cleaning ──────────────────────

    def clean_occlusion_regions(
        self,
        image: np.ndarray,
        patch_bboxes: List[Dict],
        lines: List[List[Dict]],
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        """Fill non-text areas within occlusion patches with background colour.

        For each YOLO occlusion patch, identifies sub-regions that don't
        overlap any text line and fills them with the document background.
        Also fills inter-line gaps within the occlusion.
        """
        if not patch_bboxes:
            return image

        h, w = image.shape[:2]
        border_strip = 5
        border_pixels = np.concatenate([
            image[:border_strip, :].reshape(-1, image.shape[2]),
            image[-border_strip:, :].reshape(-1, image.shape[2]),
            image[:, :border_strip].reshape(-1, image.shape[2]),
            image[:, -border_strip:].reshape(-1, image.shape[2]),
        ])
        bg_bgr = np.median(border_pixels, axis=0).astype(np.uint8)

        # Build absolute line bboxes for overlap checks
        line_boxes: List[Tuple[int, int, int, int]] = []
        for line in lines:
            if not line:
                continue
            lx1 = int(min(wd["x1"] for wd in line) * img_w)
            ly1 = int(min(wd["y1"] for wd in line) * img_h)
            lx2 = int(max(wd["x2"] for wd in line) * img_w)
            ly2 = int(max(wd["y2"] for wd in line) * img_h)
            line_boxes.append((lx1, ly1, lx2, ly2))

        for patch in patch_bboxes:
            px1 = max(patch["x1"], 0)
            py1 = max(patch["y1"], 0)
            px2 = min(patch["x2"], w)
            py2 = min(patch["y2"], h)
            if px2 <= px1 or py2 <= py1:
                continue

            # Find lines that overlap this patch vertically
            overlapping = [
                lb for lb in line_boxes
                if lb[3] > py1 and lb[1] < py2
            ]

            if not overlapping:
                # No text lines here at all → fill entire patch with bg
                image[py1:py2, px1:px2] = bg_bgr
                continue

            overlapping.sort(key=lambda b: b[1])

            # Fill region above first overlapping line
            first_top = overlapping[0][1]
            if first_top > py1:
                image[py1:first_top, px1:px2] = bg_bgr

            # Fill inter-line gaps within the patch
            for i in range(len(overlapping) - 1):
                gap_top = overlapping[i][3]
                gap_bot = overlapping[i + 1][1]
                if gap_bot > gap_top:
                    gx1 = max(px1, min(overlapping[i][0], overlapping[i + 1][0]))
                    gx2 = min(px2, max(overlapping[i][2], overlapping[i + 1][2]))
                    image[gap_top:gap_bot, gx1:gx2] = bg_bgr

            # Fill region below last overlapping line
            last_bot = overlapping[-1][3]
            if last_bot < py2:
                image[last_bot:py2, px1:px2] = bg_bgr

            # Fill left/right margins within each line's vertical span
            for lb in overlapping:
                lx1, ly1_l, lx2, ly2_l = lb
                row_top = max(py1, ly1_l)
                row_bot = min(py2, ly2_l)
                if row_bot <= row_top:
                    continue
                if px1 < lx1:
                    image[row_top:row_bot, px1:min(lx1, px2)] = bg_bgr
                if px2 > lx2:
                    image[row_top:row_bot, max(lx2, px1):px2] = bg_bgr

        return image
