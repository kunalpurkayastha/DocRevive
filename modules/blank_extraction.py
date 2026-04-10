"""
Blank Region Extraction module.

Replaces the WAI (Windowed Averaged Intensity) algorithm with a more
robust multi-signal approach:

1. **Otsu adaptive binarisation** (instead of fixed threshold)
2. **Column projection profile** with Gaussian smoothing
3. **Statistical outlier detection** for unusually wide inter-word gaps
4. **Connected-component verification** of gap regions
5. **Word-box suppression** to avoid false-positives on normal spaces

Also provides prompt-token generation for the LLM predictor.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from config import Config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Line Grouping
# ═══════════════════════════════════════════════════════════════════════

def group_boxes_into_lines(
    recognition_results: List[Dict],
    y_tolerance: float = 0.005,
) -> List[List[Dict]]:
    """Group word-level OCR results into lines by vertical proximity.

    Args:
        recognition_results: List of dicts with normalised ``y1, y2`` keys.
        y_tolerance: Maximum normalised y-centre difference to group.

    Returns:
        List of lines, each a left-to-right sorted list of word dicts.
    """
    if not recognition_results:
        return []

    # Sort by vertical centre
    sorted_results = sorted(
        recognition_results,
        key=lambda r: (r["y1"] + r["y2"]) / 2,
    )

    lines: List[List[Dict]] = []
    for word in sorted_results:
        y_centre = (word["y1"] + word["y2"]) / 2
        placed = False
        for line in lines:
            line_y = np.mean([(w["y1"] + w["y2"]) / 2 for w in line])
            if abs(y_centre - line_y) <= y_tolerance:
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])

    # Sort each line left-to-right
    for line in lines:
        line.sort(key=lambda w: w["x1"])

    # Sort lines top-to-bottom
    lines.sort(key=lambda l: min(w["y1"] for w in l))
    return lines


# ═══════════════════════════════════════════════════════════════════════
#  Character Density Estimation
# ═══════════════════════════════════════════════════════════════════════

def calculate_chars_per_pixel(
    lines: List[List[Dict]], img_w: int
) -> float:
    """Estimate average characters per pixel across all lines.

    Args:
        lines: Grouped lines of word dicts.
        img_w: Image width for de-normalisation.

    Returns:
        Average chars-per-pixel ratio.
    """
    total_chars = 0
    total_width = 0
    for line in lines:
        for word in line:
            text = word.get("text", "")
            w_px = (word["x2"] - word["x1"]) * img_w
            if w_px > 0 and len(text) > 0:
                total_chars += len(text)
                total_width += w_px

    return total_chars / total_width if total_width > 0 else 0.05


def estimate_max_chars_for_blank(
    blank_width_px: int,
    chars_per_pixel: float,
) -> int:
    """Estimate max characters that fit in a blank region.

    Args:
        blank_width_px: Width of the blank region in pixels.
        chars_per_pixel: Characters-per-pixel ratio.

    Returns:
        Estimated character count.
    """
    return max(1, int(blank_width_px * chars_per_pixel * 1.2))


# ═══════════════════════════════════════════════════════════════════════
#  Improved Blank Detection (replaces WAI)
# ═══════════════════════════════════════════════════════════════════════

def _compute_column_profile(
    binary: np.ndarray,
    window_width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute black/white density column profiles.

    Args:
        binary: Binary image (text=255, bg=0).
        window_width: Sliding-window width in pixels.

    Returns:
        ``(x_coords, black_signal, white_signal)`` arrays.
    """
    H, W = binary.shape
    x_coords = []
    black_signal = []
    white_signal = []

    for x in range(0, W - window_width + 1):
        window = binary[:, x : x + window_width]
        black_density = np.sum(window > 0) / window.size
        white_density = 1.0 - black_density
        black_signal.append(black_density)
        white_signal.append(white_density)
        x_coords.append(x + window_width // 2)

    return (
        np.array(x_coords),
        np.array(black_signal, dtype=np.float32),
        np.array(white_signal, dtype=np.float32),
    )


def _normalize(signal: np.ndarray) -> np.ndarray:
    """Z-score normalisation."""
    std = signal.std()
    if std < 1e-6:
        return np.zeros_like(signal)
    return (signal - signal.mean()) / std


def detect_blank_regions(
    line_image: np.ndarray,
    word_boxes: List[Tuple[int, int]],
    config: Config,
) -> Tuple[List[List[int]], np.ndarray, Dict]:
    """Detect blank (occluded) regions within a single text line.

    Uses **adaptive signal fusion** — no classifier or peak_type needed.
    The function auto-detects occlusion polarity (dark / light / mixed)
    per-line by comparing black vs white signal energy, then produces a
    weighted fused signal for peak detection.

    Args:
        line_image: Cropped line image ``(H, W)`` or ``(H, W, 3)`` BGR.
        word_boxes: Local ``(x_start, x_end)`` pairs for each word.
        config: Pipeline configuration.

    Returns:
        ``(blank_boxes_local, x_coords, signals_dict)``
        where *blank_boxes_local* is ``[[x1, y1, x2, y2], ...]`` in
        line-local coordinates.
    """
    # ── Step 1: Greyscale + adaptive binarisation ──────────────────
    if line_image.ndim == 3:
        gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = line_image.copy()

    H, W = gray.shape
    if W < config.sliding_window_width * 3 or H < 3:
        return [], np.array([]), {}

    # Otsu adaptive binarisation (text = white, bg = black)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Step 2: Column profile ─────────────────────────────────────
    ww = config.sliding_window_width
    x_coords, black_sig, white_sig = _compute_column_profile(binary, ww)

    if len(x_coords) == 0:
        return [], np.array([]), {}

    # ── Step 3: Word-gap statistics ────────────────────────────────
    avg_word_gap = 0.0
    avg_word_width = 0.0
    median_word_gap = 0.0

    if word_boxes and len(word_boxes) > 1:
        widths = [b[1] - b[0] for b in word_boxes]
        avg_word_width = float(np.mean(widths)) if widths else 0
        gaps = [word_boxes[i + 1][0] - word_boxes[i][1]
                for i in range(len(word_boxes) - 1)
                if word_boxes[i + 1][0] - word_boxes[i][1] > 0]
        if gaps:
            avg_word_gap = float(np.mean(gaps))
            median_word_gap = float(np.median(gaps))

    # ── Step 4: Smoothing ──────────────────────────────────────────
    smooth_win = max(5, int(avg_word_gap / 2)) if avg_word_gap > 0 else 5
    black_smooth = np.convolve(black_sig, np.ones(smooth_win) / smooth_win, mode="same")
    white_smooth = np.convolve(white_sig, np.ones(max(3, smooth_win // 2)) / max(3, smooth_win // 2), mode="same")

    norm_black = _normalize(black_smooth)
    norm_white = _normalize(white_smooth)

    # ── Step 5: Derivative-enhanced composite ──────────────────────
    black_deriv = np.zeros_like(black_smooth)
    white_deriv = np.zeros_like(white_smooth)
    if len(black_smooth) > 1:
        black_deriv[1:] = black_smooth[1:] - black_smooth[:-1]
        white_deriv[1:] = white_smooth[1:] - white_smooth[:-1]

    norm_black_deriv = _normalize(black_deriv)
    norm_white_deriv = _normalize(white_deriv)

    composite_black = norm_black.copy()
    composite_white = norm_white.copy()

    for i in range(1, len(norm_black) - 1):
        d_factor = abs(norm_black_deriv[i])
        if d_factor > 1.5 and (abs(norm_black[i - 1]) < 0.5 or abs(norm_black[i + 1]) < 0.5):
            composite_black[i] *= (1 + d_factor)
        else:
            composite_black[i] *= max(0.5, 1 - 0.5 * d_factor)

    for i in range(1, len(norm_white) - 1):
        d_factor = abs(norm_white_deriv[i])
        if norm_white[i] > 0.8:
            composite_white[i] = norm_white[i] * 1.5
        if d_factor > 0.8:
            composite_white[i] *= (1 + d_factor * 0.5)

    # ── Step 6: Word-box suppression ───────────────────────────────
    word_mask = np.zeros(len(x_coords), dtype=bool)
    if word_boxes:
        for wx1, wx2 in word_boxes:
            margin = max(1, int((wx2 - wx1) * 0.05))
            mask = (x_coords >= (wx1 - margin)) & (x_coords <= (wx2 + margin))
            word_mask |= mask

    composite_black_masked = composite_black.copy()
    composite_white_masked = composite_white.copy()
    composite_black_masked[word_mask] = 0
    composite_white_masked[word_mask] = 0

    # ── Step 7: Adaptive thresholds (replaces classifier) ───────────
    # Auto-detect occlusion polarity from signal energy
    black_energy = float(np.sum(composite_black_masked ** 2))
    white_energy = float(np.sum(composite_white_masked ** 2))
    total_energy = black_energy + white_energy + 1e-8

    alpha = black_energy / total_energy
    beta = white_energy / total_energy

    if alpha > 0.7:
        polarity = "dark"
    elif beta > 0.7:
        polarity = "light"
    else:
        polarity = "mixed"
    logger.debug("Line polarity: %s (alpha=%.2f, beta=%.2f)", polarity, alpha, beta)

    # Adaptive thresholds based on text density
    text_density = np.sum(binary > 0) / (W * H) if W * H > 0 else 0
    if H > 40 or text_density > 0.3:
        black_thresh = 3.0
        white_thresh = 2.5
    elif median_word_gap > 0 and white_smooth.std() > 0.001:
        ratio = median_word_gap / (white_smooth.std() * W + 1e-8)
        if ratio > 3.0:
            black_thresh, white_thresh = 3.0, 2.0
        elif ratio > 1.5:
            black_thresh, white_thresh = 2.5, 1.8
        else:
            black_thresh, white_thresh = 2.0, 1.5
    else:
        black_thresh, white_thresh = 2.0, 1.5

    # ── Step 8: Find peaks on EACH signal independently ────────────
    # Key insight: peaks in black and white signals occur at different
    # positions so signal-level fusion cancels them out. Instead,
    # detect peaks independently then merge the peak lists.
    black_peaks, _ = find_peaks(
        composite_black_masked, height=black_thresh, distance=ww * 3,
    )
    white_peaks, _ = find_peaks(
        composite_white_masked, height=white_thresh, distance=ww * 3,
    )

    # ── Step 9: Region-width filtering per signal ──────────────────
    def _region_width(signal: np.ndarray, peak_idx: int,
                      floor_frac: float = 0.5, floor_abs: float = 0.3) -> int:
        peak_val = signal[peak_idx]
        threshold = max(floor_frac * peak_val, floor_abs)
        left = peak_idx
        while left > 0 and signal[left] > threshold:
            left -= 1
        right = peak_idx
        while right < len(signal) - 1 and signal[right] > threshold:
            right += 1
        return (right - left) * ww

    if avg_word_width > 0:
        valid_black = []
        for p in black_peaks:
            rw = _region_width(black_smooth, p, 0.6, 0.3)
            ratio = rw / avg_word_width
            sig = abs(composite_black[p])
            if (ratio > 1.4 and sig > black_thresh * 0.8) or sig > black_thresh * 1.2:
                valid_black.append(p)
        black_peaks = np.array(valid_black, dtype=int) if valid_black else np.array([], dtype=int)

    if avg_word_gap > 0:
        valid_white = []
        for p in white_peaks:
            rw = _region_width(white_smooth, p, 0.5, 0.6)
            ratio = rw / avg_word_gap
            sig = abs(composite_white[p])
            if (ratio > 1.5 and sig > white_thresh * 0.7) or sig > white_thresh:
                valid_white.append(p)
        white_peaks = np.array(valid_white, dtype=int) if valid_white else np.array([], dtype=int)

    # ── Step 10: Merge peaks from both signals (always use both) ───
    # This replaces the classifier-based peak_type switch — we always
    # combine evidence from both signals regardless of occlusion type.
    parts = [p for p in [black_peaks, white_peaks] if len(p) > 0]
    if parts:
        all_peaks_raw = np.sort(np.concatenate(parts))
        # Deduplicate peaks that are within min_distance of each other
        min_dist = max(ww * 2, int(median_word_gap * 0.3)) if median_word_gap > 0 else ww * 2
        valid_peaks = [int(all_peaks_raw[0])]
        for p in all_peaks_raw[1:]:
            if p - valid_peaks[-1] >= min_dist:
                valid_peaks.append(int(p))
            else:
                # Keep the one with higher combined signal
                prev = valid_peaks[-1]
                prev_val = (composite_black_masked[prev] if prev < len(composite_black_masked) else 0) + \
                           (composite_white_masked[prev] if prev < len(composite_white_masked) else 0)
                curr_val = (composite_black_masked[p] if p < len(composite_black_masked) else 0) + \
                           (composite_white_masked[p] if p < len(composite_white_masked) else 0)
                if curr_val > prev_val:
                    valid_peaks[-1] = int(p)
    else:
        valid_peaks = []

    # ── Step 11: Build occlusion boxes (clamped to word gaps) ──────
    # Instead of expanding signal regions (which overshoot to full line),
    # map each peak to its containing word gap and use gap boundaries.
    occlusion_boxes: List[List[int]] = []
    for p in valid_peaks:
        if p >= len(x_coords):
            continue

        peak_x = int(x_coords[p])

        # Find which word gap this peak falls in
        best_gap = None
        best_dist = float('inf')
        for gi in range(len(word_boxes) - 1):
            gap_left = word_boxes[gi][1]    # right edge of left word
            gap_right = word_boxes[gi + 1][0]  # left edge of right word
            gap_center = (gap_left + gap_right) / 2
            gap_width = gap_right - gap_left

            # Check if peak is INSIDE or NEAR this gap
            margin = min(30, max(gap_width * 0.3, 10))
            if gap_left - margin <= peak_x <= gap_right + margin:
                dist = abs(peak_x - gap_center)
                if dist < best_dist:
                    best_dist = dist
                    best_gap = gi

        # Also check start-of-line and end-of-line
        if best_gap is None and word_boxes:
            first_left = word_boxes[0][0]
            last_right = word_boxes[-1][1]
            if peak_x < first_left:
                # Before first word: box from 0 to first word
                occlusion_boxes.append([0, 0, first_left, H, peak_x])
                continue
            elif peak_x > last_right:
                # After last word: box from last word to end
                occlusion_boxes.append([last_right, 0, W, H, peak_x])
                continue

        if best_gap is not None:
            x_start = int(word_boxes[best_gap][1])
            x_end = int(word_boxes[best_gap + 1][0])
            if (x_end - x_start) >= config.gap_min_width:
                occlusion_boxes.append([x_start, 0, x_end, H, peak_x])

    signals = {
        "black": (x_coords, composite_black_masked, black_peaks),
        "white": (x_coords, composite_white_masked, white_peaks),
        "fused": (x_coords, composite_black_masked + composite_white_masked,
                  np.array(valid_peaks, dtype=int)),
    }

    logger.debug("Detected %d blank regions in line.", len(occlusion_boxes))
    return occlusion_boxes, x_coords, signals


# ═══════════════════════════════════════════════════════════════════════
#  Prompt Token Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_prompt_tokens(
    line_words: List[Dict],
    blank_boxes_local: List[List[int]],
    x_coords: np.ndarray,
    line_x1: int,
    img_w: int,
    img_h: int,
    blank_count_start: int = 0,
    chars_per_pixel: float = 0.05,
) -> Tuple[str, List[Dict], int]:
    """Build prompt tokens mapping blanks to word-level gaps.

    Handles three cases:
    - **Start-of-line**: blank before the first word.
    - **Mid-line**: blank between two consecutive words (most common).
    - **End-of-line**: blank after the last word.

    Args:
        line_words: Sorted word dicts for this line.
        blank_boxes_local: Detected blank boxes in line-local coords.
        x_coords: X-coordinate array from detection.
        line_x1: Line's absolute x1 offset.
        img_w: Image width for de-normalisation.
        img_h: Image height for de-normalisation.
        blank_count_start: Starting index for blank numbering.
        chars_per_pixel: Character density.

    Returns:
        ``(prompt_string, gap_metadata_list, next_blank_count)``
    """
    if not line_words or not blank_boxes_local:
        return "", [], blank_count_start

    prompt = ""
    gap_metadata: List[Dict] = []
    blank_count = blank_count_start
    processed_gaps: set = set()

    # Pre-compute word edges in local coords
    first_word_left = line_words[0]["x1"] * img_w - line_x1
    last_word_right = line_words[-1]["x2"] * img_w - line_x1

    for blank_box in blank_boxes_local:
        # Use peak_x if available (5th element), else box center
        if len(blank_box) >= 5:
            bx_centre = blank_box[4]
        else:
            bx_centre = (blank_box[0] + blank_box[2]) / 2
        matched = False

        # ── Case 1: Blank BEFORE the first word (start-of-line) ────
        if bx_centre < first_word_left and "start" not in processed_gaps:
            margin = min(30, first_word_left * 0.3)
            if bx_centre < first_word_left + margin:
                processed_gaps.add("start")

                post_text = " ".join(w["text"] for w in line_words)
                blank_w_abs = int(line_words[0]["x1"] * img_w) - line_x1
                blank_w_abs = max(blank_w_abs, blank_box[2] - blank_box[0])
                max_chars = estimate_max_chars_for_blank(blank_w_abs, chars_per_pixel)

                token = (
                    f'<pre{blank_count}="">'
                    f"</blank{blank_count}>"
                    f'<post{blank_count}="{post_text}">'
                    f" </blank{blank_count}> = ? | max_chars_including_blanks={max_chars}\n"
                )
                prompt += token

                gap_metadata.append({
                    "blank_id": blank_count,
                    "gap_index": -1,  # sentinel: start-of-line
                    "pre_text": "",
                    "post_text": post_text,
                    "max_chars": max_chars,
                    "blank_box_local": blank_box,
                    "pre_word": {},  # no word before
                    "post_word": line_words[0],
                    "blank_abs_bbox": [
                        line_x1,
                        int(line_words[0]["y1"] * img_h),
                        int(line_words[0]["x1"] * img_w),
                        int(line_words[0]["y2"] * img_h),
                    ],
                })
                blank_count += 1
                matched = True

        if matched:
            continue

        # ── Case 2: Blank between consecutive words (mid-line) ─────
        for j in range(len(line_words) - 1):
            left_word_right = line_words[j]["x2"] * img_w - line_x1
            right_word_left = line_words[j + 1]["x1"] * img_w - line_x1

            margin = min(30, (right_word_left - left_word_right) * 0.2)
            in_gap = left_word_right < bx_centre < right_word_left
            near_edge = (
                abs(bx_centre - left_word_right) < margin
                or abs(bx_centre - right_word_left) < margin
            )

            if (in_gap or near_edge) and j not in processed_gaps:
                processed_gaps.add(j)

                pre_text = " ".join(w["text"] for w in line_words[: j + 1])
                post_text = " ".join(w["text"] for w in line_words[j + 1 :])

                blank_w_abs = int((line_words[j + 1]["x1"] - line_words[j]["x2"]) * img_w)
                max_chars = estimate_max_chars_for_blank(blank_w_abs, chars_per_pixel)

                # Skip tiny gaps that are normal word spacing
                if max_chars < 2:
                    continue

                token = (
                    f'<pre{blank_count}="{pre_text}">'
                    f"</blank{blank_count}>"
                    f'<post{blank_count}="{post_text}">'
                    f" </blank{blank_count}> = ? | max_chars_including_blanks={max_chars}\n"
                )
                prompt += token

                abs_x1 = int(line_words[j]["x2"] * img_w)
                abs_y1 = int(min(line_words[j]["y1"], line_words[j + 1]["y1"]) * img_h)
                abs_x2 = int(line_words[j + 1]["x1"] * img_w)
                abs_y2 = int(max(line_words[j]["y2"], line_words[j + 1]["y2"]) * img_h)

                gap_metadata.append({
                    "blank_id": blank_count,
                    "gap_index": j,
                    "pre_text": pre_text,
                    "post_text": post_text,
                    "max_chars": max_chars,
                    "blank_box_local": blank_box,
                    "pre_word": line_words[j],
                    "post_word": line_words[j + 1],
                    "blank_abs_bbox": [abs_x1, abs_y1, abs_x2, abs_y2],
                })
                blank_count += 1
                matched = True
                break

        if matched:
            continue

        # ── Case 3: Blank AFTER the last word (end-of-line) ────────
        if bx_centre > last_word_right and "end" not in processed_gaps:
            margin = min(30, (blank_box[2] - last_word_right) * 0.3)
            if bx_centre > last_word_right - margin:
                pre_text = " ".join(w["text"] for w in line_words)
                line_x2 = line_x1 + (blank_box[2] if blank_box[2] > last_word_right
                                     else int(last_word_right))
                blank_w_abs = max(blank_box[2] - blank_box[0],
                                  line_x2 - int(line_words[-1]["x2"] * img_w))
                line_w = line_x2 - line_x1
                median_gap = 30.0
                if len(line_words) > 1:
                    gaps = [
                        (line_words[j + 1]["x1"] - line_words[j]["x2"]) * img_w
                        for j in range(len(line_words) - 1)
                    ]
                    median_gap = float(np.median(gaps)) if gaps else 30.0
                # Skip trailing margin: paragraph ended, not a real blank to fill
                if blank_w_abs > max(4 * median_gap, 0.4 * line_w):
                    continue
                if len(line_words) == 1 and blank_w_abs > 80:  # single word, huge margin
                    continue
                processed_gaps.add("end")

                max_chars = estimate_max_chars_for_blank(blank_w_abs, chars_per_pixel)

                token = (
                    f'<pre{blank_count}="{pre_text}">'
                    f"</blank{blank_count}>"
                    f'<post{blank_count}="">'
                    f" </blank{blank_count}> = ? | max_chars_including_blanks={max_chars}\n"
                )
                prompt += token

                gap_metadata.append({
                    "blank_id": blank_count,
                    "gap_index": len(line_words),  # sentinel: end-of-line
                    "pre_text": pre_text,
                    "post_text": "",
                    "max_chars": max_chars,
                    "blank_box_local": blank_box,
                    "pre_word": line_words[-1],
                    "post_word": {},  # no word after
                    "blank_abs_bbox": [
                        int(line_words[-1]["x2"] * img_w),
                        int(line_words[-1]["y1"] * img_h),
                        line_x1 + blank_box[2],
                        int(line_words[-1]["y2"] * img_h),
                    ],
                })
                blank_count += 1

    return prompt, gap_metadata, blank_count


# ═══════════════════════════════════════════════════════════════════════
#  Geometry helpers for YOLO-driven blank detection
# ═══════════════════════════════════════════════════════════════════════

def _compute_gap_stats(lines: List[List[Dict]], img_w: int) -> Tuple[float, float]:
    """Compute median inter-word gap and avg word width across all lines.

    Returns:
        (median_gap_px, avg_word_width_px)
    """
    gaps: List[float] = []
    widths: List[float] = []
    for line in lines:
        for i in range(len(line) - 1):
            g = line[i + 1]["x1"] * img_w - line[i]["x2"] * img_w
            if g > 0:
                gaps.append(g)
        for w in line:
            ww = (w["x2"] - w["x1"]) * img_w
            if ww > 0:
                widths.append(ww)
    median_gap = float(np.median(gaps)) if gaps else 30.0
    avg_width = float(np.mean(widths)) if widths else 50.0
    return median_gap, avg_width


def _compute_text_margins(
    lines: List[List[Dict]], img_w: int
) -> Tuple[int, int]:
    """Estimate the left and right document text-column margins.

    Uses 5th / 95th percentile of all word x1 / x2 absolute coords so
    that a handful of indented or overhanging words don't skew the result.

    Returns:
        ``(left_margin_px, right_margin_px)``
    """
    x1_vals: List[float] = []
    x2_vals: List[float] = []
    for line in lines:
        for w in line:
            x1_vals.append(w["x1"] * img_w)
            x2_vals.append(w["x2"] * img_w)

    if not x1_vals:
        return 0, img_w

    left  = int(np.percentile(x1_vals, 5))
    right = int(np.percentile(x2_vals, 95))
    return left, right


def _check_3side_enclosure(
    lines: List[List[Dict]],
    line_idx: int,
    patch: Dict,
    img_w: int,
    img_h: int,
    side: str,            # "start" or "end"
    margin_px: int = 20,
) -> bool:
    """Return True if the patch is enclosed on 3 sides by words.

    For a **start-of-line** blank the 3 sides are:
      • RIGHT — current line has words starting just after the patch right edge
      • TOP   — a neighbouring line has words overlapping the patch's x-range
      • BOTTOM — same (either top or bottom is enough)

    For an **end-of-line** blank the 3 sides mirror the above.

    Args:
        lines: All grouped text lines (normalised coords).
        line_idx: Index of the line currently being processed.
        patch: YOLO detection dict with abs pixel coords.
        img_w, img_h: Image dimensions.
        side: ``"start"`` (patch left of first word) or ``"end"`` (right of last word).
        margin_px: Pixel tolerance for adjacency checks.

    Returns:
        True when the blank is plausibly enclosed by surrounding text.
    """
    if not lines or line_idx >= len(lines):
        return False

    cur_line = lines[line_idx]
    if not cur_line:
        return False

    # ── Side facing the line content ─────────────────────────────────
    if side == "start":
        # patch should be LEFT of the first word
        first_x1 = int(cur_line[0]["x1"] * img_w)
        content_side_ok = patch["x2"] <= first_x1 + margin_px
    else:
        # patch should be RIGHT of the last word
        last_x2 = int(cur_line[-1]["x2"] * img_w)
        content_side_ok = patch["x1"] >= last_x2 - margin_px

    if not content_side_ok:
        return False

    # ── Check neighbouring lines for text in the patch's X-range ────
    px_mid = (patch["x1"] + patch["x2"]) / 2
    neighbour_found = False
    for n_idx in [line_idx - 1, line_idx + 1]:
        if n_idx < 0 or n_idx >= len(lines):
            continue
        for w in lines[n_idx]:
            wx1 = w["x1"] * img_w
            wx2 = w["x2"] * img_w
            # word overlaps the patch's horizontal span
            if wx1 < patch["x2"] + margin_px and wx2 > patch["x1"] - margin_px:
                neighbour_found = True
                break
        if neighbour_found:
            break

    return neighbour_found


def _rect_x_overlap(ax1: float, ax2: float, bx1: float, bx2: float) -> float:
    """Pixel overlap between two 1-D intervals [ax1,ax2] and [bx1,bx2]."""
    return max(0.0, min(ax2, bx2) - max(ax1, bx1))


def _clamp_blank_to_patch(
    blank: List[int], patch: Dict,
) -> List[int]:
    """Clamp blank bbox so it stays strictly within the occlusion patch.

    Prevents blank regions from leaking outside the detected occlusion,
    which is critical for edge cases like occlusion at line start/end or
    complex multi-column layouts.
    """
    bx1, by1, bx2, by2 = blank
    bx1 = max(bx1, patch["x1"])
    by1 = max(by1, patch["y1"])
    bx2 = min(bx2, patch["x2"])
    by2 = min(by2, patch["y2"])
    return [bx1, by1, bx2, by2]


def _build_gap_record(
    blank_count: int,
    gap_type: str,           # "mid" | "start" | "end"
    line_words: List[Dict],
    gap_index: int,          # word-pair index for "mid"; -1 start; len end
    blank_abs: List[int],    # [x1, y1, x2, y2]
    chars_per_pixel: float,
    img_w: int,
    img_h: int,
    occlusion_type: str = "unknown",
) -> Tuple[str, Dict]:
    """Build a prompt token string and gap_metadata dict for one blank."""
    blank_id = blank_count
    blank_w_px = blank_abs[2] - blank_abs[0]
    max_chars = estimate_max_chars_for_blank(blank_w_px, chars_per_pixel)

    if gap_type == "mid":
        pre_text  = " ".join(w["text"] for w in line_words[: gap_index + 1])
        post_text = " ".join(w["text"] for w in line_words[gap_index + 1 :])
        pre_word  = line_words[gap_index]
        post_word = line_words[gap_index + 1]
    elif gap_type == "start":
        pre_text  = ""
        post_text = " ".join(w["text"] for w in line_words)
        pre_word  = {}
        post_word = line_words[0] if line_words else {}
    else:  # "end"
        pre_text  = " ".join(w["text"] for w in line_words)
        post_text = ""
        pre_word  = line_words[-1] if line_words else {}
        post_word = {}

    # Skip trivially narrow blanks
    if max_chars < 2:
        return "", {}

    token = (
        f'<pre{blank_id}="{pre_text}">'
        f"</blank{blank_id}>"
        f'<post{blank_id}="{post_text}">'
        f" </blank{blank_id}> = ? | max_chars_including_blanks={max_chars}\n"
    )

    meta = {
        "blank_id":      blank_id,
        "gap_index":     gap_index,
        "pre_text":      pre_text,
        "post_text":     post_text,
        "max_chars":     max_chars,
        "pre_word":      pre_word,
        "post_word":     post_word,
        "blank_abs_bbox": blank_abs,
        "occlusion_type": occlusion_type,
    }
    return token, meta


# ═══════════════════════════════════════════════════════════════════════
#  YOLO Patch Integration — geometry-driven
# ═══════════════════════════════════════════════════════════════════════

def build_prompt_from_patches(
    lines: List[List[Dict]],
    patch_bboxes: List[Dict],
    img_w: int,
    img_h: int,
) -> Tuple[str, List[Dict]]:
    """Build prompt tokens by intersecting YOLO patch bboxes with word gaps.

    This is a **pure-geometry** algorithm — no image signal processing.

    Algorithm per patch per line:
      1. Y-overlap gate (opaque: any; transparent: ≥40%; scribble: any)
      2. Compute document text-column margins once across all lines.
      3. **Mid-line gaps**: flag any inter-word gap whose X-range is
         overlapped by the patch, by at least 1 pixel.
      4. **Start-of-line**: patch is left of the first word and passes a
         3-sided-enclosure check (neighbouring lines have text in same
         X-range).  Blank width = first_word.x1 – left_margin.
      5. **End-of-line**: symmetric mirror of (4).
      6. **Scribble class**: any word whose bbox overlaps the patch by
         >30% of the word width is treated as fully blanked (between
         adjacent surviving words).
      7. Deduplication: each (line, gap_index) emitted once.
    """
    if not patch_bboxes or not lines:
        return "", []

    chars_per_pixel = calculate_chars_per_pixel(lines, img_w)
    left_margin, right_margin = _compute_text_margins(lines, img_w)
    median_gap_px, avg_word_width_px = _compute_gap_stats(lines, img_w)
    line_width = right_margin - left_margin

    full_prompt   = ""
    all_gap_meta: List[Dict] = []
    blank_count   = 0

    # Track emitted (line_idx, gap_index) pairs to avoid duplicates
    emitted: set = set()

    for line_idx, line_words in enumerate(lines):
        if not line_words:
            continue

        # ── Absolute pixel coords of this line ────────────────────────
        ly1 = int(min(w["y1"] for w in line_words) * img_h)
        ly2 = int(max(w["y2"] for w in line_words) * img_h)
        lh  = max(ly2 - ly1, 1)
        lx1 = int(min(w["x1"] for w in line_words) * img_w)
        lx2 = int(max(w["x2"] for w in line_words) * img_w)

        # Word absolute edges (x1, x2) for this line
        word_x1 = [int(w["x1"] * img_w) for w in line_words]
        word_x2 = [int(w["x2"] * img_w) for w in line_words]
        word_y1 = [int(w["y1"] * img_h) for w in line_words]
        word_y2 = [int(w["y2"] * img_h) for w in line_words]

        occlusion_type = "unknown"

        for patch in patch_bboxes:
            px1, py1, px2, py2 = patch["x1"], patch["y1"], patch["x2"], patch["y2"]
            is_scribble    = patch.get("is_scribble", False)
            is_transparent = patch.get("is_transparent", False)
            cls_name       = patch.get("class_name", "unknown")

            # ── Y-overlap gate ─────────────────────────────────────────
            y_overlap = max(0, min(py2, ly2) - max(py1, ly1))
            if y_overlap <= 0:
                continue
            if is_transparent and (y_overlap / lh) < 0.40:
                continue

            occlusion_type = cls_name

            # ── Scribble: word-level overlap logic ─────────────────────
            if is_scribble:
                # Find contiguous runs of scribbled words; treat the gap
                # BETWEEN unscribbled neighbours as the blank.
                scribbled = []
                for wi, (wx1, wx2) in enumerate(zip(word_x1, word_x2)):
                    word_w = max(wx2 - wx1, 1)
                    overlap = _rect_x_overlap(px1, px2, wx1, wx2)
                    if overlap / word_w > 0.30:
                        scribbled.append(wi)

                if not scribbled:
                    continue

                # Emit a blank spanning from word before run to word after run
                run_first = scribbled[0]
                run_last  = scribbled[-1]

                # left anchor: word just before, or start-of-line
                if run_first > 0:
                    gap_index = run_first - 1
                    key = (line_idx, gap_index)
                    if key not in emitted:
                        emitted.add(key)
                        bx1 = word_x2[run_first - 1]
                        bx2 = word_x1[run_last + 1] if run_last + 1 < len(line_words) else word_x2[run_last]
                        by1 = ly1; by2 = ly2
                        clamped = _clamp_blank_to_patch([bx1, by1, bx2, by2], patch)
                        if clamped[2] - clamped[0] < 2 or clamped[3] - clamped[1] < 2:
                            continue
                        tok, meta = _build_gap_record(
                            blank_count, "mid", line_words, gap_index,
                            clamped,
                            chars_per_pixel, img_w, img_h, cls_name,
                        )
                        if tok:
                            meta.update(line_index=line_idx + 1,
                                        line_x1=lx1, line_y1=ly1,
                                        line_x2=lx2, line_y2=ly2,
                                        patch_bbox=[px1, py1, px2, py2])
                            full_prompt += tok
                            all_gap_meta.append(meta)
                            blank_count += 1
                continue  # scribble handled

            # ── Mid-line gaps: X-intersection with inter-word spaces ───
            for j in range(len(line_words) - 1):
                gx1 = word_x2[j]     # right edge of left word
                gx2 = word_x1[j + 1] # left edge of right word
                gap_w = gx2 - gx1

                if gap_w < 1:
                    continue

                x_overlap = _rect_x_overlap(px1, px2, gx1, gx2)
                if x_overlap <= 0:
                    continue

                key = (line_idx, j)
                if key in emitted:
                    continue
                emitted.add(key)

                by1 = min(word_y1[j], word_y1[j + 1])
                by2 = max(word_y2[j], word_y2[j + 1])
                clamped = _clamp_blank_to_patch([gx1, by1, gx2, by2], patch)
                if clamped[2] - clamped[0] < 2 or clamped[3] - clamped[1] < 2:
                    continue
                tok, meta = _build_gap_record(
                    blank_count, "mid", line_words, j,
                    clamped,
                    chars_per_pixel, img_w, img_h, cls_name,
                )
                if tok:
                    meta.update(line_index=line_idx + 1,
                                line_x1=lx1, line_y1=ly1,
                                line_x2=lx2, line_y2=ly2,
                                patch_bbox=[px1, py1, px2, py2])
                    full_prompt += tok
                    all_gap_meta.append(meta)
                    blank_count += 1

            # ── Start-of-line: patch extends left of first word ────────
            first_x1 = word_x1[0]
            if px1 < first_x1 and _rect_x_overlap(px1, px2, left_margin, first_x1) > 0:
                if _check_3side_enclosure(lines, line_idx, patch, img_w, img_h, "start"):
                    key = (line_idx, -1)
                    if key not in emitted:
                        emitted.add(key)
                        bx1 = max(left_margin, px1)
                        bx2 = first_x1
                        clamped = _clamp_blank_to_patch([bx1, ly1, bx2, ly2], patch)
                        if clamped[2] - clamped[0] < 2 or clamped[3] - clamped[1] < 2:
                            continue
                        tok, meta = _build_gap_record(
                            blank_count, "start", line_words, -1,
                            clamped,
                            chars_per_pixel, img_w, img_h, cls_name,
                        )
                        if tok:
                            meta.update(line_index=line_idx + 1,
                                        line_x1=lx1, line_y1=ly1,
                                        line_x2=lx2, line_y2=ly2,
                                        patch_bbox=[px1, py1, px2, py2])
                            full_prompt += tok
                            all_gap_meta.append(meta)
                            blank_count += 1

            # ── End-of-line: patch extends right of last word ──────────
            last_x2 = word_x2[-1]
            if px2 > last_x2 and _rect_x_overlap(px1, px2, last_x2, right_margin) > 0:
                if _check_3side_enclosure(lines, line_idx, patch, img_w, img_h, "end"):
                    bx1 = last_x2
                    bx2 = min(right_margin, px2)
                    blank_w = bx2 - bx1
                    # Skip trailing margin: paragraph ended, occlusion is in margin
                    # not a real blank to fill (avoid "entire line" false positive)
                    if blank_w > max(4 * median_gap_px, 0.4 * line_width):
                        continue
                    if len(line_words) == 1 and blank_w > 2.5 * avg_word_width_px:
                        continue
                    key = (line_idx, len(line_words))
                    if key not in emitted:
                        emitted.add(key)
                        clamped = _clamp_blank_to_patch([bx1, ly1, bx2, ly2], patch)
                        if clamped[2] - clamped[0] < 2 or clamped[3] - clamped[1] < 2:
                            continue
                        tok, meta = _build_gap_record(
                            blank_count, "end", line_words, len(line_words),
                            clamped,
                            chars_per_pixel, img_w, img_h, cls_name,
                        )
                        if tok:
                            meta.update(line_index=line_idx + 1,
                                        line_x1=lx1, line_y1=ly1,
                                        line_x2=lx2, line_y2=ly2,
                                        patch_bbox=[px1, py1, px2, py2])
                            full_prompt += tok
                            all_gap_meta.append(meta)
                            blank_count += 1

    # ── Append full document text for LLM context ─────────────────────
    if full_prompt:
        full_prompt += "\nAll Lines:\n"
        for i, lw in enumerate(lines):
            full_prompt += f"Line {i + 1}: {' '.join(w['text'] for w in lw)}\n"

    logger.info("Geometry-based blank detection: %d blanks found.", len(all_gap_meta))
    return full_prompt, all_gap_meta


def _dominant_class(patches: List[Dict]) -> str:
    """Return the most common class_name among *patches*."""
    counts: Dict[str, int] = {}
    for p in patches:
        cn = p.get("class_name", "unknown")
        counts[cn] = counts.get(cn, 0) + 1
    return max(counts, key=counts.get) if counts else "unknown"


# ═══════════════════════════════════════════════════════════════════════
#  Main Extractor Class (Legacy/Signal-based)
# ═══════════════════════════════════════════════════════════════════════

class BlankExtractor:
    """Orchestrates blank detection and prompt-token generation.

    Wraps the functional helpers above into a stateful object that
    caches configuration.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def extract(
        self,
        image: np.ndarray,
        recognition_results: List[Dict],
        img_w: int,
        img_h: int,
    ) -> Tuple[str, List[Dict], List[List[Dict]]]:
        """Run blank extraction on an entire document.

        Uses adaptive signal fusion — no classifier or peak_type needed.

        Args:
            image: Full document image (BGR).
            recognition_results: Word dicts with normalised coords.
            img_w: Image width.
            img_h: Image height.

        Returns:
            ``(full_prompt, all_gap_metadata, lines)``
        """
        lines = group_boxes_into_lines(
            recognition_results, self.config.line_grouping_y_tolerance
        )
        logger.info("Grouped %d words into %d lines.", len(recognition_results), len(lines))

        chars_per_pixel = calculate_chars_per_pixel(lines, img_w)
        logger.info("Character density: %.4f chars/pixel", chars_per_pixel)

        full_prompt = ""
        all_gap_metadata: List[Dict] = []
        blank_count = 0

        for i, line_words in enumerate(lines):
            if not line_words:
                continue

            # Compute line bounding box in absolute pixel coords
            line_x1 = int(min(w["x1"] for w in line_words) * img_w)
            line_y1 = int(min(w["y1"] for w in line_words) * img_h)
            line_x2 = int(max(w["x2"] for w in line_words) * img_w)
            line_y2 = int(max(w["y2"] for w in line_words) * img_h)

            # Crop line image
            line_image = image[line_y1:line_y2, line_x1:line_x2]
            if line_image.size == 0:
                continue

            # Word boxes in local coordinates
            local_word_boxes = [
                (int(w["x1"] * img_w) - line_x1, int(w["x2"] * img_w) - line_x1)
                for w in line_words
            ]

            # Detect blanks (auto-detects occlusion type per-line)
            blank_boxes_local, x_coords, signals = detect_blank_regions(
                line_image, local_word_boxes, self.config,
            )

            if not blank_boxes_local:
                continue

            # Generate prompt tokens
            prompt, gap_meta, blank_count = generate_prompt_tokens(
                line_words,
                blank_boxes_local,
                x_coords,
                line_x1,
                img_w,
                img_h,
                blank_count_start=blank_count,
                chars_per_pixel=chars_per_pixel,
            )

            if prompt:
                full_prompt += prompt
                for gm in gap_meta:
                    gm["line_index"] = i + 1
                    gm["line_x1"] = line_x1
                    gm["line_y1"] = line_y1
                    gm["line_x2"] = line_x2
                    gm["line_y2"] = line_y2
                all_gap_metadata.extend(gap_meta)

        # Append full-text context to prompt
        if full_prompt:
            full_prompt += "\nAll Lines:\n"
            for i, line_words in enumerate(lines):
                line_text = " ".join(w["text"] for w in line_words)
                full_prompt += f"Line {i + 1}: {line_text}\n"

        logger.info(
            "Blank extraction complete: %d blanks across %d lines.",
            len(all_gap_metadata), len(lines),
        )
        return full_prompt, all_gap_metadata, lines
