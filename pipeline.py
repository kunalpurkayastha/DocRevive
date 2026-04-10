"""
DocRevive End-to-End Pipeline.

Orchestrates all modules (detection → recognition → blank extraction →
LLM prediction → text editing → inpainting → post-processing) into a
single ``restore()`` method that takes an image path and produces the
restored document.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import Config
from modules.blank_extraction import (
    BlankExtractor, group_boxes_into_lines, build_prompt_from_patches,
    _rect_x_overlap,
)
from modules.detection import DocumentDetector
from modules.inpainter import TextInpainter
from modules.llm_predictor import LLMPredictor
from modules.occlusion_detector import OcclusionDetector, TRANSPARENT_CLASSES
from modules.postprocessor import PostProcessor
from modules.recognition import TextRecognizer
from modules.text_editor import TextEditor
from utils_pkg.box_utils import is_adjacent, sort_boxes_reading_order
from utils_pkg.image_utils import correct_skew, ensure_pil_image, get_background_color
from utils_pkg.visualizer import draw_boxes, draw_boxes_with_text, save_debug_image

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _dominant_occlusion_type(patches: List[Dict]) -> str:
    """Return the most frequent class_name among detected patches."""
    if not patches:
        return "unknown"
    counts: Dict[str, int] = {}
    for p in patches:
        cn = p.get("class_name", "unknown")
        counts[cn] = counts.get(cn, 0) + 1
    return max(counts, key=counts.get)


def _words_inside_patch(
    patch: Dict,
    recognition_results: List[Dict],
    img_w: int, img_h: int,
) -> List[Dict]:
    """Return OCR words whose centres fall inside *patch*."""
    px1, py1, px2, py2 = patch["x1"], patch["y1"], patch["x2"], patch["y2"]
    inside = []
    for w in recognition_results:
        cx = (w["x1"] + w["x2"]) / 2 * img_w
        cy = (w["y1"] + w["y2"]) / 2 * img_h
        if px1 <= cx <= px2 and py1 <= cy <= py2:
            inside.append(w)
    return inside


def _patch_has_visible_text(
    patch: Dict,
    recognition_results: List[Dict],
    img_w: int, img_h: int,
    coverage_threshold: float = 0.30,
) -> bool:
    """Check if enough recognized words are visible through a transparent patch."""
    px1, py1, px2, py2 = patch["x1"], patch["y1"], patch["x2"], patch["y2"]
    patch_area = max((px2 - px1) * (py2 - py1), 1)
    text_area = 0
    for w in recognition_results:
        wx1, wy1 = w["x1"] * img_w, w["y1"] * img_h
        wx2, wy2 = w["x2"] * img_w, w["y2"] * img_h
        ox = _rect_x_overlap(px1, px2, wx1, wx2)
        oy = _rect_x_overlap(py1, py2, wy1, wy2)
        text_area += ox * oy
    return (text_area / patch_area) >= coverage_threshold



# ═══════════════════════════════════════════════════════════════════════
#  Pipeline
# ═══════════════════════════════════════════════════════════════════════

class DocRevivePipeline:
    """End-to-end document restoration pipeline.

    Modules are lazy-loaded on first use to minimise startup time.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

        # Add project root to sys.path
        if config.project_root not in sys.path:
            sys.path.insert(0, config.project_root)

        # Modules (lazy-init)
        self._recognizer: Optional[TextRecognizer] = None
        self._occlusion_detector: Optional[OcclusionDetector] = None
        self._llm: Optional[LLMPredictor] = None
        self._editor: Optional[TextEditor] = None
        self._inpainter: Optional[TextInpainter] = None
        self._postprocessor: Optional[PostProcessor] = None

    # ── Lazy Initialisation ────────────────────────────────────────

    @property
    def recognizer(self) -> TextRecognizer:
        if self._recognizer is None:
            self._recognizer = TextRecognizer(self.config)
        return self._recognizer

    @property
    def occlusion_detector(self) -> OcclusionDetector:
        if self._occlusion_detector is None:
            self._occlusion_detector = OcclusionDetector(self.config)
        return self._occlusion_detector

    @property
    def llm(self) -> LLMPredictor:
        if self._llm is None:
            self._llm = LLMPredictor(self.config)
        return self._llm

    @property
    def editor(self) -> TextEditor:
        if self._editor is None:
            self._editor = TextEditor(self.config)
        return self._editor

    @property
    def inpainter(self) -> TextInpainter:
        if self._inpainter is None:
            self._inpainter = TextInpainter(self.config)
        return self._inpainter

    @property
    def postprocessor(self) -> PostProcessor:
        if self._postprocessor is None:
            self._postprocessor = PostProcessor(self.config)
        return self._postprocessor

    # ── Eager Pre-load ─────────────────────────────────────────────

    def preload_all(self) -> None:
        """Eagerly initialise every module so the first image has no cold-start.

        Call this once after constructing the pipeline, before processing
        any images.  Each property triggers lazy-init on first access.
        """
        logger.info("Pre-loading all models …")
        _ = self.recognizer          # doctr OCR
        _ = self.occlusion_detector  # YOLO
        _ = self.llm                 # Qwen / HTTP client
        _ = self.editor              # GaMuSA wrapper
        _ = self.inpainter           # GSDM wrapper
        _ = self.postprocessor

        # Eager-load the diffusion model weights (not just the wrapper objects)
        if hasattr(self.editor, '_ensure_loaded'):
            self.editor._ensure_loaded()
        if hasattr(self.inpainter, '_ensure_loaded'):
            self.inpainter._ensure_loaded()

        logger.info("All models loaded and ready.")

    # ── Main Restoration ───────────────────────────────────────────

    def restore(self, image_path: str) -> np.ndarray:
        """Run the full restoration pipeline with occlusion-type-aware routing.

        Args:
            image_path: Path to the input document image.

        Returns:
            Restored document image as BGR NumPy array.
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        doc_output_dir = os.path.join(self.config.output_dir, base_name)
        os.makedirs(doc_output_dir, exist_ok=True)

        # ── Stage 1: Load Image ────────────────────────────────────
        logger.info("Stage 1: Loading image %s", image_path)

        cached_corrected = None
        if self.config.corrected_cache_dir:
            cached_corrected = os.path.join(
                self.config.corrected_cache_dir, f"{base_name}.jpg")
            if not os.path.isfile(cached_corrected):
                cached_corrected = None

        if cached_corrected:
            image = cv2.imread(cached_corrected)
            if image is None:
                raise FileNotFoundError(
                    f"Cached corrected image unreadable: {cached_corrected}")
            logger.info("Loaded pre-corrected image from cache.")
        else:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            angle, image = correct_skew(image)
            logger.info("Skew corrected: %.1f°", angle)

        corrected_path = os.path.join(doc_output_dir, f"{base_name}_corrected.jpg")
        cv2.imwrite(corrected_path, image)
        result_image = image.copy()
        img_h, img_w = image.shape[:2]

        # ── Stage 2-3: Detection + Recognition ─────────────────────
        ocr_cache_path = None
        if self.config.ocr_cache_dir:
            ocr_cache_path = os.path.join(self.config.ocr_cache_dir, f"{base_name}.json")
        if ocr_cache_path and os.path.isfile(ocr_cache_path):
            logger.info("Loading OCR from cache: %s", ocr_cache_path)
            with open(ocr_cache_path) as f:
                data = json.load(f)
            recognition_results = data["recognition_results"]
            raw_text = data.get("raw_text", "")
        else:
            logger.info("Stage 2-3: OCR (detection + recognition)")
            ocr_txt_path = os.path.join(doc_output_dir, f"{base_name}.txt")
            recognition_results, raw_text = self.recognizer.recognize_document(
                image, txt_output_path=ocr_txt_path,
            )

        if not recognition_results:
            logger.warning("No text detected in document.")
            out_path = os.path.join(doc_output_dir, f"{base_name}_restored.png")
            cv2.imwrite(out_path, result_image)
            return result_image

        if self.config.debug:
            boxes_abs = [
                [int(r["x1"] * img_w), int(r["y1"] * img_h),
                 int(r["x2"] * img_w), int(r["y2"] * img_h)]
                for r in recognition_results
            ]
            debug_img = draw_boxes(image, boxes_abs, label="word")
            save_debug_image(debug_img, doc_output_dir, base_name, "ocr")

        # ── Stage 4: Group into Lines ──────────────────────────────
        logger.info("Stage 4: Grouping words into lines")
        lines = group_boxes_into_lines(
            recognition_results, self.config.line_grouping_y_tolerance
        )
        logger.info("Grouped %d words into %d lines.", len(recognition_results), len(lines))

        # ── Stage 5: YOLO Occlusion Detection ─────────────────────
        yolo_cache_path = None
        if self.config.yolo_cache_dir:
            yolo_cache_path = os.path.join(self.config.yolo_cache_dir, f"{base_name}.json")
        if yolo_cache_path and os.path.isfile(yolo_cache_path):
            logger.info("Loading YOLO from cache: %s", yolo_cache_path)
            with open(yolo_cache_path) as f:
                data = json.load(f)
            patch_bboxes = data["patch_bboxes"]
            logger.info("YOLO (cached): %d occlusion patches.", len(patch_bboxes))
        else:
            logger.info("Stage 5: YOLO occlusion detection")
            patch_bboxes = self.occlusion_detector.detect(image)
            logger.info("YOLO detected %d occlusion patches.", len(patch_bboxes))

        if not patch_bboxes:
            logger.info("No occlusion detected; saving original.")
            out_path = os.path.join(doc_output_dir, f"{base_name}_restored.png")
            cv2.imwrite(out_path, result_image)
            return result_image

        dominant_type = _dominant_occlusion_type(patch_bboxes)
        logger.info("Dominant occlusion type: %s", dominant_type)

        if self.config.debug:
            _PATCH_COLOURS = {
                "opaque": (0, 0, 255),
                "transparent": (0, 255, 255),
                "scribble": (255, 0, 255),
            }
            patch_debug = image.copy()
            for p in patch_bboxes:
                kind = "scribble" if p.get("is_scribble") else (
                       "transparent" if p.get("is_transparent") else "opaque")
                colour = _PATCH_COLOURS[kind]
                label = f"{p.get('class_name', '?')} {p['conf']:.2f}"
                cv2.rectangle(patch_debug,
                              (p["x1"], p["y1"]), (p["x2"], p["y2"]),
                              colour, 2)
                cv2.putText(patch_debug, label,
                            (p["x1"], max(p["y1"] - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
            save_debug_image(patch_debug, doc_output_dir, base_name, "patches")

        # ── Stage 5b: Type-Aware Routing ───────────────────────────
        # For transparent occlusions (Stamp/Dust), check if words are
        # readable through the occlusion.  If yes → inpaint-only path.
        transparent_inpaint_only = False
        if dominant_type in TRANSPARENT_CLASSES:
            all_visible = all(
                _patch_has_visible_text(p, recognition_results, img_w, img_h)
                for p in patch_bboxes if p.get("is_transparent")
            )
            if all_visible:
                transparent_inpaint_only = True
                logger.info("Transparent occlusion with visible text → inpaint-only path")

        if transparent_inpaint_only:
            result_image = self._process_transparent_inpaint(
                image, result_image, patch_bboxes, recognition_results,
                lines, img_w, img_h, doc_output_dir, base_name, corrected_path,
            )
            return result_image

        # ── Full Reconstruction Path (all types) ───────────────────
        full_prompt, all_gap_metadata = build_prompt_from_patches(
            lines, patch_bboxes, img_w, img_h
        )

        if not full_prompt or not all_gap_metadata:
            logger.info("No blanks detected; cleaning occlusion regions only.")
            result_image = self.postprocessor.clean_occlusion_regions(
                result_image, patch_bboxes, lines, img_w, img_h
            )
            out_path = os.path.join(doc_output_dir, f"{base_name}_restored.png")
            cv2.imwrite(out_path, result_image)
            return result_image

        logger.info("Detected %d blank regions.", len(all_gap_metadata))

        prompt_path = os.path.join(doc_output_dir, f"{base_name}_prompt.txt")
        with open(prompt_path, "w") as f:
            f.write(full_prompt)

        if self.config.debug:
            blank_boxes_abs = [
                gm["blank_abs_bbox"] for gm in all_gap_metadata
                if gm.get("blank_abs_bbox")
            ]
            debug_img = draw_boxes(image, blank_boxes_abs, colour=(0, 0, 255), label="blank")
            save_debug_image(debug_img, doc_output_dir, base_name, "blanks")

        # ── Stage 6: LLM Prediction ───────────────────────────────
        logger.info("Stage 6: LLM text prediction")
        prediction_dict = self.llm.predict(full_prompt, all_gap_metadata)

        predicted_texts: Dict[int, str] = {}
        for blank_id, info in prediction_dict.items():
            text = info.get("predicted_text", "")
            predicted_texts[blank_id] = text
            logger.info("  blank_%d: '%s'", blank_id, text)

        pred_path = os.path.join(doc_output_dir, f"{base_name}_predictions.txt")
        with open(pred_path, "w") as f:
            for k, v in predicted_texts.items():
                f.write(f"{k}: {v}\n")

        # ── Stage 7: Inpaint adjacent word regions ───────────────
        logger.info("Stage 7: Inpainting adjacent word regions")
        adjacent_boxes: List[List[int]] = []
        for gm in all_gap_metadata:
            pre_w = gm.get("pre_word", {})
            post_w = gm.get("post_word", {})
            if pre_w:
                adjacent_boxes.append([
                    int(pre_w["x1"] * img_w), int(pre_w["y1"] * img_h),
                    int(pre_w["x2"] * img_w), int(pre_w["y2"] * img_h),
                ])
            if post_w:
                adjacent_boxes.append([
                    int(post_w["x1"] * img_w), int(post_w["y1"] * img_h),
                    int(post_w["x2"] * img_w), int(post_w["y2"] * img_h),
                ])

        if adjacent_boxes:
            try:
                inpaint_result_dir = self.inpainter.inpaint_regions(
                    result_image, adjacent_boxes, doc_output_dir,
                )
                if inpaint_result_dir and os.path.isdir(inpaint_result_dir):
                    for i, box in enumerate(adjacent_boxes):
                        fname = f"region_{i:04d}.png"
                        fpath = os.path.join(inpaint_result_dir, fname)
                        if os.path.exists(fpath):
                            patch = cv2.imread(fpath)
                            if patch is not None:
                                x1, y1, x2, y2 = box
                                bw, bh = x2 - x1, y2 - y1
                                if (patch.shape[1], patch.shape[0]) != (bw, bh):
                                    patch = cv2.resize(patch, (bw, bh))
                                result_image[y1:y2, x1:x2] = patch
                    cv2.imwrite(corrected_path, result_image)
                    logger.info("Pasted %d inpainted adjacent regions.", len(adjacent_boxes))
            except Exception as exc:
                logger.error("Inpainting failed: %s", exc)

        # ── Stage 8: Diffusion Text Editing ────────────────────────
        logger.info("Stage 8: Text editing (GaMuSA)")
        doc_pil = ensure_pil_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        blank_bboxes: List[Dict] = []
        for gm in all_gap_metadata:
            blank_bboxes.append({
                "blank_id": gm["blank_id"],
                "pre_word": gm.get("pre_word", {}),
                "post_word": gm.get("post_word", {}),
                "blank_abs_bbox": gm.get("blank_abs_bbox"),
            })

        dataset_dir = os.path.join(doc_output_dir, "dataset")
        edited_dir = os.path.join(doc_output_dir, "edited")

        try:
            blank_bboxes = self.editor.build_editing_dataset(
                doc_pil, blank_bboxes, recognition_results,
                predicted_texts, (img_w, img_h), base_name, dataset_dir,
            )
            self.editor.edit(dataset_dir, edited_dir)
        except Exception as exc:
            logger.error("Text editing failed: %s", exc)

        # ── Stage 9: Paste edited patches + inter-line cleanup ─────
        logger.info("Stage 9: Pasting edited patches and cleaning gaps")
        line_bboxes = [
            [int(min(w["x1"] for w in l) * img_w),
             int(min(w["y1"] for w in l) * img_h),
             int(max(w["x2"] for w in l) * img_w),
             int(max(w["y2"] for w in l) * img_h)]
            for l in lines
        ]

        try:
            final_path = self.postprocessor.paste_edited_patches(
                corrected_path, blank_bboxes, edited_dir,
                doc_output_dir, line_bboxes, base_name,
            )
            if final_path and os.path.exists(final_path):
                result_image = cv2.imread(final_path)
                logger.info("Final result saved: %s", final_path)
            else:
                out_path = os.path.join(doc_output_dir, f"{base_name}_restored.png")
                cv2.imwrite(out_path, result_image)
                logger.info("Fallback saved: %s", out_path)
        except Exception as exc:
            logger.error("Patch pasting failed: %s", exc)
            out_path = os.path.join(doc_output_dir, f"{base_name}_restored.png")
            cv2.imwrite(out_path, result_image)

        # ── Stage 10: Clean remaining occlusion regions ────────────
        logger.info("Stage 10: Cleaning remaining occlusion regions")
        result_image = self.postprocessor.clean_occlusion_regions(
            result_image, patch_bboxes, lines, img_w, img_h
        )
        out_path = os.path.join(doc_output_dir, f"{base_name}_restored.png")
        cv2.imwrite(out_path, result_image)

        return result_image

    # ── Transparent Inpaint-Only Path (Stamp/Dust with visible text) ─

    def _process_transparent_inpaint(
        self,
        image: np.ndarray,
        result_image: np.ndarray,
        patch_bboxes: List[Dict],
        recognition_results: List[Dict],
        lines: List[List[Dict]],
        img_w: int,
        img_h: int,
        output_dir: str,
        base_name: str,
        corrected_path: str,
    ) -> np.ndarray:
        """Handle transparent occlusions where text is still readable.

        Only inpaints word boxes that intersect occlusion patches, then
        cleans the occlusion region background.
        """
        logger.info("Transparent inpaint-only: inpainting words within occlusion patches")

        # Collect word boxes that intersect any transparent patch
        intersecting_boxes: List[List[int]] = []
        for w in recognition_results:
            wx1 = int(w["x1"] * img_w)
            wy1 = int(w["y1"] * img_h)
            wx2 = int(w["x2"] * img_w)
            wy2 = int(w["y2"] * img_h)
            for p in patch_bboxes:
                if not p.get("is_transparent"):
                    continue
                ox = _rect_x_overlap(p["x1"], p["x2"], wx1, wx2)
                oy = _rect_x_overlap(p["y1"], p["y2"], wy1, wy2)
                if ox > 0 and oy > 0:
                    intersecting_boxes.append([wx1, wy1, wx2, wy2])
                    break

        if intersecting_boxes:
            try:
                inpaint_result_dir = self.inpainter.inpaint_regions(
                    result_image, intersecting_boxes, output_dir,
                )
                if inpaint_result_dir and os.path.isdir(inpaint_result_dir):
                    for i, box in enumerate(intersecting_boxes):
                        fname = f"region_{i:04d}.png"
                        fpath = os.path.join(inpaint_result_dir, fname)
                        if os.path.exists(fpath):
                            patch = cv2.imread(fpath)
                            if patch is not None:
                                x1, y1, x2, y2 = box
                                bw, bh = x2 - x1, y2 - y1
                                if (patch.shape[1], patch.shape[0]) != (bw, bh):
                                    patch = cv2.resize(patch, (bw, bh))
                                result_image[y1:y2, x1:x2] = patch
                    logger.info("Inpainted %d word regions within occlusion.", len(intersecting_boxes))
            except Exception as exc:
                logger.error("Transparent inpainting failed: %s", exc)

        # Clean remaining occlusion area
        result_image = self.postprocessor.clean_occlusion_regions(
            result_image, patch_bboxes, lines, img_w, img_h
        )

        out_path = os.path.join(output_dir, f"{base_name}_restored.png")
        cv2.imwrite(out_path, result_image)
        return result_image

    # ── Batch Processing ───────────────────────────────────────────

    def restore_batch(
        self,
        image_paths: List[str],
        progress: bool = True,
        on_result: callable = None,
    ) -> List[str]:
        """Process multiple document images.

        Args:
            image_paths: List of paths to input images.
            progress: Show tqdm progress bar.
            on_result: Optional callback ``(image_path, output_path_or_None,
                       error_str_or_None) -> None`` invoked after each image.

        Returns:
            List of output image paths.
        """
        output_paths: List[str] = []
        n_success = 0

        pbar = tqdm(
            image_paths,
            desc="DocRevive",
            unit="img",
            dynamic_ncols=True,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            ),
        ) if progress else image_paths

        for path in pbar:
            try:
                logger.info("Processing: %s", path)
                result = self.restore(path)
                base_name = os.path.splitext(os.path.basename(path))[0]
                out_path = os.path.join(
                    self.config.output_dir, base_name, f"{base_name}_restored.png"
                )
                output_paths.append(out_path)
                n_success += 1
                if on_result:
                    on_result(path, out_path, None)
            except Exception as exc:
                logger.error("Failed to process %s: %s", path, exc, exc_info=True)
                output_paths.append("")
                if on_result:
                    on_result(path, None, str(exc))

            if progress and hasattr(pbar, "set_postfix_str"):
                n_done = len(output_paths)
                pbar.set_postfix_str(
                    f"ok={n_success} fail={n_done - n_success}"
                )

        return output_paths
