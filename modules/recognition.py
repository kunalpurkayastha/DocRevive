"""
Document Text Recognition module.

Uses doctr's combined OCR predictor (FAST detection + PARSeq recognition)
to extract text from detected word regions.  All inference runs on GPU.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import Config

logger = logging.getLogger(__name__)


class TextRecognizer:
    """Recognise text in word-level patches using doctr (PARSeq).

    Provides both a combined detect+recognise mode (full document) and
    a patch-level recognise mode (single cropped word image).
    """

    def __init__(self, config: Config) -> None:
        """Initialise the recogniser.

        Args:
            config: Global pipeline configuration.
        """
        from doctr.models import ocr_predictor, recognition_predictor
        import torch

        self.device = config.reco_device
        logger.info(
            "Loading OCR predictor (det=%s, reco=%s) on %s …",
            config.det_arch, config.reco_arch, self.device,
        )

        _torch_device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        # Full OCR predictor (detection + recognition)
        self.ocr_predictor = ocr_predictor(
            det_arch=config.det_arch,
            reco_arch=config.reco_arch,
            pretrained=True,
        ).to(_torch_device)

        # standalone recognition predictor for single-patch inference
        self.reco_predictor = recognition_predictor(
            arch=config.reco_arch,
            pretrained=True,
        ).to(_torch_device)

        logger.info("OCR predictor loaded successfully.")

    # ── Full-Document OCR ──────────────────────────────────────────────

    def recognize_document(
        self,
        image: np.ndarray,
        txt_output_path: Optional[str] = None,
    ) -> Tuple[List[Dict], str]:
        """Run full OCR on a document image.

        Args:
            image: Document image as BGR NumPy array ``(H, W, 3)``.
            txt_output_path: Optional path to save raw OCR text.

        Returns:
            ``(results, raw_text)`` where *results* is a list of dicts
            ``{"box": [x1,y1,x2,y2], "text": str, "confidence": float}``
            with **normalised** coordinates (0–1) and *raw_text* is the
            OCR text file content.
        """
        from doctr.io import DocumentFile

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # doctr expects list of numpy arrays
        doc = [rgb]
        output = self.ocr_predictor(doc, txt_filename=txt_output_path)

        results: List[Dict] = []
        lines_text: List[str] = []

        for page in output.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_words = []
                    for word in line.words:
                        # word.geometry is ((x1,y1), (x2,y2)) normalised
                        (nx1, ny1), (nx2, ny2) = word.geometry
                        results.append({
                            "box": [nx1, ny1, nx2, ny2],
                            "text": word.value,
                            "confidence": word.confidence,
                            "x1": nx1, "y1": ny1,
                            "x2": nx2, "y2": ny2,
                        })
                        line_words.append(word.value)
                    if line_words:
                        lines_text.append(" ".join(line_words))

        raw_text = "\n".join(lines_text)
        logger.info("Recognised %d words across %d lines.", len(results), len(lines_text))
        return results, raw_text

    # ── Patch-Level Recognition ────────────────────────────────────────

    def recognize_patch(self, patch: np.ndarray) -> Tuple[str, float]:
        """Recognise text in a single cropped word image.

        Args:
            patch: Cropped word image (BGR or grayscale).

        Returns:
            ``(text, confidence)``.
        """
        if patch is None or patch.size == 0:
            return "", 0.0

        if len(patch.shape) == 2:
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
        elif patch.shape[2] == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(patch)

        result = self.reco_predictor([np.array(pil_img)])
        if result and len(result) > 0:
            word = result[0]
            if hasattr(word, 'value'):
                return word.value, word.confidence
            elif isinstance(word, tuple) and len(word) >= 2:
                return str(word[0]), float(word[1])

        return "", 0.0

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def parse_ocr_file(txt_path: str) -> List[Dict]:
        """Parse a doctr-format OCR text file into box dicts.

        Each line is: ``x1,y1,x2,y2 "text"`` (normalised coords).

        Returns:
            List of dicts with ``x1, y1, x2, y2, text`` keys.
        """
        boxes: List[Dict] = []
        with open(txt_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('"')
                if len(parts) < 2:
                    continue
                text = parts[1]
                coords_part = parts[0].strip()
                coords_str = coords_part.split(",")
                try:
                    coords = [float(c.strip()) for c in coords_str if c.strip()]
                except ValueError:
                    continue
                if len(coords) < 4:
                    continue
                boxes.append({
                    "x1": coords[0], "y1": coords[1],
                    "x2": coords[2], "y2": coords[3],
                    "text": text,
                })
        return boxes
