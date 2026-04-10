"""
Document Text Detection module.

Wraps the doctr OCR predictor with ``det_arch='fast_base'`` for document
text detection.  Runs entirely on GPU.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np
import torch

from config import Config

logger = logging.getLogger(__name__)


class DocumentDetector:
    """Detect text bounding boxes in a document image using doctr (FAST).

    Uses the FAST text detector (``fast_base``) via the doctr library.
    The model is loaded onto GPU and kept in memory for repeated use.
    """

    def __init__(self, config: Config) -> None:
        """Initialise the detector.

        Args:
            config: Global pipeline configuration.
        """
        from doctr.models import detection_predictor

        self.device = config.det_device
        logger.info("Loading FAST detector (arch=%s) on %s …", config.det_arch, self.device)

        self.predictor = detection_predictor(
            arch=config.det_arch,
            pretrained=config.det_pretrained,
        )
        # Move model to correct device
        if hasattr(self.predictor, 'model'):
            self.predictor.model = self.predictor.model.to(self.device)

        logger.info("FAST detector loaded successfully.")

    def detect(self, image: np.ndarray) -> List[List[int]]:
        """Detect text regions and return bounding boxes.

        Args:
            image: Document image as BGR NumPy array ``(H, W, 3)``.

        Returns:
            List of ``[x1, y1, x2, y2]`` bounding boxes in absolute
            pixel coordinates, sorted top-to-bottom then left-to-right.
        """
        h_orig, w_orig = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # doctr expects a list of numpy images
        result = self.predictor([rgb])

        boxes: List[List[int]] = []
        # result is a list of pages → each page has a list of blocks
        for page in result:
            for block_boxes in page:
                for box in block_boxes:
                    # box is normalised [x1, y1, x2, y2]
                    x1 = int(box[0] * w_orig)
                    y1 = int(box[1] * h_orig)
                    x2 = int(box[2] * w_orig)
                    y2 = int(box[3] * h_orig)
                    if (x2 - x1) > 2 and (y2 - y1) > 2:
                        boxes.append([x1, y1, x2, y2])

        # Sort reading order: top-to-bottom, left-to-right
        boxes.sort(key=lambda b: (b[1], b[0]))
        logger.info("Detected %d text regions.", len(boxes))
        return boxes
