"""
Occlusion detector using a finetuned YOLOv9c model.

Supports 6 occlusion classes (trained on the nomixed dataset) and
exposes class-aware confidence thresholds so the pipeline can treat
opaque vs. transparent patches differently.
"""

import logging
from typing import Dict, List

import numpy as np
from ultralytics import YOLO

from config import Config

logger = logging.getLogger(__name__)

# ── Class taxonomy (6-class nomixed YOLOv9c) ─────────────────────────────
YOLO_CLASS_NAMES = {
    0: "Black_Ink",
    1: "White_Burnt",
    2: "White_Whitener",
    3: "Through_Dust",
    4: "Sim",
    5: "Through_Stamp",
}

OPAQUE_CLASSES = {"Black_Ink", "White_Burnt", "White_Whitener"}
TRANSPARENT_CLASSES = {"Through_Dust", "Through_Stamp"}
SCRIBBLE_CLASSES = {"Sim"}

_DEFAULT_CONF = {
    "Black_Ink":       0.20,
    "White_Burnt":     0.20,
    "White_Whitener":  0.20,
    "Through_Dust":    0.30,
    "Sim":             0.25,
    "Through_Stamp":   0.35,
}


class OcclusionDetector:
    """Multi-class YOLO occlusion detector with class-aware filtering."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        if not self.config.yolo_model_path:
            logger.warning("No YOLO model path configured.")
            return
        logger.info("Loading YOLO occlusion model from %s", self.config.yolo_model_path)
        self.model = YOLO(self.config.yolo_model_path)
        device = (
            getattr(self.config, "device", None)
            or getattr(self.config, "det_device", "cuda")
        )
        if device and device != "cpu":
            self.model.to(device)

    # ── Primary API ───────────────────────────────────────────────────────

    def detect(self, image_bgr: np.ndarray) -> List[Dict]:
        """Run YOLO and return class-annotated patch detections.

        Each returned dict contains:
            x1, y1, x2, y2  – absolute pixel coords
            conf             – raw confidence
            class_id         – integer class ID
            class_name       – human-readable class name
            is_opaque        – True if text is fully obscured
            is_transparent   – True if text may be partially visible
            is_scribble      – True if this is a Sim scribble
        """
        self._ensure_loaded()
        if self.model is None:
            return []

        # Use a low base conf; per-class filtering happens below
        base_conf = min(_DEFAULT_CONF.values())

        results = self.model.predict(
            source=image_bgr,
            imgsz=self.config.yolo_imgsz,
            conf=base_conf,
            verbose=False,
        )

        detected: List[Dict] = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().item())
                cls_id = int(box.cls[0].cpu().item())
                cls_name = YOLO_CLASS_NAMES.get(cls_id, f"class_{cls_id}")

                # Per-class confidence gate
                min_conf = _DEFAULT_CONF.get(cls_name, 0.25)
                if conf < min_conf:
                    continue

                detected.append({
                    "x1": int(xyxy[0]),
                    "y1": int(xyxy[1]),
                    "x2": int(xyxy[2]),
                    "y2": int(xyxy[3]),
                    "conf": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "is_opaque": cls_name in OPAQUE_CLASSES,
                    "is_transparent": cls_name in TRANSPARENT_CLASSES,
                    "is_scribble": cls_name in SCRIBBLE_CLASSES,
                })

        logger.info(
            "YOLO detected %d occlusion patches: %s",
            len(detected),
            _class_summary(detected),
        )
        return detected


def _class_summary(detections: List[Dict]) -> str:
    """Short per-class count string for logging."""
    counts: Dict[str, int] = {}
    for d in detections:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
