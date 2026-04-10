"""
Text Inpainting module.

Wraps the existing GSDM (Generative Score-based Diffusion Model)
inpainting model for removing/cleaning visible text overlapping with
stamps or noise.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

from config import Config

logger = logging.getLogger(__name__)


class TextInpainter:
    """Inpaint text regions using the GSDM model.

    Creates binary masks for target regions and runs the diffusion-based
    inpainting model to clean/replace text.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model = None
        self._opt = None

    def _load_config(self, input_dir: str, output_dir: str) -> dict:
        """Load and configure the GSDM YAML config.

        Args:
            input_dir: Directory of input crops.
            output_dir: Where to save inpainted outputs.

        Returns:
            Configured options dict.
        """
        with open(self.config.inpaint_config_path, "r") as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        opt["phase"] = "val"
        opt["SPM"]["resume_state"] = os.path.join(
            self.config.inpaint_ckpt_dir, "spm.pt"
        )
        opt["RM"]["path"]["resume_state"] = os.path.join(
            self.config.inpaint_ckpt_dir, "rm"
        )
        opt["RM"]["gpu_ids"] = [int(x) for x in self.config.inpaint_gpu_ids.split(",")]
        opt["distributed"] = False
        opt["save_sp"] = False

        # Ensure val_dataset paths are set
        if "val_dataset" not in opt:
            opt["val_dataset"] = {}
        opt["val_dataset"]["input_dir"] = input_dir
        opt["val_dataset"]["output_dir"] = output_dir

        return opt

    # ── Inpainting ─────────────────────────────────────────────────

    def inpaint_regions(
        self,
        image: np.ndarray,
        boxes: List[List[int]],
        output_dir: str,
    ) -> str:
        """Inpaint specified regions in the document image.

        Args:
            image: Full document image (BGR).
            boxes: List of ``[x1, y1, x2, y2]`` regions to inpaint.
            output_dir: Directory for intermediate/output files.

        Returns:
            Path to the inpainted output directory.
        """
        if not boxes:
            logger.info("No regions to inpaint.")
            return ""

        # Prepare input crops
        input_dir = os.path.join(output_dir, "inpaint_input")
        result_dir = os.path.join(output_dir, "inpaint_output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        for i, box in enumerate(boxes):
            x1 = max(0, box[0])
            y1 = max(0, box[1])
            x2 = min(image.shape[1], box[2])
            y2 = min(image.shape[0], box[3])

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_path = os.path.join(input_dir, f"region_{i:04d}.png")
            cv2.imwrite(crop_path, crop)

        self._run_model(input_dir, result_dir)
        return result_dir

    def inpaint_word_boxes(
        self,
        doc_image: np.ndarray,
        all_boxes: List[Dict],
        img_w: int,
        img_h: int,
        output_dir: str,
    ) -> str:
        """Inpaint all OCR word regions (for stamp documents).

        Args:
            doc_image: Full document (BGR).
            all_boxes: OCR word boxes with normalised coordinates.
            img_w: Image width.
            img_h: Image height.
            output_dir: Output directory.

        Returns:
            Path to directory with inpainted crops.
        """
        input_dir = os.path.join(output_dir, "input_all_boxes")
        result_dir = os.path.join(output_dir, "output_all_boxes")
        os.makedirs(input_dir, exist_ok=True)

        for i, box in enumerate(all_boxes):
            x1 = max(0, int(box["x1"] * img_w))
            y1 = max(0, int(box["y1"] * img_h))
            x2 = min(img_w, int(box["x2"] * img_w))
            y2 = min(img_h, int(box["y2"] * img_h))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = doc_image[y1:y2, x1:x2]
            text = box.get("text", f"box{i}")
            safe_text = "".join(c if c.isalnum() else "_" for c in text)[:20]
            crop_path = os.path.join(input_dir, f"{i:04d}_{safe_text}.png")
            cv2.imwrite(crop_path, crop)

        self._run_model(input_dir, result_dir)
        return result_dir

    # ── Model lifecycle ──────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load the GSDM model once and keep it on GPU for the entire run."""
        if self._model is not None:
            return

        proj_root = self.config.project_root
        if proj_root not in sys.path:
            sys.path.insert(0, proj_root)

        opt = self._load_config("__placeholder__", "__placeholder__")

        from model.model import GSDM

        logger.info("Loading GSDM inpainting model (persistent) …")
        self._model = GSDM(opt)
        self._model.load_network(logger)
        self._opt = opt
        logger.info("GSDM model loaded and will stay on GPU.")

    # ── Internal ───────────────────────────────────────────────────

    def _run_model(self, input_dir: str, output_dir: str) -> None:
        """Run the persistent GSDM model on all images in *input_dir*."""
        os.makedirs(output_dir, exist_ok=True)

        img_files = [f for f in os.listdir(input_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not img_files:
            logger.warning("No images in %s to inpaint.", input_dir)
            return

        self._ensure_loaded()

        self._opt["val_dataset"]["input_dir"] = input_dir
        self._opt["val_dataset"]["output_dir"] = output_dir

        try:
            import util
            from torch.utils.data import DataLoader

            val_set = util.val_dataset(
                input_dir=input_dir,
                data_shape=self._opt["val_dataset"]["resolution"],
            )
            val_loader = DataLoader(
                val_set, batch_size=1, shuffle=False,
                num_workers=0, pin_memory=True,
            )

            logger.info("Inpainting dataset: %d images", len(val_set))

            for input_data in val_loader:
                output = self._model.inference(input_data["image"])
                spm_mask = output["SPM"]

                mask_img = util.tensor2img(spm_mask, out_type=np.uint8, min_max=(-1, 1))
                inverted_mask = 255 - mask_img
                util.save_img(
                    inverted_mask,
                    os.path.join(output_dir, input_data["name"][0]),
                )

                input_image = cv2.imread(
                    os.path.join(input_dir, input_data["name"][0])
                )
                if input_image is not None:
                    reshape_dim = (input_image.shape[1], input_image.shape[0])
                    output_image = cv2.imread(
                        os.path.join(output_dir, input_data["name"][0])
                    )
                    if output_image is not None:
                        reshaped = cv2.resize(output_image, reshape_dim)
                        util.save_img(
                            reshaped,
                            os.path.join(output_dir, input_data["name"][0]),
                        )

                logger.debug("Inpainted: %s", input_data["name"][0])

            logger.info("Inpainting complete: %s → %s", input_dir, output_dir)

        except Exception as exc:
            logger.error("Inpainting failed: %s", exc, exc_info=True)
