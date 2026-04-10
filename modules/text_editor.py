"""
Diffusion-Based Text Editing module.

Wraps the existing GaMuSA (Glyph-aware Mutual Self-Attention) text
editing model.  Takes a style reference patch and a target text string,
then generates an image of the target text in the same visual style.

Multi-GPU strategy: GaMuSA uses stateful UNet attention-controller hooks
that are NOT thread-safe across DataParallel splits.  Instead we use
*process-level parallelism* — each worker process owns one GPU and edits
an independent subset of images, then results are merged back.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from config import Config

logger = logging.getLogger(__name__)


# ── Persistent pool worker functions ──────────────────────────────────
#
# _pool_initializer runs once per worker process when the pool is first
# created.  It loads the GaMuSA model and stashes it in module globals.
# _pool_edit_task is submitted for each batch of items and reuses the
# already-loaded model — no repeated loading.

_wk_pipeline = None
_wk_config: dict = {}


def _pool_initializer(
    gpu_queue,
    proj_root: str,
    ckpt_path: str,
    config_path: str,
    monitor_cfg: str,
    seed: int,
    starting_layer: int,
    ddim_steps: int,
    scale: float,
) -> None:
    """Called exactly once per worker process to load GaMuSA."""
    global _wk_pipeline, _wk_config

    gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    import transformers as _tf
    if not hasattr(_tf, "CLIPFeatureExtractor"):
        _tf.CLIPFeatureExtractor = getattr(
            _tf, "CLIPImageProcessor", type("_Stub", (), {}))

    from pytorch_lightning import seed_everything
    from src.MuSA.GaMuSA import GaMuSA
    from utils import create_model, load_state_dict

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(config_path).to(device)
    model.load_state_dict(load_state_dict(ckpt_path), strict=False)
    model.eval()
    _wk_pipeline = GaMuSA(model, monitor_cfg)
    _wk_config = dict(
        device=device, seed=seed,
        starting_layer=starting_layer, ddim_steps=ddim_steps, scale=scale,
    )
    seed_everything(seed)
    logger.info("[GPU %s] GaMuSA persistent worker ready.", gpu_id)


def _pool_edit_task(
    style_dir: str,
    output_dir: str,
    items: List[Tuple[str, str, str]],
) -> List[Tuple[str, str]]:
    """Edit a chunk of images using the pre-loaded model."""
    import torch
    from torchvision import transforms as T
    from src.MuSA.GaMuSA_app import text_editing

    cfg = _wk_config
    device = cfg["device"]

    def _load_tensor(path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = T.ToTensor()(T.Resize((256, 256))(img))
        return img.unsqueeze(0).to(device)

    results: List[Tuple[str, str]] = []
    for img_name, style_text, target_text in items:
        path = os.path.join(style_dir, img_name)
        if not os.path.exists(path):
            continue
        pil_img = Image.open(path)
        w, h = pil_img.size
        src  = _load_tensor(path)
        styl = _load_tensor(path)
        try:
            _, edited = text_editing(
                _wk_pipeline, src, styl,
                style_text, target_text,
                starting_layer=cfg["starting_layer"],
                ddim_steps=cfg["ddim_steps"],
                scale=cfg["scale"],
            )
            out_pil = Image.fromarray(
                (edited * 255).astype(np.uint8)
            ).resize((w, h))
            out_path = os.path.join(output_dir, img_name)
            out_pil.save(out_path)
            results.append((img_name, out_path))
        except Exception as exc:
            import traceback
            traceback.print_exc()
    return results


class TextEditor:
    """Edit text in document images using the GaMuSA diffusion model.

    Loads the model once for single-GPU usage, or spawns per-GPU worker
    processes for multi-GPU usage (``config.text_editor_gpu_ids``).
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model    = None
        self._pipeline = None
        self._pool     = None       # persistent ProcessPoolExecutor for multi-GPU
        self._pool_gpu_ids = None

    def _ensure_loaded(self) -> None:
        """Lazy-load GaMuSA model on the primary GPU (single-GPU path)."""
        if self._model is not None:
            return

        logger.info("Loading GaMuSA text editing model from %s …",
                    self.config.text_editor_ckpt_path)

        proj_root = self.config.project_root
        if proj_root not in sys.path:
            sys.path.insert(0, proj_root)

        import torch
        from pytorch_lightning import seed_everything
        from src.MuSA.GaMuSA import GaMuSA
        from utils import create_model, load_state_dict

        _te_device = (
            self.config.text_editor_device
            if torch.cuda.is_available() else "cpu"
        )
        model = create_model(self.config.text_editor_config_path).to(_te_device)
        model.load_state_dict(
            load_state_dict(self.config.text_editor_ckpt_path), strict=False
        )
        model.eval()
        self._model    = model
        self._pipeline = GaMuSA(model, self.config.text_editor_monitor_cfg)
        seed_everything(self.config.seed)
        logger.info("GaMuSA model loaded successfully.")

    # ── Style Box Selection ────────────────────────────────────────────

    @staticmethod
    def select_style_box(
        all_boxes: List[Dict],
        target_text: str,
    ) -> Optional[Dict]:
        """Select an OCR box whose text length best matches *target_text*."""
        target_len = len(target_text.replace(" ", ""))
        if target_len == 0:
            return None
        for box in all_boxes:
            if box.get("text", "") == target_text:
                return box
        best, best_diff = None, float("inf")
        for box in all_boxes:
            box_len = len(box.get("text", "").replace(" ", ""))
            diff = abs(box_len - target_len)
            if diff < best_diff:
                best_diff = diff
                best = box
        return best

    # ── Dataset Preparation ────────────────────────────────────────────

    def build_editing_dataset(
        self,
        doc_image: Image.Image,
        blank_bboxes: List[Dict],
        all_boxes: List[Dict],
        predicted_texts: Dict[int, str],
        img_size: Tuple[int, int],
        doc_name: str,
        dataset_dir: str,
    ) -> List[Dict]:
        """Build the ``i_s/`` and ``i_t.txt`` dataset for GaMuSA."""
        os.makedirs(os.path.join(dataset_dir, "i_s"), exist_ok=True)
        img_w, img_h = img_size

        i_s_lines: List[str] = []
        i_t_lines: List[str] = []

        for blank in blank_bboxes:
            blank_id    = blank.get("blank_id", blank.get("id", 0))
            target_text = predicted_texts.get(blank_id, "")
            if not target_text:
                continue

            words = target_text.split()
            if not words:
                continue

            pre_word = blank.get("pre_word", {})
            post_word = blank.get("post_word", {})
            abs_bbox  = blank.get("blank_abs_bbox")

            if abs_bbox:
                bx1, by1, bx2, by2 = abs_bbox
            elif pre_word and post_word:
                bx1 = int(pre_word.get("x2", 0) * img_w)
                by1 = int(min(pre_word.get("y1", 0), post_word.get("y1", 0)) * img_h)
                bx2 = int(post_word.get("x1", 1) * img_w)
                by2 = int(max(pre_word.get("y2", 1), post_word.get("y2", 1)) * img_h)
            elif pre_word:
                bx1 = int(pre_word.get("x2", 0) * img_w)
                by1 = int(pre_word.get("y1", 0) * img_h)
                bx2 = bx1 + 100
                by2 = int(pre_word.get("y2", 1) * img_h)
            elif post_word:
                bx2 = int(post_word.get("x1", 1) * img_w)
                by1 = int(post_word.get("y1", 0) * img_h)
                bx1 = max(0, bx2 - 100)
                by2 = int(post_word.get("y2", 1) * img_h)
            elif "bbox" in blank:
                bx1, by1, bx2, by2 = blank["bbox"]
            else:
                continue

            blank_width  = bx2 - bx1
            blank_height = by2 - by1
            if blank_width <= 0 or blank_height <= 0:
                continue

            gap_width      = min(10, blank_width // (len(words) + 1))
            total_char_len = sum(len(w) for w in words)
            if total_char_len == 0:
                continue

            sub_boxes: List[Tuple] = []
            x_cursor = bx1
            for w_idx, word in enumerate(words):
                w_frac  = len(word) / total_char_len
                w_width = int(w_frac * (blank_width - gap_width * (len(words) - 1)))
                w_width = max(w_width, 10)
                sub_x2  = min(x_cursor + w_width, bx2)
                sub_boxes.append((x_cursor, by1, sub_x2, by2, word))
                x_cursor = sub_x2 + gap_width

            blank["sub_boxes"] = sub_boxes

            for s_idx, (sx1, sy1, sx2, sy2, word_text) in enumerate(sub_boxes):
                style_box = self.select_style_box(all_boxes, word_text)
                if style_box is None:
                    continue
                sx1_ = int(style_box.get("x1", 0) * img_w)
                sy1_ = int(style_box.get("y1", 0) * img_h)
                sx2_ = int(style_box.get("x2", 1) * img_w)
                sy2_ = int(style_box.get("y2", 1) * img_h)
                style_crop = doc_image.crop((sx1_, sy1_, sx2_, sy2_))
                img_name   = f"{doc_name}_b{blank_id}_w{s_idx}.png"
                style_crop.save(os.path.join(dataset_dir, "i_s", img_name))
                style_text = style_box.get("text", "")
                i_s_lines.append(f"{img_name} {style_text}")
                i_t_lines.append(f"{img_name} {word_text}")

        with open(os.path.join(dataset_dir, "i_s.txt"), "w") as f:
            f.write("\n".join(i_s_lines) + "\n")
        with open(os.path.join(dataset_dir, "i_t.txt"), "w") as f:
            f.write("\n".join(i_t_lines) + "\n")

        logger.info("Built editing dataset: %d style images, %d target texts.",
                    len(i_s_lines), len(i_t_lines))
        return blank_bboxes

    # ── Image Loading ──────────────────────────────────────────────────

    @staticmethod
    def _load_image_tensor(image_path: str, height: int = 256, width: int = 256):
        import torch
        from torchvision import transforms as T
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        img = Image.open(image_path).convert("RGB")
        img = T.ToTensor()(T.Resize((height, width))(img))
        return img.unsqueeze(0).to(device)

    # ── GPU list ──────────────────────────────────────────────────────

    def _gpu_ids(self) -> List[int]:
        """Parse text_editor_gpu_ids or fall back to inpaint_gpu_ids."""
        raw = getattr(self.config, "text_editor_gpu_ids", None) or \
              getattr(self.config, "inpaint_gpu_ids", "0")
        try:
            return [int(g.strip()) for g in raw.split(",") if g.strip()]
        except ValueError:
            return [0]

    # ── Editing ───────────────────────────────────────────────────────

    def edit(self, dataset_dir: str, output_dir: str) -> None:
        """Run GaMuSA text editing, distributing work across all GPUs.

        Strategy:
          - If only 1 GPU (or fallback): single-process sequential loop.
          - If multiple GPUs: spawn N worker processes (one per GPU),
            each loads its own model copy and edits its image subset.

        Args:
            dataset_dir: Directory with ``i_s/``, ``i_s.txt``, ``i_t.txt``.
            output_dir: Where to save edited images.
        """
        style_dir = os.path.join(dataset_dir, "i_s")
        if not os.path.exists(style_dir):
            logger.warning("No style directory found at %s", style_dir)
            return

        style_images = [f for f in os.listdir(style_dir)
                        if f.lower().endswith(".png")]
        if not style_images:
            logger.warning("No style images found.")
            return

        style_dict  = self._read_text_file(os.path.join(dataset_dir, "i_s.txt"))
        target_dict = self._read_text_file(os.path.join(dataset_dir, "i_t.txt"))
        os.makedirs(output_dir, exist_ok=True)

        items = [
            (img_name,
             style_dict.get(img_name, ""),
             target_dict.get(img_name, style_dict.get(img_name, "")))
            for img_name in style_images
        ]

        gpu_ids = self._gpu_ids()
        n_gpus  = len(gpu_ids)

        if n_gpus <= 1 or len(items) <= 1:
            # ── Single-GPU path ────────────────────────────────────────
            self._edit_single_gpu(style_dir, output_dir, items,
                                  style_dict, target_dict)
        else:
            # ── Multi-GPU path: one process per GPU ────────────────────
            logger.info("Text editing on %d GPUs (%s), %d images total.",
                        n_gpus, ",".join(str(g) for g in gpu_ids), len(items))
            self._edit_multi_gpu(style_dir, output_dir, items, gpu_ids)

        logger.info("Text editing complete for %d images.", len(style_images))

    def _edit_single_gpu(
        self,
        style_dir: str,
        output_dir: str,
        items: List[Tuple[str, str, str]],
        style_dict: Dict[str, str],
        target_dict: Dict[str, str],
    ) -> None:
        """Sequential single-GPU editing loop."""
        self._ensure_loaded()

        from src.MuSA.GaMuSA_app import text_editing

        for img_name, style_text, target_text in items:
            path = os.path.join(style_dir, img_name)
            pil_img = Image.open(path)
            w, h = pil_img.size
            source_image = self._load_image_tensor(path, 256, 256)
            style_image  = self._load_image_tensor(path, 256, 256)
            try:
                _, edited = text_editing(
                    self._pipeline,
                    source_image, style_image,
                    style_text, target_text,
                    starting_layer=self.config.text_editor_starting_layer,
                    ddim_steps=self.config.text_editor_num_inference_steps,
                    scale=self.config.text_editor_guidance_scale,
                )
                edited_pil = Image.fromarray(
                    (edited * 255).astype(np.uint8)
                ).resize((w, h))
                edited_pil.save(os.path.join(output_dir, img_name))
                logger.debug("Edited: %s → %s", style_text, target_text)
            except Exception as exc:
                logger.error("Failed to edit %s: %s", img_name, exc)

    def _ensure_pool(self, gpu_ids: List[int]) -> None:
        """Create or reuse a persistent ProcessPoolExecutor for multi-GPU editing."""
        if self._pool is not None and self._pool_gpu_ids == gpu_ids:
            return

        if self._pool is not None:
            self._pool.shutdown(wait=False)

        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        ctx = mp.get_context("spawn")
        gpu_queue = ctx.Queue()
        for gid in gpu_ids:
            gpu_queue.put(gid)

        self._pool = ProcessPoolExecutor(
            max_workers=len(gpu_ids),
            mp_context=ctx,
            initializer=_pool_initializer,
            initargs=(
                gpu_queue,
                self.config.project_root,
                self.config.text_editor_ckpt_path,
                self.config.text_editor_config_path,
                self.config.text_editor_monitor_cfg,
                self.config.seed,
                self.config.text_editor_starting_layer,
                self.config.text_editor_num_inference_steps,
                self.config.text_editor_guidance_scale,
            ),
        )
        self._pool_gpu_ids = list(gpu_ids)
        logger.info("Created persistent GaMuSA pool with %d workers on GPUs %s.",
                     len(gpu_ids), gpu_ids)

    def _edit_multi_gpu(
        self,
        style_dir: str,
        output_dir: str,
        items: List[Tuple[str, str, str]],
        gpu_ids: List[int],
    ) -> None:
        """Distribute editing across persistent GPU workers."""
        import math
        from concurrent.futures import as_completed

        try:
            self._ensure_pool(gpu_ids)
        except Exception as exc:
            logger.error("Pool creation failed (%s); falling back to single-GPU.", exc)
            style_dict  = self._read_text_file(
                os.path.join(os.path.dirname(style_dir), "i_s.txt"))
            target_dict = self._read_text_file(
                os.path.join(os.path.dirname(style_dir), "i_t.txt"))
            self._edit_single_gpu(style_dir, output_dir, items,
                                  style_dict, target_dict)
            return

        n_gpus   = len(gpu_ids)
        chunk_sz = math.ceil(len(items) / n_gpus)
        chunks   = [items[i * chunk_sz : (i + 1) * chunk_sz]
                    for i in range(n_gpus)
                    if items[i * chunk_sz : (i + 1) * chunk_sz]]

        completed = 0
        try:
            futs = {
                self._pool.submit(_pool_edit_task, style_dir, output_dir, chunk): i
                for i, chunk in enumerate(chunks)
            }
            for fut in as_completed(futs):
                worker_idx = futs[fut]
                try:
                    results = fut.result()
                    completed += len(results)
                    logger.info("Worker %d done: %d images.", worker_idx, len(results))
                except Exception as exc:
                    logger.error("Worker %d failed: %s", worker_idx, exc)
        except Exception as pool_exc:
            logger.error("Multi-GPU edit failed (%s); falling back to single-GPU.", pool_exc)
            self._pool.shutdown(wait=False)
            self._pool = None
            style_dict  = self._read_text_file(
                os.path.join(os.path.dirname(style_dir), "i_s.txt"))
            target_dict = self._read_text_file(
                os.path.join(os.path.dirname(style_dir), "i_t.txt"))
            self._edit_single_gpu(style_dir, output_dir, items,
                                  style_dict, target_dict)
            return

        logger.info("Multi-GPU editing: %d/%d images completed.", completed, len(items))

    def shutdown(self) -> None:
        """Shut down the persistent multi-GPU pool if active."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
            logger.info("GaMuSA persistent pool shut down.")

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _read_text_file(path: str) -> Dict[str, str]:
        """Read a ``<name> <text>`` mapping file."""
        mapping: Dict[str, str] = {}
        if not os.path.exists(path):
            return mapping
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    mapping[parts[0]] = parts[1].strip()
        return mapping

