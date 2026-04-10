"""
DocRevive Configuration Module.

Central configuration dataclass holding all hyperparameters, model paths,
and processing settings for the document restoration pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Config:
    """Configuration for the DocRevive pipeline.

    All model paths default to the existing weights/checkpoints in the project.
    Large models (Qwen) are stored on /data/ to avoid filling the root partition.
    """

    # ── Project Paths ──────────────────────────────────────────────────
    project_root: str = os.path.dirname(os.path.abspath(__file__))
    output_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs"
    )
    ocr_cache_dir: Optional[str] = None       # precomputed OCR JSON cache
    yolo_cache_dir: Optional[str] = None      # precomputed YOLO JSON cache
    corrected_cache_dir: Optional[str] = None # precomputed skew-corrected images

    # ── Detection (doctr fast_base) ────────────────────────────────────
    det_arch: str = "fast_base"
    det_device: str = "cuda"
    det_pretrained: bool = True

    # ── Recognition (doctr parseq) ─────────────────────────────────────
    reco_arch: str = "parseq"
    reco_device: str = "cuda"
    reco_pretrained: bool = True

    # ── Blank Extraction (adaptive signal fusion) ──────────────────────
    line_grouping_y_tolerance: float = 0.005  # normalised y-center tolerance
    sliding_window_width: int = 5  # pixel window for column projection
    gap_min_width: int = 15  # minimum pixel width for a valid blank gap
    blank_binarize_method: str = "otsu"  # "otsu" or "fixed"
    blank_binarize_threshold: int = 128  # only used when method == "fixed"
    blank_smooth_sigma: float = 1.5
    blank_peak_type: str = "both"  # "black", "white", or "both"
    # Statistical thresholds for gap outlier detection
    gap_outlier_sigma: float = 2.0  # std deviations above mean gap width

    # ── Offline LLM (Qwen via vLLM) ───────────────────────────────────
    # Large base model (~67 GB) — stored outside the project on /data/qwen
    qwen_model_path: str = "/data/qwen"
    vllm_tensor_parallel_size: int = 4
    vllm_gpu_memory_utilization: float = 0.85
    vllm_max_model_len: int = 4096
    llm_max_tokens: int = 512
    llm_temperature: float = 0.2
    llm_top_p: float = 0.9
    llm_mode: str = "local"  # "local" | "http" | "roberta" | "qwen3" | "dummy"
    llm_server_url: str = "http://127.0.0.1:8199"

    # ── RoBERTa (lightweight fill-mask predictor) ──────────────────────
    # Must have tokenizer files (vocab.json, merges.txt). Use "roberta-large" or
    # finetune output "final" dir. Checkpoint dirs (checkpoint-*) lack tokenizer.
    # roberta_model_name: str = "roberta-large"
    roberta_model_name: str = "weights/roberta"

    # ── Qwen3 (fine-tuned causal LM, same task as RoBERTa) ─────────────────────
    qwen3_model_name: str = "weights/qwen3"

    # ── Generic device fallback (used by YOLO and any model without explicit device) ──
    device: str = "cuda"

    # ── Occlusion Detection (finetuned YOLOv9c) ──────────────────────
    yolo_model_path: str = "weights/yolo_occlusion.pt"
    yolo_imgsz: int = 768
    yolo_conf_thresh: float = 0.25

    # ── Text Editing (GaMuSA) ──────────────────────────────────────────
    text_editor_ckpt_path: str = "weights/model.pth"
    text_editor_config_path: str = "configs/inference.yaml"
    text_editor_device: str = "cuda"
    text_editor_starting_layer: int = 10
    text_editor_num_inference_steps: int = 50
    text_editor_guidance_scale: float = 2.0
    text_editor_benchmark: bool = True
    text_editor_style_patch_size: Tuple[int, int] = (256, 256)
    text_editor_monitor_cfg: dict = field(default_factory=lambda: {
        "max_length": 25,
        "loss_weight": 1.0,
        "attention": "position",
        "backbone": "transformer",
        "backbone_ln": 3,
        "checkpoint": "weights/vision_model.pth",
        "charset_path": "src/module/abinet/data/charset_36.txt",
    })

    # ── Inpainting (GSDM) ─────────────────────────────────────────────
    inpaint_config_path: str = "config/inference.yaml"
    inpaint_ckpt_dir: str = "weights/checkpoint"
    inpaint_device: str = "cuda"
    inpaint_gpu_ids: str = "7"

    # ── Post-Processing ────────────────────────────────────────────────
    patch_border_size: int = 2
    feather_kernel_size: int = 15
    inter_line_gap_threshold: int = 5  # px gap between lines to skip fill

    # ── Batch Processing ───────────────────────────────────────────────
    batch_size: int = 1  # images processed in parallel
    num_workers: int = 4  # dataloader workers
    seed: int = 42

    # ── Debug ──────────────────────────────────────────────────────────
    debug: bool = False
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Resolve relative paths to absolute using project_root."""
        relative_fields = [
            "text_editor_ckpt_path",
            "text_editor_config_path",
            "inpaint_config_path",
            "inpaint_ckpt_dir",
            "yolo_model_path",
            "roberta_model_name",
            "qwen3_model_name",
        ]
        for field_name in relative_fields:
            val = getattr(self, field_name)
            if val and not os.path.isabs(val):
                setattr(self, field_name, os.path.join(self.project_root, val))

        os.makedirs(self.output_dir, exist_ok=True)

        # Auto-configure inpaint_gpu_ids to use all GPUs visible in
        # CUDA_VISIBLE_DEVICES when the caller leaves it at the default "0".
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cvd and cvd.lower() not in ("nodevfile", "none", "-1"):
            n_visible = len(cvd.split(","))
            if n_visible > 1 and self.inpaint_gpu_ids == "0":
                # Remap to 0-indexed virtual IDs within visible set
                self.inpaint_gpu_ids = ",".join(str(i) for i in range(n_visible))
