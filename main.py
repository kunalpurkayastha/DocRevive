#!/usr/bin/env python3
"""
DocRevive — Document Restoration System.

Entry point for single-image and batch processing of occluded documents.

Usage:
    # Single image (eager-load, no cold start)
    python main.py --input /path/to/image.jpg --output_dir ./outputs

    # Batch directory — auto-uses all GPUs with persistent workers
    python main.py --input_dir /path/to/images/ --output_dir ./outputs

    # Explicit worker count
    python main.py --input_dir /path/to/images/ --batch_workers 8

    # Debug mode
    python main.py --input /path/to/image.jpg --debug
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

warnings.filterwarnings("ignore")


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark    = False
    torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DocRevive — Document Restoration System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input/Output ───────────────────────────────────────────────
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to a single input document image.",
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Directory of input images (batch mode).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max images to process from --input_dir (0 = all).",
    )
    parser.add_argument(
        "--per_occ", type=int, default=0,
        help=(
            "Sample N images per occlusion type (from filename prefix). "
            "Requires --input_dir. Occlusion types: Black_Ink, White_Burnt, "
            "White_Whitener, Through_Dust, Through_Stamp, Sim. "
            "0 = disabled (use --limit or all images)."
        ),
    )
    parser.add_argument(
        "--ocr_cache_dir", type=str, default=None,
        help="Directory with precomputed OCR JSON files (from scripts/precompute_ocr.py).",
    )
    parser.add_argument(
        "--yolo_cache_dir", type=str, default=None,
        help="Directory with precomputed YOLO JSON files (from scripts/precompute_yolo.py).",
    )
    parser.add_argument(
        "--corrected_cache_dir", type=str, default=None,
        help="Directory with precomputed skew-corrected images (from scripts/precompute_all.py).",
    )

    # ── Model Paths ────────────────────────────────────────────────
    parser.add_argument(
        "--ckpt_path", type=str, default="weights/model.pth",
        help="Path to GaMuSA text editing checkpoint.",
    )
    parser.add_argument(
        "--qwen_model", type=str, default="/data/qwen",
        help="Path to Qwen model for offline LLM prediction.",
    )
    parser.add_argument(
        "--qwen3_model", type=str, default=None,
        help="Path to fine-tuned Qwen3 checkpoint (when --llm_mode qwen3).",
    )
    parser.add_argument(
        "--inpaint_ckpt_dir", type=str, default="weights/checkpoint",
        help="Directory with GSDM inpainting checkpoints.",
    )

    # ── GPU / Parallelism ─────────────────────────────────────────
    parser.add_argument(
        "--tensor_parallel", type=int, default=4,
        help="vLLM tensor parallelism size (must divide 8192 — use 1/2/4/8).",
    )
    parser.add_argument(
        "--vllm_gpus", type=str, default="0,1,2,3",
        help="GPU IDs for the shared vLLM server (comma-separated).",
    )
    parser.add_argument(
        "--gpu_ids", type=str, default="0,1,2,3,4,5,6,7",
        help="GPU IDs for inpainting/text-editing models (comma-separated).",
    )
    parser.add_argument(
        "--batch_workers", type=int, default=0,
        help=(
            "Number of persistent pipeline workers for batch processing. "
            "0 = auto (one per GPU in --gpu_ids). "
            "Each worker occupies one GPU and loads all models once."
        ),
    )

    # ── Processing ─────────────────────────────────────────────────
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )

    # ── Text Editing ───────────────────────────────────────────────
    parser.add_argument(
        "--starting_layer", type=int, default=10,
        help="Starting layer for GaMuSA inference.",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50,
        help="DDIM steps for text editing.",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=2.0,
        help="Guidance scale for text editing.",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Benchmark mode (simplified outputs).",
    )

    # ── Debug ──────────────────────────────────────────────────────
    parser.add_argument(
        "--no_llm", action="store_true",
        help=(
            "Use RoBERTa or Qwen3 fill predictor instead of the large Qwen LLM. "
            "No vLLM server is started — all GPUs are free for YOLO / GSDM / GaMuSA."
        ),
    )
    parser.add_argument(
        "--llm_mode", type=str, default=None,
        choices=["local", "http", "roberta", "qwen3", "dummy"],
        help="Override LLM prediction mode. Takes precedence over --no_llm.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save per-stage debug visualisations.",
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    # ── Resume ─────────────────────────────────────────────────────
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip images whose _restored.png already exists in output_dir.",
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="Force re-processing of all images (ignore existing outputs).",
    )

    # ── Legacy / compatibility ──────────────────────────────────────
    parser.add_argument("--num_workers", type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--gpu_start",   type=int, default=0,
                        help=argparse.SUPPRESS)

    return parser.parse_args()


def _occlusion_type(path: str) -> str:
    """Extract occlusion type from filename prefix (before first __)."""
    name = os.path.basename(path)
    if "__" in name:
        return name.split("__", 1)[0]
    return "unknown"


def _is_done(output_dir: str, image_path: str) -> bool:
    """Check if a restored output already exists for this image."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    restored = os.path.join(output_dir, stem, f"{stem}_restored.png")
    return os.path.exists(restored)


def _log_type_table(logger: logging.Logger, paths: list[str],
                    label: str) -> dict[str, int]:
    """Log a per-occlusion-type count table and return the counts."""
    counts: dict[str, int] = defaultdict(int)
    for p in paths:
        counts[_occlusion_type(p)] += 1
    logger.info("─── %s (%d images) ───", label, len(paths))
    for t in sorted(counts):
        logger.info("  %-20s %d", t, counts[t])
    return dict(counts)


MANIFEST_NAME = "manifest.json"
RESULTS_NAME  = "results.jsonl"


def _load_manifest(output_dir: str) -> dict:
    path = os.path.join(output_dir, MANIFEST_NAME)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_manifest(output_dir: str, manifest: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, MANIFEST_NAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def _load_results_jsonl(output_dir: str) -> list[dict]:
    """Load all previous result records from results.jsonl."""
    path = os.path.join(output_dir, RESULTS_NAME)
    records = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def _build_results_callback(output_dir: str, manifest: dict, logger: logging.Logger):
    """Return an on_result callback that writes JSONL incrementally and updates manifest."""
    jsonl_path = os.path.join(output_dir, RESULTS_NAME)
    jsonl_file = open(jsonl_path, "a")
    _counter = {"n": 0}

    def on_result(image_path: str, output_path: str | None, error: str | None):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        occ_type = _occlusion_type(image_path)
        ts = datetime.now().isoformat()
        status = "success" if output_path else "failed"

        record = {
            "stem": stem,
            "image": image_path,
            "occlusion_type": occ_type,
            "status": status,
            "output": output_path,
            "timestamp": ts,
        }
        if error:
            record["error"] = str(error)[:500]

        jsonl_file.write(json.dumps(record) + "\n")
        jsonl_file.flush()

        if status == "success":
            manifest["completed"][stem] = {
                "path": image_path, "output": output_path, "timestamp": ts,
            }
        else:
            manifest["failed"][stem] = {
                "path": image_path, "error": str(error or "")[:500], "timestamp": ts,
            }

        _counter["n"] += 1
        if _counter["n"] % 10 == 0:
            _save_manifest(output_dir, manifest)

    def close():
        _save_manifest(output_dir, manifest)
        jsonl_file.close()

    on_result.close = close
    return on_result


def _print_results_table(output_dir: str, logger: logging.Logger) -> None:
    """Read results.jsonl and print a per-occlusion-type summary table."""
    records = _load_results_jsonl(output_dir)
    if not records:
        logger.info("No results recorded yet.")
        return

    type_stats: dict[str, dict] = defaultdict(lambda: {"success": 0, "failed": 0})
    for r in records:
        occ = r.get("occlusion_type", "unknown")
        st  = r.get("status", "failed")
        type_stats[occ][st] += 1

    total_ok = sum(v["success"] for v in type_stats.values())
    total_fail = sum(v["failed"] for v in type_stats.values())

    logger.info("─── Results by occlusion type ───")
    logger.info("  %-20s %8s %8s %8s", "Type", "Success", "Failed", "Total")
    logger.info("  %s", "─" * 50)
    for occ in sorted(type_stats):
        s = type_stats[occ]
        logger.info("  %-20s %8d %8d %8d", occ, s["success"], s["failed"],
                     s["success"] + s["failed"])
    logger.info("  %s", "─" * 50)
    logger.info("  %-20s %8d %8d %8d", "TOTAL", total_ok, total_fail,
                 total_ok + total_fail)


def main() -> None:
    """Main entry point."""
    args = parse_args()
    do_resume = args.resume and not args.no_resume

    # ── Logging — console + file ──────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    log_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, args.log_level))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_fmt)
    root_logger.addHandler(console_handler)

    log_file = os.path.join(args.output_dir, "run.log")
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(log_fmt)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger("docrevive")
    logger.info("=" * 60)
    logger.info("DocRevive — run started at %s", datetime.now().isoformat())
    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Resume mode: %s", do_resume)
    logger.info("=" * 60)

    # ── Seed ───────────────────────────────────────────────────────
    set_seed(args.seed)

    # ── Build Config kwargs ────────────────────────────────────────
    config_kwargs = dict(
        output_dir                    = args.output_dir,
        text_editor_ckpt_path         = args.ckpt_path,
        qwen_model_path               = args.qwen_model,
        inpaint_ckpt_dir              = args.inpaint_ckpt_dir,
        inpaint_gpu_ids               = args.gpu_ids,
        vllm_tensor_parallel_size     = args.tensor_parallel,
        text_editor_starting_layer    = args.starting_layer,
        text_editor_num_inference_steps = args.num_inference_steps,
        text_editor_guidance_scale    = args.guidance_scale,
        text_editor_benchmark         = args.benchmark,
        debug                         = args.debug,
        log_level                     = args.log_level,
        seed                          = args.seed,
        llm_mode                      = args.llm_mode or ("roberta" if args.no_llm else "local"),
        ocr_cache_dir                 = args.ocr_cache_dir,
        yolo_cache_dir                = args.yolo_cache_dir,
        corrected_cache_dir           = args.corrected_cache_dir,
    )
    if args.qwen3_model:
        config_kwargs["qwen3_model_name"] = args.qwen3_model

    # ── Collect Input Images ───────────────────────────────────────
    image_paths: list[str] = []

    if args.input:
        if not os.path.exists(args.input):
            logger.error("Input file not found: %s", args.input)
            sys.exit(1)
        image_paths.append(args.input)

    elif args.input_dir:
        if not os.path.isdir(args.input_dir):
            logger.error("Input directory not found: %s", args.input_dir)
            sys.exit(1)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"):
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))
        image_paths = sorted(set(image_paths))

        if args.per_occ > 0:
            by_type: dict[str, list[str]] = defaultdict(list)
            for p in image_paths:
                by_type[_occlusion_type(p)].append(p)
            sampled: list[str] = []
            for occ_type in sorted(by_type.keys()):
                pool = by_type[occ_type]
                n = min(args.per_occ, len(pool))
                sampled.extend(random.sample(pool, n))
            image_paths = sampled
            logger.info(
                "Per-occ sampling: %d images per type → %d total.",
                args.per_occ, len(image_paths),
            )
        elif args.limit > 0:
            image_paths = image_paths[: args.limit]
            logger.info("Limiting to first %d image(s).", args.limit)

    else:
        logger.error("Must specify --input or --input_dir.")
        sys.exit(1)

    if not image_paths:
        logger.error("No images found.")
        sys.exit(1)

    all_selected = list(image_paths)
    _log_type_table(logger, all_selected, "Selected images")

    # ── Manifest: persist the full selected set ───────────────────
    manifest = _load_manifest(args.output_dir)
    if "selected_images" not in manifest:
        manifest["selected_images"] = [os.path.abspath(p) for p in all_selected]
    manifest.setdefault("completed", {})
    manifest.setdefault("failed", {})
    manifest["last_run_start"] = datetime.now().isoformat()
    manifest["args"] = {k: v for k, v in vars(args).items()
                        if not k.startswith("_")}
    _save_manifest(args.output_dir, manifest)
    logger.info("Manifest saved: %s", os.path.join(args.output_dir, MANIFEST_NAME))

    # ── Resume: skip already-completed images ─────────────────────
    if do_resume:
        already_done = [p for p in image_paths if _is_done(args.output_dir, p)]
        if already_done:
            done_set = set(already_done)
            image_paths = [p for p in image_paths if p not in done_set]
            logger.info("Resume: %d already done, %d remaining.",
                        len(already_done), len(image_paths))
            _log_type_table(logger, already_done, "Already completed")
            _log_type_table(logger, image_paths, "Remaining to process")
        else:
            logger.info("Resume: no previously completed images found.")
    else:
        logger.info("Resume disabled — processing all %d images.", len(image_paths))

    if not image_paths:
        logger.info("All images already processed. Nothing to do.")
        logger.info("Output dir: %s", args.output_dir)
        return

    logger.info("Processing %d image(s).", len(image_paths))

    # ── Incremental results callback ──────────────────────────────
    on_result = _build_results_callback(args.output_dir, manifest, logger)

    # ── Resolve GPU assignments ────────────────────────────────────
    all_gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(",") if g.strip()]
    n_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if n_visible > 0 and cuda_visible.strip():
        physical = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
        if physical and len(all_gpu_ids) > len(physical):
            all_gpu_ids = physical
    elif n_visible > 0 and len(all_gpu_ids) > n_visible:
        all_gpu_ids = list(range(n_visible))
    n_batch_workers = args.batch_workers or args.num_workers or len(all_gpu_ids)
    worker_gpu_ids = all_gpu_ids[:n_batch_workers]

    vllm_gpu_ids     = args.vllm_gpus
    vllm_tensor_par  = args.tensor_parallel

    # ── Dispatch ────────────────────────────────────────────────────
    run_start = time.time()
    use_parallel = len(image_paths) > 1 and len(worker_gpu_ids) > 0

    try:
        if use_parallel:
            from parallel import ParallelRunner

            logger.info(
                "Batch mode: %d images, %d workers on GPUs %s, "
                "vLLM on GPUs %s (tensor_parallel=%d).",
                len(image_paths), len(worker_gpu_ids), worker_gpu_ids,
                vllm_gpu_ids, vllm_tensor_par,
            )

            runner = ParallelRunner(
                worker_gpu_ids       = worker_gpu_ids,
                vllm_gpu_ids         = vllm_gpu_ids,
                vllm_tensor_parallel = vllm_tensor_par,
                config_kwargs        = config_kwargs,
            )
            runner.run(image_paths, on_result=on_result)

        else:
            from config import Config
            from pipeline import DocRevivePipeline

            config   = Config(**config_kwargs)
            pipeline = DocRevivePipeline(config)
            pipeline.preload_all()

            if len(image_paths) == 1:
                try:
                    pipeline.restore(image_paths[0])
                    stem = os.path.splitext(os.path.basename(image_paths[0]))[0]
                    out = os.path.join(args.output_dir, stem, f"{stem}_restored.png")
                    on_result(image_paths[0], out if os.path.exists(out) else None, None)
                except Exception as exc:
                    on_result(image_paths[0], None, str(exc))
            else:
                pipeline.restore_batch(image_paths, on_result=on_result)

    except KeyboardInterrupt:
        logger.warning("Interrupted! Saving progress …")
    finally:
        on_result.close()

    # ── Final summary ──────────────────────────────────────────────
    elapsed = time.time() - run_start
    n_done = len(manifest.get("completed", {}))
    n_fail = len(manifest.get("failed", {}))
    n_total = len(manifest.get("selected_images", []))
    n_remaining = n_total - n_done - n_fail

    logger.info("=" * 60)
    logger.info("Run finished at %s", datetime.now().isoformat())
    logger.info("Elapsed      : %.1f min (%.0f sec)", elapsed / 60, elapsed)
    logger.info("This run     : %d images dispatched", len(image_paths))
    logger.info("Overall      : %d/%d completed, %d failed, %d remaining",
                n_done, n_total, n_fail, n_remaining)

    _print_results_table(args.output_dir, logger)

    if n_remaining > 0:
        logger.info("Re-run the same command to continue processing "
                    "the remaining %d images.", n_remaining)
    logger.info("Log file     : %s", log_file)
    logger.info("Manifest     : %s", os.path.join(args.output_dir, MANIFEST_NAME))
    logger.info("Results JSONL : %s", os.path.join(args.output_dir, RESULTS_NAME))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
