"""
Multi-GPU Eager-Load Batch Runner for DocRevive.

Architecture
------------
  • 1 shared vLLM server   — GPUs 0-3 (tensor_parallel=4)
  • N persistent workers   — one per GPU (default: all 8)

Each worker subprocess:
  1. Loads ALL models once at startup (OCR, YOLO, GaMuSA, GSDM, postprocessor).
  2. Connects to the shared vLLM server over HTTP (no in-process Qwen copy).
  3. Pulls image paths from a shared work-queue and processes them indefinitely.
  4. Exits when it receives the sentinel ``None`` from the queue.

This eliminates per-image cold-start entirely: the first image processed by
each worker is just as fast as the last.
"""

from __future__ import annotations

import logging
import math
import os
import signal
import subprocess
import sys
import time
from multiprocessing import Process, Queue, get_context
from queue import Empty
from typing import List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  vLLM Server Management
# ═══════════════════════════════════════════════════════════════════════

_VLLM_HOST = "127.0.0.1"
_VLLM_PORT = 8199


def _wait_for_server(url: str, timeout: int = 300) -> bool:
    """Block until the vLLM server responds or *timeout* seconds elapse."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    return False


def start_vllm_server(
    model_path: str,
    gpu_ids: str = "0,1,2,3",
    tensor_parallel: int = 4,
    gpu_mem_util: float = 0.85,
    max_model_len: int = 4096,
) -> subprocess.Popen:
    """Launch a background vLLM OpenAI-compatible server.

    Args:
        model_path: Path to the Qwen model.
        gpu_ids: Comma-separated GPU indices for the server.
        tensor_parallel: Number of GPUs for tensor parallelism (must divide 8192).
        gpu_mem_util: GPU memory fraction.
        max_model_len: Maximum context length.

    Returns:
        The ``Popen`` handle for the server process.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--tensor-parallel-size", str(tensor_parallel),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--host", _VLLM_HOST,
        "--port", str(_VLLM_PORT),
        "--download-dir", "/data/hf_cache",
        "--disable-log-requests",
    ]

    logger.info("Starting vLLM server on GPUs %s (tensor_parallel=%d) …",
                gpu_ids, tensor_parallel)
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    url = f"http://{_VLLM_HOST}:{_VLLM_PORT}"
    if not _wait_for_server(url, timeout=300):
        proc.terminate()
        raise RuntimeError("vLLM server failed to start within 5 min")

    logger.info("vLLM server ready at %s", url)
    return proc


def stop_vllm_server(proc: subprocess.Popen) -> None:
    """Gracefully shut down the vLLM server."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    logger.info("vLLM server stopped.")


# ═══════════════════════════════════════════════════════════════════════
#  Persistent Worker
# ═══════════════════════════════════════════════════════════════════════

def _eager_worker(
    gpu_id: int,
    config_kwargs: dict,
    work_queue: Queue,
    result_queue: Queue,
) -> None:
    """Persistent pipeline worker — loads all models ONCE then loops.

    Protocol:
      • Receives image-path strings from *work_queue*.
      • Receives ``None`` as the shutdown sentinel — exits the loop.
      • Pushes ``(image_path, output_path | None)`` to *result_queue*.
      • On fatal error: pushes ``("__worker_error__", gpu_id, traceback_str)``.

    Args:
        gpu_id:        Physical GPU index this worker owns.
        config_kwargs: Arguments for ``Config()``.
        work_queue:    Shared queue of image paths (+ None sentinels).
        result_queue:  Queue for collected results.
    """
    import traceback

    def put_error(exc: BaseException) -> None:
        result_queue.put(("__worker_error__", gpu_id, traceback.format_exc()))

    try:
        # ── 1. Pin to a single GPU ─────────────────────────────────────
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        import torch
        torch.cuda.set_device(0)  # virtual device 0 = physical gpu_id

        sys.path.insert(0, config_kwargs.get("project_root", "."))

        from config import Config
        from pipeline import DocRevivePipeline

        # ── 2. Build config for this worker ───────────────────────────
        cfg = Config(**config_kwargs)
        cfg.det_device            = "cuda:0"
        cfg.reco_device           = "cuda:0"
        cfg.text_editor_device    = "cuda:0"
        cfg.text_editor_gpu_ids   = "0"
        cfg.inpaint_device        = "cuda:0"
        cfg.inpaint_gpu_ids       = "0"
        cfg.device                = "cuda:0"

        # ── 3. Connect to shared vLLM server (unless --no_llm) ────────
        if cfg.llm_mode not in ("dummy", "roberta", "qwen3"):
            cfg.llm_mode       = "http"
            cfg.llm_server_url = f"http://{_VLLM_HOST}:{_VLLM_PORT}"

        # ── 4. Eager-load ALL models ───────────────────────────────────
        pipeline = DocRevivePipeline(cfg)
        pipeline.preload_all()
        logger.info("[GPU %d] All models loaded. Waiting for work …", gpu_id)

        # ── 5. Process images from the queue ──────────────────────────
        while True:
            path = work_queue.get()
            if path is None:            # shutdown sentinel
                logger.info("[GPU %d] Received shutdown signal.", gpu_id)
                break

            try:
                logger.info("[GPU %d] Processing: %s", gpu_id, os.path.basename(path))
                pipeline.restore(path)
                base_name = os.path.splitext(os.path.basename(path))[0]
                out_path  = os.path.join(
                    cfg.output_dir, base_name, f"{base_name}_restored.png"
                )
                result_queue.put((path, out_path if os.path.exists(out_path) else None))
            except Exception as exc:
                logger.error("[GPU %d] Failed %s: %s", gpu_id, path, exc,
                             exc_info=True)
                # Include error so main process can log it
                result_queue.put((path, None, traceback.format_exc()))

    except Exception as exc:
        logger.error("[GPU %d] Fatal error: %s", gpu_id, exc, exc_info=True)
        put_error(exc)


# ═══════════════════════════════════════════════════════════════════════
#  ParallelRunner
# ═══════════════════════════════════════════════════════════════════════

class ParallelRunner:
    """Distribute DocRevive across multiple GPUs with eager model loading.

    Usage::

        runner = ParallelRunner(
            worker_gpu_ids=[0,1,2,3,4,5,6,7],
            vllm_gpu_ids="0,1,2,3",
            vllm_tensor_parallel=4,
            config_kwargs={...},
        )
        results = runner.run(image_paths)

    All workers load their models once at startup.  Subsequent images are
    processed without any model cold-start overhead.
    """

    def __init__(
        self,
        worker_gpu_ids: Optional[List[int]] = None,
        vllm_gpu_ids: str = "0,1,2,3",
        vllm_tensor_parallel: int = 4,
        config_kwargs: Optional[dict] = None,
        # Legacy compat args
        num_workers: int = 0,
        gpu_start:   int = 0,
    ) -> None:
        if worker_gpu_ids is not None:
            self.worker_gpu_ids = worker_gpu_ids
        elif num_workers > 0:
            # Legacy: gpu_start..gpu_start+num_workers-1
            self.worker_gpu_ids = list(range(gpu_start, gpu_start + num_workers))
        else:
            self.worker_gpu_ids = list(range(8))  # default: all 8 GPUs

        self.vllm_gpu_ids        = vllm_gpu_ids
        self.vllm_tensor_parallel = vllm_tensor_parallel
        self.config_kwargs       = config_kwargs or {}

    def run(
        self,
        image_paths: List[str],
        on_result: Optional[callable] = None,
    ) -> List[Tuple[str, Optional[str]]]:
        """Process *image_paths* in parallel with persistent eager workers.

        Args:
            image_paths: Paths to input images.
            on_result:   Optional callback ``(image_path, output_path_or_None,
                         error_str_or_None) -> None`` invoked after each image.

        Returns:
            List of ``(image_path, output_path | None)`` tuples.
        """
        if not image_paths:
            return []

        from tqdm import tqdm

        from config import Config
        cfg = Config(**self.config_kwargs)

        no_llm = self.config_kwargs.get("llm_mode", "local") in ("dummy", "roberta", "qwen3")

        server_proc = None
        if no_llm:
            logger.info("--no_llm mode: skipping vLLM server. All GPUs free for pipeline.")
        else:
            server_proc = start_vllm_server(
                model_path      = cfg.qwen_model_path,
                gpu_ids         = self.vllm_gpu_ids,
                tensor_parallel = self.vllm_tensor_parallel,
                gpu_mem_util    = cfg.vllm_gpu_memory_utilization,
                max_model_len   = cfg.vllm_max_model_len,
            )

        try:
            ctx = get_context("spawn")
            n_workers = len(self.worker_gpu_ids)
            work_queue: Queue = ctx.Queue()
            result_queue: Queue = ctx.Queue()
            workers: List[Process] = []
            for gpu_id in self.worker_gpu_ids:
                p = ctx.Process(
                    target=_eager_worker,
                    args=(gpu_id, self.config_kwargs, work_queue, result_queue),
                    daemon=False,
                )
                p.start()
                workers.append(p)
            logger.info("Spawned %d persistent workers on GPUs %s",
                        n_workers, self.worker_gpu_ids)

            for path in image_paths:
                work_queue.put(path)
            for _ in workers:
                work_queue.put(None)

            # ── Collect results with live progress bar ─────────────
            results: List[Tuple[str, Optional[str]]] = []
            expected_results = len(image_paths)
            received_results = 0
            n_success = 0

            pbar = tqdm(
                total=expected_results,
                desc="DocRevive",
                unit="img",
                dynamic_ncols=True,
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}] "
                    "{postfix}"
                ),
            )

            while received_results < expected_results:
                try:
                    item = result_queue.get(timeout=5)
                except Empty:
                    dead_workers = [w for w in workers if not w.is_alive()]
                    if len(dead_workers) == len(workers):
                        break
                    continue

                if isinstance(item, tuple) and len(item) == 3 and item[0] == "__worker_error__":
                    _, gid, tb = item
                    logger.error("[GPU %d] Worker crashed:\n%s", gid, tb)
                    continue

                if isinstance(item, tuple) and len(item) == 3 and item[1] is None:
                    path, _, tb = item
                    logger.error("Image %s failed:\n%s", os.path.basename(path), tb)
                    results.append((path, None, tb))
                    received_results += 1
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"ok={n_success} fail={received_results - n_success}"
                    )
                    if on_result:
                        on_result(path, None, tb)
                    continue

                img_path = item[0]
                out_path = item[1] if len(item) >= 2 else None
                results.append(item)
                received_results += 1
                if out_path is not None:
                    n_success += 1
                pbar.update(1)
                pbar.set_postfix_str(
                    f"ok={n_success} fail={received_results - n_success}"
                )
                if on_result:
                    on_result(img_path, out_path, None)

            pbar.close()

            for w in workers:
                w.join()

            logger.info("Batch complete: %d/%d successful.",
                        n_success, len(image_paths))
            return results

        finally:
            if server_proc is not None:
                stop_vllm_server(server_proc)
