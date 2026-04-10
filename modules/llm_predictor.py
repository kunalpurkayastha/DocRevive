"""
Offline LLM Missing Text Prediction module.

Uses vLLM with the Qwen model (stored at ``/data/qwen/``) for offline,
GPU-accelerated text prediction.  Replaces the OpenAI API approach.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

import requests as _requests

from config import Config

logger = logging.getLogger(__name__)


class LLMPredictor:
    """Predict missing text for blank regions using a local Qwen model
    served via vLLM.

    The model is loaded once and reused across all predictions.  Uses
    tensor parallelism to distribute across multiple GPUs.
    """

    # System prompt identical in intent to the OpenAI version
    SYSTEM_PROMPT = (
        "You are a document restoration assistant. "
        "You are given tokenized fragments of text lines from a scanned "
        "document where some text is missing or occluded. "
        "Each token has the format: "
        '<preN="..."></blankN><postN="..."> </blankN> = ? | max_chars_including_blanks=M\n'
        "Your task is to predict the missing text for each </blankN>=? field. "
        "The max_chars value indicates the maximum number of characters "
        "(including spaces) that should fit in each blank. "
        "Try to keep your predictions within this constraint while "
        "ensuring they make sense in context. "
        'Return your answer ONLY as a valid JSON object where keys are "0", "1", ... '
        "and values are the predicted missing strings. "
        "Do not include any explanation or extra text outside the JSON object."
    )

    def __init__(self, config: Config) -> None:
        """Initialise the vLLM engine with the Qwen model.

        Args:
            config: Global pipeline configuration.
        """
        self.config = config
        self._llm = None  # Lazy-loaded

    def _ensure_loaded(self) -> None:
        """Lazy-load vLLM engine on first use (in-process mode only)."""
        if self._llm is not None:
            return
        if getattr(self.config, "llm_mode", "local") == "http":
            return  # HTTP mode doesn't need local model

        from vllm import LLM

        logger.info(
            "Loading Qwen model from %s with tensor_parallel=%d …",
            self.config.qwen_model_path,
            self.config.vllm_tensor_parallel_size,
        )
        self._llm = LLM(
            model=self.config.qwen_model_path,
            tensor_parallel_size=self.config.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
            max_model_len=self.config.vllm_max_model_len,
            trust_remote_code=True,
            dtype="bfloat16",
            download_dir="/data/hf_cache",
        )
        logger.info("Qwen model loaded successfully.")

    # ── Dummy (no-LLM) prediction ──────────────────────────────────

    @staticmethod
    def _predict_dummy(gap_metadata: List[Dict]) -> Dict[int, Dict]:
        """Return placeholder text without loading any model.

        Each blank is filled with a sequence of underscores scaled to
        ``max_chars``, giving downstream stages realistic bbox sizes to
        work with while requiring zero GPU memory.
        """
        predictions: Dict[int, Dict] = {}
        for gm in gap_metadata:
            n = max(1, gm.get("max_chars", 5))
            # Use a repeating word-like pattern so text editing has something
            # stylistically reasonable to render (underscores are invisible)
            placeholder = ("_ " * ((n + 1) // 2)).strip()[:n]
            predictions[gm["blank_id"]] = {
                **gm,
                "predicted_text": placeholder,
            }
            logger.debug("Dummy blank_%d → %r", gm["blank_id"], placeholder)
        return predictions

    # ── RoBERTa fill-mask prediction ───────────────────────────────

    def _ensure_roberta(self) -> None:
        """Lazy-load the HuggingFace RoBERTa fill-mask pipeline (once)."""
        if getattr(self, "_roberta_pipe", None) is not None:
            return
        from transformers import pipeline
        model_name = getattr(self.config, "roberta_model_name", "roberta-large")
        device_str = getattr(self.config, "device", "cuda")
        # Map device string → int index for transformers pipeline
        try:
            if ":" in str(device_str):
                dev = int(str(device_str).split(":")[-1])
            elif str(device_str).lower() == "cuda":
                dev = 0
            else:
                dev = -1  # CPU
        except (ValueError, AttributeError):
            dev = 0
        logger.info("Loading RoBERTa fill-mask '%s' on device %s …",
                    model_name, device_str)
        self._roberta_pipe = pipeline(
            "fill-mask", model=model_name, device=dev, top_k=1,
        )
        logger.info("RoBERTa ready.")

    def _predict_roberta(
        self,
        gap_metadata: List[Dict],
    ) -> Dict[int, Dict]:
        """Predict missing text using RoBERTa masked language modelling.

        For each blank:
          • Build ``"{pre_text} <mask> {post_text}"``
          • Get top-1 token prediction.
          • Repeat for `n_words` estimated from ``max_chars`` (≈ chars/6).
        """
        self._ensure_roberta()
        MASK = "<mask>"
        predictions: Dict[int, Dict] = {}

        for gm in gap_metadata:
            pre_text  = (gm.get("pre_text")  or "").strip()
            post_text = (gm.get("post_text") or "").strip()
            max_chars = max(1, gm.get("max_chars", 6))
            n_words   = max(1, round(max_chars / 6))
            k_token   = f"[K={max_chars}]"  # length-hint token matching training format

            predicted_words: List[str] = []
            for _ in range(n_words):
                so_far = " ".join(predicted_words)
                parts  = [k_token] + [p for p in [pre_text, so_far, MASK, post_text] if p]
                query  = " ".join(parts)
                try:
                    result = self._roberta_pipe(query)
                    top    = result[0] if isinstance(result[0], dict) else result[0][0]
                    predicted_words.append(top["token_str"].strip())
                except Exception as exc:
                    logger.warning("RoBERTa fill failed for blank %d: %s",
                                   gm["blank_id"], exc)
                    break

            predicted = " ".join(predicted_words)
            predictions[gm["blank_id"]] = {**gm, "predicted_text": predicted}
            logger.debug("RoBERTa blank_%d → %r", gm["blank_id"], predicted)

        return predictions

    # ── Qwen3 (fine-tuned causal LM, same format as training) ────────────────

    def _ensure_qwen3(self) -> None:
        """Lazy-load the fine-tuned Qwen3 causal LM (once)."""
        if getattr(self, "_qwen3_model", None) is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        model_name = getattr(self.config, "qwen3_model_name",
                             "/data/kp/qwen3_finetune/checkpoints/qwen3_finetune/final")
        device_str = getattr(self.config, "device", "cuda")
        try:
            dev = int(str(device_str).split(":")[-1]) if ":" in str(device_str) else 0
        except (ValueError, AttributeError):
            dev = 0
        device = f"cuda:{dev}" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Qwen3 (finetuned) '%s' on %s …", model_name, device)
        self._qwen3_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._qwen3_tokenizer.pad_token is None:
            self._qwen3_tokenizer.pad_token = self._qwen3_tokenizer.eos_token
        self._qwen3_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )
        self._qwen3_model.eval()
        logger.info("Qwen3 (finetuned) ready.")

    def _predict_qwen3(self, gap_metadata: List[Dict]) -> Dict[int, Dict]:
        """Predict missing text using fine-tuned Qwen3 causal LM.

        Uses training format: "[K=N] pre _____ post. Missing word: " → generate.
        """
        self._ensure_qwen3()
        predictions: Dict[int, Dict] = {}

        for gm in gap_metadata:
            pre_text = (gm.get("pre_text") or "").strip()
            post_text = (gm.get("post_text") or "").strip()
            max_chars = max(1, gm.get("max_chars", 6))
            k_token = f"[K={max_chars}]"
            prompt = f"{k_token} {pre_text} _____ {post_text}. Missing word: "
            import torch
            inputs = self._qwen3_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            ).to(self._qwen3_model.device)
            with torch.no_grad():
                out = self._qwen3_model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=self._qwen3_tokenizer.pad_token_id,
                )
            generated = self._qwen3_tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            # Trim to first line/phrase and max_chars
            if "\n" in generated:
                generated = generated.split("\n")[0]
            if "." in generated:
                generated = generated.split(".")[0].strip()
            predicted = generated[:max_chars].strip()
            if not predicted:
                predicted = "_" * max(1, min(max_chars, 5))
            predictions[gm["blank_id"]] = {**gm, "predicted_text": predicted}
            logger.debug("Qwen3 blank_%d → %r", gm["blank_id"], predicted)

        return predictions

    # ── HTTP-based prediction (shared vLLM server) ─────────────────

    def _predict_http(
        self,
        prompt_string: str,
        gap_metadata: List[Dict],
    ) -> Dict[int, Dict]:
        """Query the shared vLLM server via OpenAI-compatible API."""
        url = getattr(self.config, "llm_server_url", "http://127.0.0.1:8199")
        user_prompt = f"Restore the missing text for each blank:\n\n{prompt_string}"

        payload = {
            "model": self.config.qwen_model_path,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.config.llm_max_tokens,
            "temperature": self.config.llm_temperature,
            "top_p": self.config.llm_top_p,
        }

        try:
            resp = _requests.post(
                f"{url}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            raw_text = resp.json()["choices"][0]["message"]["content"].strip()
            logger.info("LLM HTTP response: %s", raw_text[:200])
            return self._parse_response(raw_text, gap_metadata)
        except Exception as exc:
            logger.error("LLM HTTP prediction failed: %s", exc)
            return {
                gm["blank_id"]: {**gm, "predicted_text": ""}
                for gm in gap_metadata
            }

    # ── Prediction ─────────────────────────────────────────────────

    def predict(
        self,
        prompt_string: str,
        gap_metadata: List[Dict],
    ) -> Dict[int, Dict]:
        """Predict missing text for all blanks in a prompt.

        Automatically uses HTTP mode or in-process mode based on config.

        Args:
            prompt_string: The ``<pre/blank/post>`` prompt generated by
                blank extraction.
            gap_metadata: List of gap metadata dicts (one per blank).

        Returns:
            Dict mapping ``blank_id`` → enriched metadata including
            ``predicted_text``.
        """
        if not prompt_string.strip() or not gap_metadata:
            return {}

        llm_mode = getattr(self.config, "llm_mode", "local")

        # ── Dummy mode ────────────────────────────────────
        if llm_mode == "dummy":
            return self._predict_dummy(gap_metadata)

        # ── RoBERTa fill-mask ─────────────────────────────
        if llm_mode == "roberta":
            return self._predict_roberta(gap_metadata)

        # ── Qwen3 (fine-tuned causal LM) ───────────────────
        if llm_mode == "qwen3":
            return self._predict_qwen3(gap_metadata)

        # ── HTTP mode (shared vLLM server) ─────────────────
        if llm_mode == "http":
            return self._predict_http(prompt_string, gap_metadata)

        # In-process mode: load model locally
        self._ensure_loaded()

        from vllm import SamplingParams

        user_prompt = f"Restore the missing text for each blank:\n\n{prompt_string}"

        full_prompt = (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        sampling_params = SamplingParams(
            max_tokens=self.config.llm_max_tokens,
            temperature=self.config.llm_temperature,
            top_p=self.config.llm_top_p,
            stop=["<|im_end|>"],
        )

        try:
            outputs = self._llm.generate([full_prompt], sampling_params)
            raw_text = outputs[0].outputs[0].text.strip()
            logger.info("LLM raw response: %s", raw_text[:200])

            predictions = self._parse_response(raw_text, gap_metadata)
        except Exception as exc:
            logger.error("LLM prediction failed: %s", exc, exc_info=True)
            predictions = {
                gm["blank_id"]: {**gm, "predicted_text": ""}
                for gm in gap_metadata
            }

        return predictions

    def predict_batch(
        self,
        prompts: List[str],
        all_metadata: List[List[Dict]],
    ) -> List[Dict[int, Dict]]:
        """Predict for multiple documents in a single batch.

        Args:
            prompts: List of prompt strings.
            all_metadata: List of gap-metadata lists (one per document).

        Returns:
            List of prediction dicts (one per document).
        """
        if not prompts:
            return []

        self._ensure_loaded()

        from vllm import SamplingParams

        full_prompts = []
        for prompt in prompts:
            user_prompt = f"Restore the missing text for each blank:\n\n{prompt}"
            full_prompts.append(
                f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        sampling_params = SamplingParams(
            max_tokens=self.config.llm_max_tokens,
            temperature=self.config.llm_temperature,
            top_p=self.config.llm_top_p,
            stop=["<|im_end|>"],
        )

        try:
            outputs = self._llm.generate(full_prompts, sampling_params)
        except Exception as exc:
            logger.error("Batch LLM prediction failed: %s", exc)
            return [
                {gm["blank_id"]: {**gm, "predicted_text": ""} for gm in meta}
                for meta in all_metadata
            ]

        results = []
        for output, metadata in zip(outputs, all_metadata):
            raw_text = output.outputs[0].text.strip()
            results.append(self._parse_response(raw_text, metadata))

        return results

    # ── Response Parsing ───────────────────────────────────────────

    @staticmethod
    def _parse_response(
        raw_text: str, gap_metadata: List[Dict]
    ) -> Dict[int, Dict]:
        """Parse LLM response into a prediction dict.

        Tries JSON parsing first, then regex fallback.
        """
        predictions: Dict[int, Dict] = {}

        # Try direct JSON parse
        parsed = None
        try:
            # Extract JSON from response (may have extra text)
            json_match = re.search(r"\{[^{}]*\}", raw_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Fallback: regex for key-value pairs
        if parsed is None:
            parsed = {}
            pattern = r'["\']?(\d+)["\']?\s*:\s*["\']([^"\']*)["\']'
            for match in re.finditer(pattern, raw_text):
                parsed[match.group(1)] = match.group(2)

        # Fallback 2: predicted_blanks format
        if not parsed:
            dict_pattern = r"predicted_blanks\s*=\s*\{([^}]*)\}"
            dict_match = re.search(dict_pattern, raw_text, re.DOTALL)
            if dict_match:
                pairs = re.findall(r'(\d+)\s*:\s*"([^"]*)"', dict_match.group(1))
                for key, value in pairs:
                    parsed[key] = value

        # Map to metadata
        for gm in gap_metadata:
            blank_id = gm["blank_id"]
            text = ""
            # Try different key formats
            for key in [str(blank_id), blank_id, f"blank_{blank_id}"]:
                if key in parsed:
                    text = str(parsed[key])
                    break

            predictions[blank_id] = {
                **gm,
                "predicted_text": text,
            }

        return predictions
