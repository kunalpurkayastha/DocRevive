#!/usr/bin/env python3
"""
UCSM — Unified Context Similarity Metric for text restoration evaluation.

Implements the metric from the paper:
  S_edit: normalized edit similarity (1 - Levenshtein/max_len)
  S_sem: semantic similarity (GPT-2 embedding cosine, rescaled to [0,1])
  S_len: length similarity (min/max)
  S = geometric mean of S_edit^α, S_sem^β, S_len^γ  (α=β=γ=1)
  E_context: normalized -log P(GT|pre,post) from language model
  UCSM = S^(1 - E_context)
"""

from __future__ import annotations

from typing import Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None


def levenshtein_distance(s: str, t: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(s), len(t)
    if m == 0:
        return n
    if n == 0:
        return m
    # Use two rows to save memory
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost  # substitution
            )
        prev = curr
    return prev[n]


def s_edit(P: str, GT: str) -> float:
    """Normalized edit similarity: 1 - d_lev(GT,P) / max(|GT|,|P|)."""
    if not GT and not P:
        return 1.0
    denom = max(len(GT), len(P), 1)
    d = levenshtein_distance(GT, P)
    return 1.0 - d / denom


def s_len(P: str, GT: str) -> float:
    """Length similarity: min(|GT|,|P|) / max(|GT|,|P|)."""
    if not GT and not P:
        return 1.0
    a, b = len(GT), len(P)
    if max(a, b) == 0:
        return 1.0
    return min(a, b) / max(a, b)


def s_sem(P: str, GT: str, embed_fn=None) -> float:
    """Semantic similarity: (1 + cosine(e(GT), e(P))) / 2 in [0,1]."""
    if embed_fn is None or np is None:
        return 0.5  # neutral when no embedding model
    try:
        e_gt = embed_fn(GT)
        e_p = embed_fn(P)
        if e_gt is None or e_p is None:
            return 0.5
        e_gt, e_p = np.asarray(e_gt), np.asarray(e_p)
        cos = np.dot(e_gt, e_p) / (np.linalg.norm(e_gt) * np.linalg.norm(e_p) + 1e-12)
        return 0.5 * (float(cos) + 1.0)
    except Exception:
        return 0.5


def e_context(
    pre: str,
    post: str,
    GT: str,
    lm_logprob_fn=None,
    m: Optional[float] = None,
    M: Optional[float] = None,
) -> float:
    """
    Normalized context error: how predictable is GT from (pre, post)?
    E_context = (-log P(GT|pre,post) - m) / (M - m), clipped to [0,1].
    Lower = more predictable.
    """
    if lm_logprob_fn is None:
        return 0.5  # neutral when no LM
    try:
        full = (pre + " " + GT + " " + post).strip()
        ctx_only = (pre + " " + post).strip()
        if not full or not ctx_only:
            return 0.5
        log_p_full = lm_logprob_fn(full)
        log_p_ctx = lm_logprob_fn(ctx_only)
        # P(GT|pre,post) ≈ P(pre+GT+post) / P(pre+post)  -> log ratio
        log_p_gt_given_ctx = log_p_full - log_p_ctx if log_p_ctx > -1e9 else -10.0
        neg_log = -log_p_gt_given_ctx
        # Default calibration (can be overridden)
        m_val = m if m is not None else -2.0
        M_val = M if M is not None else 10.0
        e = (neg_log - m_val) / (M_val - m_val)
        return float(max(0.0, min(1.0, e)))
    except Exception:
        return 0.5


def ucsm(
    P: str,
    GT: str,
    pre: str = "",
    post: str = "",
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    embed_fn=None,
    lm_logprob_fn=None,
    m: Optional[float] = None,
    M: Optional[float] = None,
) -> Tuple[float, dict]:
    """
    Compute UCSM(P, GT) and component scores.

    Returns:
        (ucsm_score, {"s_edit": ..., "s_sem": ..., "s_len": ..., "s": ..., "e_context": ..., "ucsm": ...})
    """
    s_ed = s_edit(P, GT)
    s_se = s_sem(P, GT, embed_fn)
    s_le = s_len(P, GT)

    w_sum = alpha + beta + gamma
    s_combined = (s_ed ** alpha * s_se ** beta * s_le ** gamma) ** (1.0 / w_sum)

    e_ctx = e_context(pre, post, GT, lm_logprob_fn, m, M)

    ucsm_val = s_combined ** (1.0 - e_ctx)

    return ucsm_val, {
        "s_edit": s_ed,
        "s_sem": s_se,
        "s_len": s_le,
        "s": s_combined,
        "e_context": e_ctx,
        "ucsm": ucsm_val,
    }
