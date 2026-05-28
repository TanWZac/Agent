"""Semantic comparison utilities for response matching and scoring.

Provides a small facade over the embedding model for computing semantic
similarity between texts. Falls back to token-set similarity when the
sentence-transformers model is not available.
"""

from __future__ import annotations

from typing import Iterable, Tuple
import logging

logger = logging.getLogger("core.semantic")


def _token_jaccard(a: str, b: str) -> float:
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    inter = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(inter) / len(union)


def semantic_score(text: str, candidates: Iterable[str]) -> Tuple[float, str]:
    """Return (best_score, best_candidate) comparing `text` against candidates.

    Tries to use `src.core.embeddings.embed` + cosine similarity. If that
    import fails (model not installed), falls back to token Jaccard.
    """
    try:
        from src.core.embeddings import embed, cosine_similarity
        import numpy as np

        cand_list = list(candidates)
        if not cand_list:
            return 0.0, ""
        # Compute embeddings
        emb_q = embed([text])
        emb_c = embed(cand_list)
        sims = cosine_similarity(emb_q[0], emb_c)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        return best_score, cand_list[best_idx]

    except Exception as e:
        logger.debug("Embedding not available, using token Jaccard fallback: %s", e)
        best_score = 0.0
        best_cand = ""
        for c in candidates:
            s = _token_jaccard(text, c)
            if s > best_score:
                best_score = s
                best_cand = c
        return best_score, best_cand
