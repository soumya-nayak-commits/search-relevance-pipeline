"""Query expansion: pseudo-relevance feedback (PRF) and entity-alias expansion."""
from __future__ import annotations

from typing import List

import numpy as np

from .query_understanding import ALIAS_TABLE, link_entities


def prf_query_vector(q_vec: np.ndarray, doc_vectors: np.ndarray, m: int = 3, alpha: float = 0.7) -> np.ndarray:
    """Blend the query vector with the mean of the top-m first-pass documents.

    ``alpha`` is the weight on the original query. All vectors are unit length, so the
    blended vector is re-normalized to keep dot product == cosine similarity.
    """
    if m <= 0:
        return q_vec
    top_m = np.argsort(-(doc_vectors @ q_vec))[:m]
    q_new = alpha * q_vec + (1 - alpha) * doc_vectors[top_m].mean(axis=0)
    return q_new / np.linalg.norm(q_new)


def expand_with_entities(text: str) -> str:
    """Append known aliases of linked entities, so vocabulary mismatches can still match."""
    extra: List[str] = []
    for ent in link_entities(text):
        extra += ALIAS_TABLE[ent]
    return f"{text} {' '.join(extra)}" if extra else text
