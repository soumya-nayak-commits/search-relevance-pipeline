"""Retrieval metrics, written by hand and checked against hand calculations.

Conventions
-----------
* ``retrieved`` is a ranked list of document ids (best first).
* ``relevant`` is the set of correct document ids for the query (binary relevance).
* Ids are strings everywhere, to avoid int/str mismatches between tables.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set

import numpy as np


def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / k


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: Set[str], k: int | None = None) -> float:
    """1 / rank of the first correct document; 0 if none in the (top-k) list."""
    for rank, d in enumerate(retrieved[:k] if k else retrieved, start=1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, d in enumerate(retrieved[:k], start=1)
        if d in relevant
    )
    n_ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, n_ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_run(run: Dict[str, List[str]], qrels: Dict[str, Set[str]], k: int = 10) -> Dict[str, float]:
    """Average MRR@k, P@k and nDCG@k over every query in ``run``."""
    qids = list(run)
    return {
        f"MRR@{k}": float(np.mean([reciprocal_rank(run[q], qrels[q], k) for q in qids])),
        f"P@{k}": float(np.mean([precision_at_k(run[q], qrels[q], k) for q in qids])),
        f"nDCG@{k}": float(np.mean([ndcg_at_k(run[q], qrels[q], k) for q in qids])),
    }


def mean_recall(run: Dict[str, List[str]], qrels: Dict[str, Set[str]], k: int) -> float:
    return float(np.mean([recall_at_k(run[q], qrels[q], k) for q in run]))


def compare_runs(
    run_a: Dict[str, List[str]],
    run_b: Dict[str, List[str]],
    qrels: Dict[str, Set[str]],
    k: int = 10,
) -> dict:
    """Per-query comparison of reciprocal rank (B minus A).

    Averages hide damage, so report how many queries improved, tied, or got worse.
    Returns counts plus the per-query diff array (in ``qids`` order).
    """
    qids = list(run_a)
    diff = np.array(
        [reciprocal_rank(run_b[q], qrels[q], k) - reciprocal_rank(run_a[q], qrels[q], k) for q in qids]
    )
    return {
        "qids": qids,
        "diff": diff,
        "winners": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
        "losers": int((diff < 0).sum()),
    }
