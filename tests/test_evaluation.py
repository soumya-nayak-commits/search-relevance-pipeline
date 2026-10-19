"""Metrics checked against hand calculations (same fake-data approach as the notebook)."""
import math

from src.evaluation import (
    compare_runs, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank,
)

RETRIEVED = list("abcdefghij")


def close(a, b):
    return math.isclose(a, b, abs_tol=1e-3)


def test_correct_doc_at_rank_5():
    rel = {"e"}
    assert close(precision_at_k(RETRIEVED, rel, 5), 0.2)
    assert close(precision_at_k(RETRIEVED, rel, 10), 0.1)
    assert close(reciprocal_rank(RETRIEVED, rel), 0.2)
    assert close(ndcg_at_k(RETRIEVED, rel, 10), 1 / math.log2(6))  # 0.387


def test_best_and_worst_case():
    assert close(reciprocal_rank(RETRIEVED, {"a"}), 1.0)
    assert close(ndcg_at_k(RETRIEVED, {"a"}, 10), 1.0)
    assert reciprocal_rank(RETRIEVED, {"z"}) == 0.0
    assert ndcg_at_k(RETRIEVED, {"z"}, 10) == 0.0


def test_two_relevant_docs():
    # the case the notebook flagged as untested: IDCG with 2 correct documents
    rel = {"a", "c"}
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))  # ~0.92
    assert close(ndcg_at_k(RETRIEVED, rel, 10), expected)
    assert close(recall_at_k(RETRIEVED, rel, 2), 0.5)
    assert close(recall_at_k(RETRIEVED, rel, 3), 1.0)


def test_rr_cutoff_is_mrr_at_k():
    assert reciprocal_rank(RETRIEVED, {"k"}) == 0.0
    assert reciprocal_rank(list("abcdefghijk"), {"k"}, k=10) == 0.0


def test_compare_runs_counts():
    qrels = {"q1": {"a"}, "q2": {"a"}, "q3": {"a"}}
    run_a = {"q1": ["b", "a"], "q2": ["a", "b"], "q3": ["a", "b"]}
    run_b = {"q1": ["a", "b"], "q2": ["a", "b"], "q3": ["b", "a"]}
    out = compare_runs(run_a, run_b, qrels)
    assert (out["winners"], out["ties"], out["losers"]) == (1, 1, 1)
