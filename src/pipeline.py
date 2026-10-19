"""Two-stage retrieval: bi-encoder candidates, then cross-encoder rerank.

Stage 1 (fast, rough): embed the query, dot product against every document vector.
Stage 2 (slow, accurate): the cross-encoder reads (query, document) together and
re-scores only the candidates it is given. It cannot recover a document that
stage 1 missed, so recall@N of stage 1 is the ceiling for the whole pipeline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

BI_ENCODER = "all-MiniLM-L6-v2"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class SearchPipeline:
    def __init__(
        self,
        index_dir: str | Path = "data/index",
        bi_encoder: str = BI_ENCODER,
        reranker: Optional[str] = RERANKER,
    ):
        # heavy imports kept local so the module can be imported (e.g. in tests) cheaply
        from sentence_transformers import CrossEncoder, SentenceTransformer

        index_dir = Path(index_dir)
        self.doc_vectors = np.load(index_dir / "doc_vectors.npy")
        self.doc_ids: List[str] = json.loads((index_dir / "doc_ids.json").read_text())
        self.doc_texts: List[str] = json.loads((index_dir / "doc_texts.json").read_text())
        self._text_by_id: Dict[str, str] = dict(zip(self.doc_ids, self.doc_texts))

        # Models are loaded once at startup, never per request.
        self.encoder = SentenceTransformer(bi_encoder)
        self.reranker = CrossEncoder(reranker, max_length=512) if reranker else None

    # ------------------------------------------------------------------ stages
    def embed_query(self, query: str) -> np.ndarray:
        return self.encoder.encode(query, normalize_embeddings=True)

    def retrieve(self, q_vec: np.ndarray, n: int = 50) -> List[Tuple[str, float]]:
        """Exact (brute-force) top-n by dot product == cosine similarity."""
        scores = self.doc_vectors @ q_vec
        n = min(n, len(scores))
        top = np.argpartition(-scores, n - 1)[:n] if n < len(scores) else np.arange(len(scores))
        top = top[np.argsort(-scores[top])]
        return [(self.doc_ids[p], float(scores[p])) for p in top]

    def rerank(self, query: str, cand_ids: List[str]) -> List[Tuple[str, float]]:
        pairs = [(query, self._text_by_id[d]) for d in cand_ids]
        scores = self.reranker.predict(pairs, batch_size=32)
        order = np.argsort(-scores)
        return [(cand_ids[i], float(scores[i])) for i in order]

    # --------------------------------------------------------------------- API
    def search(self, query: str, k: int = 10, candidates: int = 50, rerank: bool = True) -> dict:
        """Run the full pipeline and report per-stage latency in milliseconds."""
        t0 = time.perf_counter()
        q_vec = self.embed_query(query)
        t1 = time.perf_counter()
        cands = self.retrieve(q_vec, n=candidates)
        t2 = time.perf_counter()

        if rerank and self.reranker is not None:
            ranked = self.rerank(query, [d for d, _ in cands])
        else:
            ranked = cands
        t3 = time.perf_counter()

        results = [
            {"doc_id": d, "score": s, "snippet": self._text_by_id[d][:200]} for d, s in ranked[:k]
        ]
        return {
            "results": results,
            "latency_ms": {
                "embed": (t1 - t0) * 1000,
                "retrieve": (t2 - t1) * 1000,
                "rerank": (t3 - t2) * 1000,
                "total": (t3 - t0) * 1000,
            },
        }
