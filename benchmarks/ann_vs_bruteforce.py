"""Exact search (brute force) vs an HNSW index (FAISS): speed and recall@50.

    pip install faiss-cpu
    python -m benchmarks.ann_vs_bruteforce                   # real SciFact vectors
    python -m benchmarks.ann_vs_bruteforce --synthetic 500000

Exact top-50 is the ground truth; recall@50 is the overlap of HNSW's top-50 with it.
At 5k documents brute force is already fast, so expect little or no speedup on the
real data. The synthetic run shows where ANN starts to matter. Note that random
vectors are a harder case for ANN than real embeddings, so treat that run as a
scaling illustration, not a quality claim.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    import faiss

    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--n-queries", type=int, default=300)
    ap.add_argument("--synthetic", type=int, default=0, help="use N random unit vectors instead")
    ap.add_argument("--k", type=int, default=50)
    args = ap.parse_args()

    idx = Path(args.index_dir)
    if args.synthetic:
        rng = np.random.default_rng(0)
        docs = rng.normal(size=(args.synthetic, 384)).astype("float32")
        docs /= np.linalg.norm(docs, axis=1, keepdims=True)
        qs = rng.normal(size=(args.n_queries, 384)).astype("float32")
        qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    else:
        from sentence_transformers import SentenceTransformer

        docs = np.ascontiguousarray(np.load(idx / "doc_vectors.npy"), dtype="float32")
        texts = list(json.loads((idx / "queries.json").read_text()).values())[: args.n_queries]
        qs = SentenceTransformer("all-MiniLM-L6-v2").encode(texts, normalize_embeddings=True).astype("float32")

    d = docs.shape[1]
    flat = faiss.IndexFlatIP(d)
    flat.add(docs)

    t0 = time.perf_counter()
    for q in qs:
        flat.search(q[None, :], args.k)
    flat_ms = (time.perf_counter() - t0) / len(qs) * 1000
    _, truth = flat.search(qs, args.k)

    print(f"{len(docs)} vectors, {len(qs)} queries, k={args.k}\n")
    print("| index | efSearch | ms / query | recall@k vs exact |\n|---|---|---|---|")
    print(f"| flat (exact) | - | {flat_ms:.3f} | 1.000 |")

    hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    t0 = time.perf_counter()
    hnsw.add(docs)
    build_s = time.perf_counter() - t0
    for ef in (16, 64, 128):
        hnsw.hnsw.efSearch = ef
        t0 = time.perf_counter()
        for q in qs:
            hnsw.search(q[None, :], args.k)
        ms = (time.perf_counter() - t0) / len(qs) * 1000
        _, got = hnsw.search(qs, args.k)
        recall = np.mean([len(set(g) & set(t)) / args.k for g, t in zip(got, truth)])
        print(f"| HNSW (M=32) | {ef} | {ms:.3f} | {recall:.3f} |")
    print(f"\nHNSW build time: {build_s:.2f}s")


if __name__ == "__main__":
    main()
