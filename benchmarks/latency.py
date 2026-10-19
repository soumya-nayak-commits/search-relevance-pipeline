"""Per-stage latency (p50 / p95) of the search pipeline.

    python -m src.build_index --out data/index      # once
    python -m benchmarks.latency --n 100            # with reranker
    python -m benchmarks.latency --n 100 --no-rerank
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.pipeline import SearchPipeline

STAGES = ["embed", "retrieve", "rerank", "total"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--n", type=int, default=100, help="number of queries")
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--out", default=None, help="optional path for a JSON results file")
    args = ap.parse_args()

    queries = list(json.loads((Path(args.index_dir) / "queries.json").read_text()).values())
    random.Random(42).shuffle(queries)
    queries = queries[: args.n]

    pipe = SearchPipeline(args.index_dir)
    for q in queries[:3]:  # warm-up: first calls are slower (lazy init, caches)
        pipe.search(q, candidates=args.candidates, rerank=not args.no_rerank)

    samples = {s: [] for s in STAGES}
    for q in queries:
        lat = pipe.search(q, candidates=args.candidates, rerank=not args.no_rerank)["latency_ms"]
        for s in STAGES:
            samples[s].append(lat[s])

    table = {
        s: {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "mean": float(np.mean(v))}
        for s, v in samples.items()
    }
    print(f"\n{len(queries)} queries, {args.candidates} candidates, rerank={not args.no_rerank}\n")
    print("| stage | p50 (ms) | p95 (ms) | mean (ms) |\n|---|---|---|---|")
    for s, r in table.items():
        print(f"| {s} | {r['p50']:.1f} | {r['p95']:.1f} | {r['mean']:.1f} |")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
