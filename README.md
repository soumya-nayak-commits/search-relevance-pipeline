# Search relevance pipeline

A two-stage retrieval pipeline over [SciFact](https://huggingface.co/datasets/BeIR/scifact) (BEIR): dense retrieval, cross-encoder reranking, rule-based query understanding, query expansion, and a hand-written evaluation harness. It ships as a small FastAPI service with per-stage latency reporting.

The analysis is in [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb). The reusable code is in [`src/`](src/).

## Pipeline

```
query ──► intent + entities ──► embed (MiniLM) ──► top-50 by dot product ──► cross-encoder rerank ──► top-k
          (rules, alias table)    384-d, unit length    exact, brute force        reads (query, doc) pairs
```

- **Corpus:** 5,183 scientific abstracts; **queries:** scientific claims; **qrels:** human relevance judgments (binary).
- **Bi-encoder:** `all-MiniLM-L6-v2`, title + abstract as one string per document (one document = one chunk).
- **Rerankers compared:** `cross-encoder/ms-marco-MiniLM-L-6-v2` and `BAAI/bge-reranker-base`.
- **Protocol:** choices are tuned on the train qrels (809 queries); the headline comparison uses the 300 test queries.

## Results (300 test queries)

| System | MRR@10 | P@10 | nDCG@10 |
|---|---|---|---|
| Bi-encoder only | 0.6047 | 0.0883 | 0.6451 |
| + MS MARCO MiniLM reranker | 0.6628 | 0.0923 | 0.6936 |
| + bge-reranker-base | 0.6754 | 0.0927 | 0.7049 |

Recall of the first stage: @10 = 0.783, @20 = 0.837, **@50 = 0.892**. About 11% of correct documents never reach the reranker, which caps what reranking can achieve. P@10 is low by design because most queries have only one or two relevant documents.

## Findings

- **The reranker stage is the big gain** (about +6 MRR, +5 nDCG). Swapping one reranker for the other added about +1 point.
- **Averages hide damage.** With bge-reranker-base, 73 queries improved, 189 were unchanged and 38 got worse. The five worst regressions were all at rank 1 before reranking.
- **A hypothesis that did not replicate.** From five regressions I hypothesized that the MS MARCO reranker mishandles negated claims (test: -0.027 average change in reciprocal rank on 40 negation-style queries vs +0.071 on the rest). On 74 negated queries in train the reranker was still clearly better than the bi-encoder (MRR@10 0.6783 vs 0.6185), and routing negated queries to the bi-encoder scored slightly *below* reranker-only (0.6732 vs 0.6787). The effect was not robust.
- **Rules beat a learned intent classifier on tiny data:** 0.80 vs 0.64 accuracy on 50 hand labels (the rules were later fixed against those same 50 and reach 0.94, an optimistic number). The labels are defined by cue words and embeddings capture topic more than grammar.
- **Query expansion:** PRF moved recall@50 by about +0.7 points and nothing else. Entity-alias expansion fixed one query outright but matched only 14 of 809 train queries (1.7%), so coverage, not the method, is the limit.

## Run it

```bash
pip install -r requirements-dev.txt

python -m src.build_index --out data/index      # embeds the corpus once, writes data/index/
pytest                                          # metric and query-understanding tests
jupyter notebook notebooks/analysis.ipynb

uvicorn app:app --port 8000
curl "localhost:8000/search?q=Lamins%20have%20no%20effect%20on%20nuclear%20membrane%20structure&k=5"
```

The response includes the ranked results, the detected intent and linked entities, and `latency_ms` for each stage (`understand`, `embed`, `retrieve`, `rerank`, `total`).

### Docker

```bash
docker build -t search-api .
docker run -p 8000:8000 -v "$(pwd)/data/index:/app/data/index" search-api
```

The embedding index is built once and mounted; it is not baked into the image.

## Benchmarks

```bash
python -m benchmarks.latency --n 100                 # p50 / p95 per stage
python -m benchmarks.latency --n 100 --no-rerank
python -m benchmarks.ann_vs_bruteforce               # FAISS HNSW vs exact: speed and recall@50
python -m benchmarks.ann_vs_bruteforce --synthetic 500000
```

Latency (CPU, 100 queries, 50 candidates): *not measured yet. Run the command above and paste the table here, with the machine it ran on.*

| stage | p50 (ms) | p95 (ms) |
|---|---|---|
| embed | | |
| retrieve | | |
| rerank | | |
| total | | |

ANN vs exact (recall@50 against exact search): *not measured yet.*

## Repo layout

```
app.py                      FastAPI service
src/pipeline.py             embed -> retrieve -> rerank, with per-stage timing
src/evaluation.py           MRR, P@k, nDCG, recall@k, per-query win/tie/loss comparison
src/query_understanding.py  intent rules, alias table, entity linking (exact + fuzzy)
src/expansion.py            PRF and entity-alias query expansion
src/build_index.py          one-off corpus embedding
tests/                      metric checks against hand calculations; intent/entity tests
benchmarks/                 latency and ANN-vs-brute-force scripts
notebooks/analysis.ipynb    the full analysis
data/intent_labels_set_a.json   50 hand-labeled train queries
```

## Limitations

- Test queries were read during error analysis, so the test numbers are slightly optimistic.
- The bi-encoder reads 256 tokens and the median document is longer, so more than half of the documents are truncated. Chunking was not tried.
- Intent labels come from one person on 50 queries; the final rules were tuned on that same set and have not been scored on fresh labels.
- The alias table is hand-written and tiny. In qrels-based evaluation, relevant but unlabeled documents count as misses.
- This is a single-node demo: brute-force search over 5k vectors, models loaded in-process, no auth, caching or monitoring, and no streaming ingestion. It is not a production system.

## Next steps

Chunking or a longer-context embedding model, reciprocal rank fusion, an imported entity table (HGNC / UMLS), a fresh labeled intent set, and fine-tuning the bi-encoder.
