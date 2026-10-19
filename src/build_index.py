"""Embed the SciFact corpus once and save everything the service and notebook need.

    python -m src.build_index --out data/index

Writes: doc_vectors.npy, doc_ids.json, doc_texts.json, queries.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BI_ENCODER = "all-MiniLM-L6-v2"


def build_index(out_dir: str | Path = "data/index", model_name: str = BI_ENCODER) -> Path:
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
    queries = load_dataset("BeIR/scifact", "queries")["queries"]

    # One string per document (title + abstract); ids kept in the same order,
    # so vector i belongs to doc_ids[i].
    doc_ids = [str(d["_id"]) for d in corpus]
    doc_texts = [(d["title"] + " " + d["text"]).strip() for d in corpus]

    model = SentenceTransformer(model_name)
    vectors = model.encode(doc_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    np.save(out / "doc_vectors.npy", vectors.astype("float32"))
    (out / "doc_ids.json").write_text(json.dumps(doc_ids))
    (out / "doc_texts.json").write_text(json.dumps(doc_texts))
    (out / "queries.json").write_text(json.dumps({str(q["_id"]): q["text"] for q in queries}))
    print(f"saved {vectors.shape} to {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/index")
    ap.add_argument("--model", default=BI_ENCODER)
    args = ap.parse_args()
    build_index(args.out, args.model)
