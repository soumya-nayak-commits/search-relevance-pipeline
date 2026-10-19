"""FastAPI service around the retrieval pipeline.

    uvicorn app:app --host 0.0.0.0 --port 8000
    curl "localhost:8000/search?q=Lamins%20have%20no%20effect%20on%20nuclear%20membrane%20structure"
"""
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query

from src.pipeline import SearchPipeline
from src.query_understanding import detect_intent, link_entities

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["pipeline"] = SearchPipeline(index_dir=os.getenv("INDEX_DIR", "data/index"))
    yield
    state.clear()


app = FastAPI(title="Search relevance pipeline", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    ready = "pipeline" in state
    return {"status": "ok" if ready else "starting", "documents": len(state["pipeline"].doc_ids) if ready else 0}


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, max_length=500),
    k: int = Query(10, ge=1, le=50),
    candidates: int = Query(50, ge=1, le=200, description="first-stage candidates handed to the reranker"),
    rerank: bool = True,
):
    t0 = time.perf_counter()
    understanding = {"intent": detect_intent(q), "entities": link_entities(q)}
    understand_ms = (time.perf_counter() - t0) * 1000

    out = state["pipeline"].search(q, k=k, candidates=max(candidates, k), rerank=rerank)
    out["latency_ms"] = {"understand": understand_ms, **out["latency_ms"]}
    out["latency_ms"]["total"] += understand_ms
    out["query_understanding"] = understanding
    return out
