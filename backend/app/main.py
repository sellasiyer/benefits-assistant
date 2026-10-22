"""FastAPI service exposing the agent.

Endpoints
  GET  /health            liveness + which providers are configured
  POST /ingest            (re)index everything in data/docs
  POST /documents         upload one .md/.txt file and index it
  POST /chat              JSON request/response (easy to test, easy to call from Java)
  POST /chat/stream       Server-Sent Events: step -> token... -> citations -> done
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .agent import graph
from .config import settings
from .vectorstore import doc_count, ingest_dir, ingest_text

log = logging.getLogger("benefits")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if doc_count() == 0:  # first run: build the index
        log.info("indexed %s", ingest_dir())
    yield


app = FastAPI(title="Benefits Assistant (GenAI RAG demo)", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class Citation(BaseModel):
    id: int
    source: str
    section: str
    score: float
    text: str


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    route: str
    citations: list[Citation]
    redacted: list[str]
    trace: list[str]
    latency_ms: int


def _citations(state: dict) -> list[dict]:
    return [{k: d[k] for k in ("id", "source", "section", "score", "text")} for d in state.get("docs", [])] \
        if state.get("route") == "generate" else []


@app.get("/health")
def health():
    return {"status": "ok", "llm": settings.llm_provider, "embeddings": settings.embedding_provider, "chunks": doc_count()}


@app.post("/ingest")
def ingest():
    return {"indexed": ingest_dir(), "chunks": doc_count()}


@app.post("/documents")
async def upload(file: UploadFile):
    if not file.filename or not file.filename.endswith((".md", ".txt")):
        raise HTTPException(400, "Only .md or .txt files")
    n = ingest_text((await file.read()).decode("utf-8"), file.filename)
    return {"source": file.filename, "chunks": n}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    rid, t0 = uuid.uuid4().hex[:8], time.perf_counter()
    state = await graph.ainvoke({"question": req.question, "trace": []})
    ms = int((time.perf_counter() - t0) * 1000)
    # audit log: redacted question only, never raw PHI
    log.info("rid=%s route=%s ms=%d q=%r", rid, state.get("route"), ms, state.get("safe_question", "[blocked]"))
    return ChatResponse(
        request_id=rid, answer=state["answer"], route=state.get("route", ""),
        citations=_citations(state), redacted=state.get("redacted", []),
        trace=state["trace"], latency_ms=ms,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def events():
        final: dict = {"trace": []}
        streamed = False
        async for mode, payload in graph.astream(
            {"question": req.question, "trace": []}, stream_mode=["updates", "messages"]
        ):
            if mode == "messages":
                chunk, meta = payload
                if meta.get("langgraph_node") == "generate" and chunk.content:
                    streamed = True
                    yield {"event": "token", "data": json.dumps({"text": chunk.content})}
            else:  # "updates": {node_name: partial_state}
                for node, update in payload.items():
                    update = update or {}
                    for k, v in update.items():
                        final[k] = final.get(k, []) + v if k == "trace" else v
                    yield {"event": "step", "data": json.dumps({"node": node, "trace": update.get("trace", [])})}
        if not streamed and final.get("answer"):  # tool / refusal answers arrive whole
            yield {"event": "token", "data": json.dumps({"text": final["answer"]})}
        yield {"event": "citations", "data": json.dumps(_citations(final))}
        yield {"event": "done", "data": json.dumps({"route": final.get("route"), "redacted": final.get("redacted", [])})}

    return EventSourceResponse(events())
