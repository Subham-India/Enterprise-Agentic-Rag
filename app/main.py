import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.agents.graph import ask

# Keep the console readable: hide the per-request HTTP log lines
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup/shutdown hooks go here later (warm up models, close clients)
    yield


app = FastAPI(title="Enterprise RAG", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None  # same id = same conversation


class QueryResponse(BaseModel):
    answer: str
    route: str
    search_query: str
    sources: list[str]
    thread_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    # A plain `def` endpoint runs in FastAPI's thread pool, so the
    # blocking agent call doesn't freeze the server.
    thread_id = req.thread_id or str(uuid.uuid4())
    state = ask(req.question, thread_id)
    return QueryResponse(
        answer=state["answer"],
        route=state.get("route", ""),
        search_query=state.get("search_query", ""),
        sources=state.get("sources", []),
        thread_id=thread_id,
    )