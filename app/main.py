import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agents.graph import ask
from app.guardrails import guard, initialize_rails

# Keep the console readable: hide the per-request HTTP log lines
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_rails()  # build the guardrails once, at startup
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
    # blocking calls below don't freeze the server.
    thread_id = req.thread_id or str(uuid.uuid4())

    # 1. Guardrails gate: attacks and off-topic questions stop here
    fired, reply = guard(req.question)
    if fired:
        return QueryResponse(
            answer=reply,
            route="guardrail",
            search_query="",
            sources=[],
            thread_id=thread_id,
        )

    # 2. Otherwise run the agent
    state = ask(req.question, thread_id)
    return QueryResponse(
        answer=state["answer"],
        route=state.get("route", ""),
        search_query=state.get("search_query", ""),
        sources=state.get("sources", []),
        thread_id=thread_id,
    )


# Serve the chat page. Must be registered after all API routes.
app.mount("/", StaticFiles(directory="web", html=True), name="web")