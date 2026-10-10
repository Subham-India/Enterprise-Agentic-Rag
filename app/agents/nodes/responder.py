import re

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.nodes.planner import format_history
from app.agents.state import AgentState
from app.gateway.client import get_llm

RAG_PROMPT = """You are a technical documentation assistant. Answer the question using ONLY the context below.

Rules:
- If the context does not contain the answer, say you could not find it in the documents. Do not guess.
- Be concise and practical. Keep commands and code exactly as written in the context.
- Cite sources inline as [1], [2], matching the numbered context blocks.

Chat history:
{history}

Context:
{context}

Question: {question}

Answer:"""

CHAT_PROMPT = """You are a friendly technical documentation assistant for Kubernetes topics
(jobs, cron jobs, pod autoscaling, monitoring, work queues).

Chat history:
{history}

User: {question}

Reply briefly and naturally. If it is small talk, invite them to ask a technical question."""

REFUSAL = (
    "I can only help with questions about the Kubernetes documentation in my knowledge base "
    "(jobs, cron jobs, autoscaling, monitoring and work queues). "
    "Could you ask something along those lines?"
)


def build_context(documents: list[dict]) -> str:
    blocks = []
    for i, d in enumerate(documents, start=1):
        blocks.append(f"[{i}] (source: {d['source']})\n{d['text']}")
    return "\n\n".join(blocks)


def responder_node(state: AgentState) -> dict:
    question = state["question"]

    if state.get("route") == "off_topic":
        return {
            "answer": REFUSAL,
            "sources": [],
            "messages": [HumanMessage(question), AIMessage(REFUSAL)],
        }

    history = format_history(state.get("messages", []))
    llm = get_llm("responder")
    sources: list[str] = []

    if state.get("route") == "chat":
        answer = llm.invoke(CHAT_PROMPT.format(history=history, question=question)).content
    else:
        documents = state.get("documents") or []
        prompt = RAG_PROMPT.format(
            history=history, context=build_context(documents), question=question
        )
        answer = llm.invoke(prompt).content

        # Only list sources the answer actually cited, like [1] or [2]
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        sources = sorted({documents[n - 1]["source"] for n in cited if 1 <= n <= len(documents)})
        if sources:
            answer += "\n\nSources: " + ", ".join(sources)

    return {
        "answer": answer,
        "sources": sources,
        "messages": [HumanMessage(question), AIMessage(answer)],
    }