from langchain_core.messages import HumanMessage

from app.agents.state import AgentState
from app.gateway.client import get_llm

PLANNER_PROMPT = """You are the routing step of a technical documentation assistant.
The knowledge base covers Kubernetes: jobs, cron jobs, pod autoscaling (HPA/VPA), job monitoring, work queues and cluster architecture.

Chat history:
{history}

Latest user message: {question}

Pick exactly one:
- CHAT: only a greeting, thanks, or small talk about the assistant itself.
- SEARCH: <a standalone search query>: any question about Kubernetes, containers or cloud infrastructure, even if it might not be covered in the knowledge base. The query must make sense without the chat history (replace words like "it" or "that" with what they refer to).
- OFF_TOPIC: anything else (general knowledge, creative writing, personal advice, unrelated coding, news, math, and so on).

Reply with one line only, starting with CHAT, SEARCH: or OFF_TOPIC."""


def format_history(messages, limit: int = 6) -> str:
    lines = []
    for m in messages[-limit:]:
        role = "User" if m.type == "human" else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines) or "(none)"


def planner_node(state: AgentState) -> dict:
    question = state["question"]
    prompt = PLANNER_PROMPT.format(
        history=format_history(state.get("messages", [])),
        question=question,
    )
    reply = get_llm("planner").invoke(prompt).content.strip()

    upper = reply.upper()
    if upper.startswith("SEARCH:"):
        return {"route": "search", "search_query": reply.split(":", 1)[1].strip() or question}
    if upper.startswith("OFF_TOPIC"):
        return {"route": "off_topic", "search_query": ""}
    if upper.startswith("CHAT"):
        return {"route": "chat", "search_query": ""}
    # Unexpected format: searching is the safe default
    return {"route": "search", "search_query": question}