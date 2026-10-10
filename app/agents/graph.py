from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents.nodes.planner import planner_node
from app.agents.nodes.responder import responder_node
from app.agents.nodes.retriever import retriever_node
from app.agents.state import AgentState


def route_after_planner(state: AgentState) -> str:
    """Conditional edge: search the documents, or answer directly."""
    return "retriever" if state.get("route") == "search" else "responder"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("responder", responder_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"retriever": "retriever", "responder": "responder"},
    )
    graph.add_edge("retriever", "responder")
    graph.add_edge("responder", END)

    # MemorySaver keeps conversation history per thread_id (in memory only)
    return graph.compile(checkpointer=MemorySaver())


rag_agent = build_graph()


def ask(question: str, thread_id: str = "default") -> dict:
    """Run one turn of the conversation and return the final state."""
    return rag_agent.invoke(
        {"question": question},
        config={"configurable": {"thread_id": thread_id}},
    )