import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    # Conversation history. operator.add means nodes *append* to it.
    sources: list[str]       # files the answer actually cited
    messages: Annotated[list[BaseMessage], operator.add]
    question: str            # the user's latest question
    route: str               # "search" or "chat", set by the planner
    search_query: str        # standalone rewritten query for retrieval
    documents: list[dict]    # chunks returned by the retriever
    answer: str              # the final answer