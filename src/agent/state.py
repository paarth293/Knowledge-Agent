"""
Agent State Definition
========================
The AgentState is the shared memory of the entire LangGraph agent.
Every node reads from this state and writes back partial updates.

Think of it as: "What does the agent need to know at any point in time?"
"""

from typing import Annotated, TypedDict, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The complete state of the Personal Knowledge Agent.
    Every field here is accessible by every node in the graph.

    Field explanations:
    - messages: Full conversation history. `add_messages` is a "reducer" —
      instead of replacing the list, it APPENDS new messages to existing ones.
      This is how LangGraph maintains conversation history automatically.

    - query: The current user question (extracted from latest message).

    - retrieved_chunks: The chunks retrieved from Qdrant for this turn.
      Cleared and repopulated each time retrieval runs.

    - sources: Citation info to show the user (which doc, which page).

    - user_id: Who is asking. Used to filter Qdrant search results.

    - conversation_id: Groups messages into sessions. Passed to checkpointer.

    - next_action: The routing decision made by the Router node.
      Values: "retrieve" | "web_search" | "direct_answer"
    """

    messages: Annotated[list[BaseMessage], add_messages]

    query: str

    retrieved_chunks: list[dict]

    sources: list[dict]

    user_id: str

    conversation_id: str

    next_action: Optional[str]