"""
LangGraph Agent Graph Assembly (with Checkpointing Memory)
===========================================================
This file wires all nodes into a complete, executable StateGraph
and attaches a MemorySaver checkpointer for short-term session memory.

Graph flow:
  START → router_node
            ├── "retrieve"      → retriever_node → answer_node → END
            ├── "web_search"    → answer_node → END
            └── "direct_answer" → answer_node → END
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent.nodes.router_node import router_node
from src.agent.nodes.retriever_node import retriever_node
from src.agent.nodes.answer_node import answer_node


def route_decision(state: AgentState) -> str:
    """
    Reads state["next_action"] set by router_node and returns
    the next node name to execute.
    """
    return state.get("next_action", "retrieve")


def create_graph():
    """
    Builds and compiles the LangGraph StateGraph with Checkpointing.
    """

    # 1. Initialize the StateGraph with our custom AgentState schema
    graph = StateGraph(AgentState)

    # 2. Register all nodes (functions)
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("answer", answer_node)

    # 3. Set entry point
    graph.add_edge(START, "router")

    # 4. Add conditional routing from the Router node
    graph.add_conditional_edges(
        source="router",
        path=route_decision,
        path_map={
            "retrieve": "retriever",
            "web_search": "answer",
            "direct_answer": "answer"
        }
    )

    # 5. Connect remaining nodes to completion
    graph.add_edge("retriever", "answer")
    graph.add_edge("answer", END)

    # 6. Initialize Checkpointer for short-term in-memory conversation state
    # MemorySaver associates graph state with a `thread_id` session key
    checkpointer = MemorySaver()

    # 7. Compile graph WITH the checkpointer attached
    compiled_graph = graph.compile(checkpointer=checkpointer)

    print("[Graph] Compiled graph successfully with MemorySaver checkpointer.")
    return compiled_graph


# Single compiled graph instance exported for the application
agent_graph = create_graph()