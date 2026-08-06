"""
LangGraph Agent Graph Assembly
================================
This file wires all nodes into a complete, executable StateGraph.

Graph flow:
  START → router_node
            ├── "retrieve"      → retriever_node → answer_node → END
            ├── "web_search"    → answer_node → END   (web search coming in Milestone 4)
            └── "direct_answer" → answer_node → END
"""

from langgraph.graph import StateGraph, START, END
from src.agent.state import AgentState
from src.agent.nodes.router_node import router_node
from src.agent.nodes.retriever_node import retriever_node
from src.agent.nodes.answer_node import answer_node


def create_graph():
    """
    Builds and compiles the LangGraph StateGraph.
    Call this once and reuse the compiled graph.
    """

    # 1. Create the graph, typed to our AgentState
    graph = StateGraph(AgentState)

    # 2. Register all nodes
    # Each node is a function: (AgentState) -> dict (partial state update)
    graph.add_node("router",    router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("answer",    answer_node)

    # 3. Set entry point
    # START is a special LangGraph constant meaning "the very beginning"
    graph.add_edge(START, "router")

    # 4. Add conditional edges FROM the router
    # The router_node sets state["next_action"] to "retrieve", "web_search", or "direct_answer"
    # This conditional edge reads that value and routes accordingly
    graph.add_conditional_edges(
        source="router",      # which node is making the routing decision
        path=route_decision,  # function that reads state and returns the next node name
        path_map={            # map return values to actual node names
            "retrieve":       "retriever",
            "web_search":     "answer",     # web search node added in Milestone 4
            "direct_answer":  "answer"
        }
    )

    # 5. Add remaining edges (always-run, no conditions)
    graph.add_edge("retriever", "answer")  # after retrieval, always generate answer
    graph.add_edge("answer",    END)        # after answer, conversation turn is done

    # 6. Compile the graph
    # This validates the graph and prepares it for execution
    compiled_graph = graph.compile()

    print("[Graph] Agent graph compiled successfully.")
    return compiled_graph


def route_decision(state: AgentState) -> str:
    """
    Reads state["next_action"] and returns the next node to execute.
    This function is used by add_conditional_edges as the routing function.
    """
    return state.get("next_action", "retrieve")


# Create the graph instance (used by the rest of the app)
agent_graph = create_graph()