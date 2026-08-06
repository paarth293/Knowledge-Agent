"""
Script 05: Test the LangGraph Agent
=====================================
Runs the assembled LangGraph agent with a real question.

Run with:
  python scripts/05_test_agent.py
"""

from langchain_core.messages import HumanMessage
from src.agent.graph import agent_graph


def run_agent(question: str, user_id: str, conversation_id: str) -> None:
    """Runs one turn of the agent and prints the result."""

    print(f"\n{'='*60}")
    print(f"USER: {question}")
    print(f"{'='*60}")

    # Initial state — this is what the agent starts with
    initial_state = {
        "messages":          [HumanMessage(content=question)],
        "query":             "",
        "retrieved_chunks":  [],
        "sources":           [],
        "user_id":           user_id,
        "conversation_id":   conversation_id,
        "next_action":       None
    }

    # Run the graph
    # The graph processes: router → retriever (maybe) → answer → END
    final_state = agent_graph.invoke(initial_state)

    # Get the last message (the AI's answer)
    last_message = final_state["messages"][-1]
    print(f"\nAGENT: {last_message.content}")

    # Show sources if retrieval happened
    if final_state.get("sources"):
        print("\nSOURCES:")
        for i, s in enumerate(final_state["sources"]):
            print(f"  [{i+1}] {s['filename']} — Page {s['page_number']}")


if __name__ == "__main__":
    USER_ID = "user-paarth-001"
    CONV_ID = "test-conversation-001"

    # Test 1: Document question (should trigger retrieval)
    run_agent("What is this document about?", USER_ID, CONV_ID)

    # Test 2: General question (should skip retrieval)
    run_agent("What is 2 + 2?", USER_ID, CONV_ID)