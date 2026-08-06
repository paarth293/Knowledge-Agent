"""
Script 05: Test the LangGraph Agent (Multi-Turn Session Memory)
================================================================
What this does:
  1. Sends a first question under a specific `thread_id`.
  2. Sends a follow-up question under the SAME `thread_id`.
  3. Verifies that the agent remembers previous turns using the checkpointer.

Run with:
  python scripts/05_test_agent.py
"""

from langchain_core.messages import HumanMessage
from src.agent.graph import agent_graph


def send_message(question: str, user_id: str, conversation_id: str) -> None:
    """
    Sends a message to the agent using thread_id configuration.
    """
    print(f"\n{'='*60}")
    print(f"USER: {question}")
    print(f"{'='*60}")

    # State input for this turn
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "query": "",
        "retrieved_chunks": [],
        "sources": [],
        "user_id": user_id,
        "conversation_id": conversation_id,
        "next_action": None
    }

    # Pass `thread_id` inside config!
    # This tells LangGraph which session state to load and append to.
    config = {
        "configurable": {
            "thread_id": conversation_id
        }
    }

    # Run the graph
    final_state = agent_graph.invoke(initial_state, config=config)

    # Get latest response
    last_message = final_state["messages"][-1]
    print(f"\nAGENT: {last_message.content}")

    # Print citation sources if retrieval was used
    if final_state.get("sources"):
        print("\nSOURCES:")
        for i, s in enumerate(final_state["sources"]):
            print(f"  [{i+1}] {s['filename']} — Page {s['page_number']}")

    # Print total message count in session to verify memory persistence
    total_msgs = len(final_state["messages"])
    print(f"\n[Session Memory] Total messages stored in thread '{conversation_id}': {total_msgs}")


if __name__ == "__main__":
    USER_ID = "user-paarth-001"
    CONV_ID = "session-thread-999"

    print("--- TURN 1 ---")
    send_message("What are the key topics in the document?", USER_ID, CONV_ID)

    print("\n" + "#"*60 + "\n")

    print("--- TURN 2 (Follow-up relying on Turn 1 memory) ---")
    send_message("Can you summarize the second topic you just mentioned?", USER_ID, CONV_ID)