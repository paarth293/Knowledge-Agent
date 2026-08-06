"""
Router Node
============
Decides what to do with the user's message.

Inputs from state:  messages (to read the latest user message)
Outputs to state:   next_action, query
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from src.agent.state import AgentState

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


def router_node(state: AgentState) -> dict:
    """
    Classifies the user's question and sets next_action.

    Returns partial state update:
    {
        "query": "the user's question",
        "next_action": "retrieve" | "web_search" | "direct_answer"
    }
    """

    # Get the latest user message
    latest_message = state["messages"][-1]
    query = latest_message.content

    # Ask LLM to classify the question
    classification_prompt = f"""Classify the following question into exactly one category.
Return ONLY the category name, nothing else.

Categories:
- retrieve: The question is about specific documents, notes, PDFs, or uploaded files.
- web_search: The question requires real-time information (news, current events, prices, weather).
- direct_answer: The question is general knowledge that doesn't need documents or web search.

Question: {query}

Category:"""

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    decision = response.content.strip().lower()

    # Validate decision (default to retrieve if LLM gives unexpected output)
    if decision not in ["retrieve", "web_search", "direct_answer"]:
        decision = "retrieve"

    print(f"[Router] Query: '{query[:60]}...' → Decision: {decision}")

    return {
        "query": query,
        "next_action": decision
    }