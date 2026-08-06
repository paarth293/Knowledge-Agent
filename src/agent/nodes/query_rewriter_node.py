"""
Query Rewriter Node
====================
Rewrites follow-up questions into standalone queries using conversation history.

Inputs from state:  messages (conversation history), query
Outputs to state:   query (the rewritten version)
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from src.agent.state import AgentState

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0,
               api_key=os.getenv("GROQ_API_KEY"))


def query_rewriter_node(state: AgentState) -> dict:
    """
    If there is prior conversation history, rewrites the current query
    to be a standalone question that doesn't require reading the history.

    Example:
        History: "Q: What are the main topics? A: Data structures, algorithms..."
        Current:  "Tell me more about the second one."
        Rewritten: "Tell me more about algorithms as described in the document."
    """

    messages = state["messages"]
    query    = state["query"]

    # If this is the first message, no rewriting needed
    if len(messages) <= 1:
        return {"query": query}

    # Build history string (last 4 message pairs for context)
    history_messages = messages[:-1]  # all except the latest
    history_str = "\n".join([
        f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content[:200]}"
        for msg in history_messages[-4:]  # last 4 messages
    ])

    rewrite_prompt = f"""Given this conversation history and the follow-up question,
rewrite the follow-up as a standalone question that contains all necessary context.
If the question is already standalone, return it unchanged.
Return ONLY the rewritten question, nothing else.

Conversation history:
{history_str}

Follow-up question: {query}

Standalone question:"""

    response = llm.invoke([HumanMessage(content=rewrite_prompt)])
    rewritten_query = response.content.strip()

    if rewritten_query != query:
        print(f"[QueryRewriter] Rewrote: '{query[:50]}' → '{rewritten_query[:50]}'")

    return {"query": rewritten_query}