"""
Answer Generator Node
======================
Builds the grounded prompt from retrieved chunks and generates the LLM answer.

Inputs from state:  query, retrieved_chunks, sources, messages (conversation history)
Outputs to state:   messages (appends AI response)
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.state import AgentState

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


def answer_node(state: AgentState) -> dict:
    """
    Builds and sends the grounded RAG prompt to the LLM.
    Appends the AI's response to the messages list.

    Returns partial state update:
    {
        "messages": [AIMessage(content="...")]
    }
    """

    query            = state["query"]
    retrieved_chunks = state.get("retrieved_chunks", [])
    sources          = state.get("sources", [])

    # Build context block from retrieved chunks
    if retrieved_chunks:
        context_block = ""
        for i, (chunk_text, source) in enumerate(zip(retrieved_chunks, sources)):
            context_block += (
                f"[Source {i+1}: {source['filename']}, Page {source['page_number']}]\n"
                f"{chunk_text}\n\n"
            )

        prompt = f"""You are a precise assistant. Answer using ONLY the context below.
If the context is insufficient, say: "I don't have enough information in the provided documents."
Always cite the Source number.

Context:
{context_block}

Question: {query}

Answer:"""
    else:
        # No chunks retrieved — answer from general knowledge
        prompt = f"Answer this general question clearly and concisely:\n\n{query}"

    # Get answer from Groq
    response = llm.invoke([HumanMessage(content=prompt)])
    answer_text = response.content

    print(f"[Answer] Generated response ({len(answer_text)} chars)")

    # Append AI message to conversation history
    # The add_messages reducer in AgentState will merge this with existing messages
    return {
        "messages": [AIMessage(content=answer_text)]
    }