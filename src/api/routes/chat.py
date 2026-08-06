"""
Chat Route
===========
POST /api/chat — Sends a message to the agent, returns streaming SSE response.
"""

import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from src.agent.graph import agent_graph

router = APIRouter()


class ChatRequest(BaseModel):
    message:         str
    conversation_id: str = "default-conversation"


@router.post("")
async def chat(request: ChatRequest):
    """
    Receives a message, runs the agent, returns a Server-Sent Events stream.

    The frontend connects to this endpoint with EventSource API and receives
    tokens one by one as they are generated, creating the streaming text effect.
    """

    user_id = "user-paarth-001"  # hardcoded — replace with auth later

    initial_state = {
        "messages":         [HumanMessage(content=request.message)],
        "query":            "",
        "retrieved_chunks": [],
        "sources":          [],
        "user_id":          user_id,
        "conversation_id":  request.conversation_id,
        "next_action":      None
    }

    config = {"configurable": {"thread_id": request.conversation_id}}

    async def event_generator():
        """Yields SSE-formatted events for the frontend."""
        try:
            # Run the agent (blocking for now — streaming in v2)
            final_state = agent_graph.invoke(initial_state, config=config)

            # Get the agent's answer
            answer = final_state["messages"][-1].content
            sources = final_state.get("sources", [])

            # Stream the answer word-by-word
            words = answer.split(" ")
            for word in words:
                yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"

            # Send sources after streaming is complete
            yield f"data: {json.dumps({'type': 'done', 'sources': sources})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")