"""
Retriever Node
===============
Queries Qdrant for chunks relevant to the user's query.

Inputs from state:  query, user_id
Outputs to state:   retrieved_chunks, sources
"""

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.agent.state import AgentState

QDRANT_PATH     = "./qdrant_db"
COLLECTION_NAME = "document_chunks"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K           = 5

# Load once at module level (not inside the function — avoid reloading on every call)
embed_model = SentenceTransformer(EMBEDDING_MODEL)
qdrant      = QdrantClient(path=QDRANT_PATH)


def retriever_node(state: AgentState) -> dict:
    """
    Embeds the query, searches Qdrant, updates retrieved_chunks and sources.

    Returns partial state update:
    {
        "retrieved_chunks": [...],
        "sources": [...]
    }
    """

    query   = state["query"]
    user_id = state["user_id"]

    # Embed the query
    query_vector = embed_model.encode(query).tolist()

    # Search Qdrant (filtered by user_id for multi-tenancy security)
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        ),
        limit=TOP_K,
        with_payload=True
    )

    if not results:
        print("[Retriever] No chunks found for this query.")
        return {"retrieved_chunks": [], "sources": []}

    # Build retrieved chunks list
    retrieved_chunks = [r.payload["text"] for r in results]

    # Build sources list (for citations shown to the user)
    sources = [
        {
            "filename":    r.payload["filename"],
            "page_number": r.payload["page_number"],
            "score":       round(r.score, 4)
        }
        for r in results
    ]

    print(f"[Retriever] Found {len(retrieved_chunks)} chunks. "
          f"Top score: {sources[0]['score']}")

    return {
        "retrieved_chunks": retrieved_chunks,
        "sources": sources
    }