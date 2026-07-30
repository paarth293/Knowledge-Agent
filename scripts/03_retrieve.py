"""
Script 03: Semantic Retrieval from Qdrant
==========================================
What this does:
  1. Takes a question from the user
  2. Embeds the question using the same model used during indexing
  3. Searches Qdrant for the K most similar chunks
  4. Prints results with similarity scores and source info

CRITICAL RULE: You MUST use the same embedding model that was used in Script 02.
If you use a different model, the vector spaces won't match and results are garbage.

Run with:
  python scripts/03_retrieve.py
"""

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


# ─────────────────────────────────────────
# CONFIGURATION — must match Script 02
# ─────────────────────────────────────────

QDRANT_PATH     = "./qdrant_db"
COLLECTION_NAME = "document_chunks"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # MUST be same as Script 02
TOP_K           = 5                            # How many chunks to retrieve
USER_ID         = "user-paarth-001"           # MUST match what was used in indexing


# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────

print("[INFO] Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)

print("[INFO] Connecting to local Qdrant...")
qdrant = QdrantClient(path=QDRANT_PATH)


# ─────────────────────────────────────────
# RETRIEVAL FUNCTION
# ─────────────────────────────────────────

def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Takes a question, finds the most semantically similar document chunks.

    Steps:
      1. Embed the question → get a 384-dimensional vector
      2. Query Qdrant with this vector
      3. Filter by user_id (security: only search this user's documents)
      4. Return top-K results sorted by similarity score (highest first)
    """

    # Step 1: Embed the question
    print(f"\n[INFO] Embedding question...")
    question_vector = embed_model.encode(question).tolist()
    # question_vector is now: [0.031, -0.201, 0.654, ...] — 384 numbers

    # Step 2: Search Qdrant
    # Qdrant uses HNSW index to find approximate nearest neighbors
    # This is milliseconds even with millions of vectors
    print(f"[INFO] Searching Qdrant for top {top_k} similar chunks...")

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=question_vector,

        # Security filter: ONLY search documents belonging to this user
        # Without this, User A's query could return User B's documents
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=USER_ID)
                )
            ]
        ),

        limit=top_k,
        with_payload=True   # include metadata (text, page_number, filename) in results
    )

    # Convert results to a clean list of dicts
    retrieved_chunks = []
    for result in results:
        retrieved_chunks.append({
            "text":        result.payload["text"],
            "page_number": result.payload["page_number"],
            "filename":    result.payload["filename"],
            "score":       round(result.score, 4)  # cosine similarity: 0 to 1
        })

    return retrieved_chunks


# ─────────────────────────────────────────
# MAIN — Test with sample questions
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Test questions — change these to match your PDF content
    test_questions = [
        "What is the main topic of this document?",
        "What are the key concepts explained?",
        "Summarize the introduction"
    ]

    for question in test_questions:
        print("\n" + "="*60)
        print(f"QUESTION: {question}")
        print("="*60)

        chunks = retrieve(question)

        if not chunks:
            print("[WARN] No results found. Did you run Script 02 first?")
            continue

        for i, chunk in enumerate(chunks):
            print(f"\n[Result {i+1}] Score: {chunk['score']} | "
                  f"Page: {chunk['page_number']} | File: {chunk['filename']}")
            print(f"Text: {chunk['text'][:300]}...")

        print(f"\n[INFO] Highest similarity score: {chunks[0]['score']}")
        print("[INFO] Score guide: >0.8 = very relevant, 0.5-0.8 = somewhat relevant, <0.5 = likely irrelevant")