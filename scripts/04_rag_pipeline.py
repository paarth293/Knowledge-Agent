"""
Script 04: Complete RAG Pipeline
==================================
What this does:
  1. Takes a user question
  2. Retrieves top-K most relevant chunks from Qdrant (Script 03 logic)
  3. Builds a grounded prompt with retrieved context
  4. Calls Groq's Llama 3.3 70B to generate the answer
  5. Streams the answer word-by-word
  6. Shows which documents the answer came from (citations)

This is the complete Naive RAG pipeline.

Run with:
  python scripts/04_rag_pipeline.py
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Load environment variables from .env file
load_dotenv()


# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

QDRANT_PATH     = "./qdrant_db"
COLLECTION_NAME = "document_chunks"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
USER_ID         = "user-paarth-001"
TOP_K           = 5


# ─────────────────────────────────────────
# SETUP CLIENTS
# ─────────────────────────────────────────

print("[INFO] Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)

print("[INFO] Connecting to Qdrant...")
qdrant = QdrantClient(path=QDRANT_PATH)

print("[INFO] Initializing Groq LLM...")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,               # 0 = deterministic, less hallucination
    api_key=os.getenv("GROQ_API_KEY")
)


# ─────────────────────────────────────────
# STEP 1: RETRIEVE RELEVANT CHUNKS
# ─────────────────────────────────────────

def retrieve_chunks(question: str) -> list[dict]:
    """Embed question, search Qdrant, return top-K chunks."""
    question_vector = embed_model.encode(question).tolist()

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=question_vector,
        query_filter=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=USER_ID))]
        ),
        limit=TOP_K,
        with_payload=True
    )

    return [
        {
            "text":        r.payload["text"],
            "page_number": r.payload["page_number"],
            "filename":    r.payload["filename"],
            "score":       round(r.score, 4)
        }
        for r in results
    ]


# ─────────────────────────────────────────
# STEP 2: BUILD GROUNDED PROMPT
# ─────────────────────────────────────────

def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Constructs the prompt that instructs the LLM to answer
    ONLY from the retrieved context.

    This prompt structure is critical:
    1. System instructions come first (establishes LLM behavior)
    2. Context section with source labels (so LLM can cite)
    3. The user's actual question last

    The phrase "ONLY the information provided" is the key hallucination guard.
    If the answer is not in context, the LLM must say "I don't know."
    """

    # Build context block with source labels
    context_block = ""
    for i, chunk in enumerate(chunks):
        context_block += (
            f"[Source {i+1}: {chunk['filename']}, Page {chunk['page_number']}]\n"
            f"{chunk['text']}\n\n"
        )

    prompt = f"""You are a precise, helpful assistant that answers questions based on provided documents.

STRICT RULES:
1. Answer using ONLY the information in the Context section below.
2. If the context does not contain enough information, say exactly: "I don't have enough information in the provided documents to answer this."
3. Always cite which Source (by number) your answer comes from.
4. Do not add outside knowledge. Do not guess.

Context:
{context_block}

Question: {question}

Answer:"""

    return prompt


# ─────────────────────────────────────────
# STEP 3: GENERATE STREAMED ANSWER
# ─────────────────────────────────────────

def answer_question(question: str) -> None:
    """
    Full RAG pipeline:
    1. Retrieve chunks
    2. Build grounded prompt
    3. Stream answer from Groq LLM
    4. Display citations
    """

    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")

    # Step 1: Retrieve
    chunks = retrieve_chunks(question)

    if not chunks:
        print("[ERROR] No relevant chunks found. Make sure you ran Script 02 first.")
        return

    print(f"\n[INFO] Retrieved {len(chunks)} chunks.")
    print(f"[INFO] Highest similarity: {chunks[0]['score']}")

    # Step 2: Build prompt
    prompt = build_prompt(question, chunks)

    # Step 3: Stream answer from Groq
    print("\n--- ANSWER (streaming) ---\n")
    full_answer = ""

    for token in llm.stream([HumanMessage(content=prompt)]):
        print(token.content, end="", flush=True)
        full_answer += token.content

    # Step 4: Print citations
    print("\n\n--- SOURCES ---")
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}] {chunk['filename']} — Page {chunk['page_number']} "
              f"(similarity: {chunk['score']})")


# ─────────────────────────────────────────
# MAIN — Interactive Q&A loop
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("\n🤖 Personal Knowledge Agent — RAG Pipeline")
    print("Type your question or 'quit' to exit\n")

    while True:
        user_question = input("Your question: ").strip()

        if user_question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_question:
            print("[WARN] Please enter a question.")
            continue

        answer_question(user_question)
        print("\n" + "-"*60 + "\n")