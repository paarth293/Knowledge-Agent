"""
Script 02: Embed Chunks and Store in Qdrant
=============================================
What this does:
  1. Parses and chunks the PDF (reuses logic from Script 01)
  2. Loads BAAI/bge-small-en-v1.5 embedding model locally
  3. Converts every chunk into a 384-dimensional vector
  4. Creates a local Qdrant collection (file-based, no Docker)
  5. Stores all chunk vectors + metadata as Qdrant Points

Run with:
  python scripts/02_embed_and_store.py
"""

import uuid
import re
import pdfplumber
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

PDF_PATH         = "data/sample.pdf"
QDRANT_PATH      = "./qdrant_db"              # local disk storage
COLLECTION_NAME  = "document_chunks"
EMBEDDING_MODEL  = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM    = 384                        # output size of bge-small
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50
USER_ID          = "user-paarth-001"          # hardcoded for testing
DOCUMENT_ID      = str(uuid.uuid4())          # unique ID for this document


# ─────────────────────────────────────────
# HELPER: Parse + Clean + Chunk
# (Same as Script 01 — copied so this script is self-contained)
# ─────────────────────────────────────────

def parse_pdf(file_path):
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 20:
                pages.append({"page_number": i + 1, "text": text})
    return pages

def clean_text(text):
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = []
    idx = 0
    for page in pages:
        for chunk_text in splitter.split_text(clean_text(page["text"])):
            chunks.append({
                "text": chunk_text,
                "page_number": page["page_number"],
                "chunk_index": idx
            })
            idx += 1
    return chunks


# ─────────────────────────────────────────
# STEP 1: LOAD EMBEDDING MODEL
# ─────────────────────────────────────────

print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
print("[INFO] First run downloads ~90MB. Cached locally after that.")
embed_model = SentenceTransformer(EMBEDDING_MODEL)
print("[INFO] Embedding model loaded.")


# ─────────────────────────────────────────
# STEP 2: SETUP LOCAL QDRANT
# ─────────────────────────────────────────

# QdrantClient(path=...) creates a file-based local database.
# Your vectors are stored in ./qdrant_db/ on disk.
# Same API as Qdrant Cloud — zero code changes needed when upgrading.

qdrant = QdrantClient(path=QDRANT_PATH)
print(f"[INFO] Connected to local Qdrant at: {QDRANT_PATH}")

# Create collection if it doesn't exist
existing_collections = [c.name for c in qdrant.get_collections().collections]

if COLLECTION_NAME not in existing_collections:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,      # must match the embedding model output exactly
            distance=Distance.COSINE # use cosine similarity for bge models
        )
    )
    print(f"[INFO] Created collection: {COLLECTION_NAME}")
else:
    print(f"[INFO] Collection '{COLLECTION_NAME}' already exists. Using existing.")


# ─────────────────────────────────────────
# STEP 3: PARSE → CHUNK → EMBED → STORE
# ─────────────────────────────────────────

# 3a. Parse and chunk the PDF
print(f"\n[INFO] Parsing PDF: {PDF_PATH}")
pages = parse_pdf(PDF_PATH)
chunks = chunk_pages(pages)
print(f"[INFO] Created {len(chunks)} chunks from {len(pages)} pages.")

# 3b. Embed all chunks in one batch (much faster than one by one)
print(f"\n[INFO] Embedding {len(chunks)} chunks... (this takes ~30-60 seconds)")
chunk_texts = [c["text"] for c in chunks]
vectors = embed_model.encode(
    chunk_texts,
    batch_size=32,          # process 32 chunks at a time
    show_progress_bar=True
)
print(f"[INFO] Embedding complete. Vector shape: {vectors.shape}")
# vectors.shape should be: (num_chunks, 384)

# 3c. Build Qdrant Points
# Each Point = one searchable item in Qdrant
# id: unique identifier for this point
# vector: the 384 numbers representing chunk meaning
# payload: metadata we need to return with search results
points = []
for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
    points.append(
        PointStruct(
            id=str(uuid.uuid4()),       # unique ID for this chunk
            vector=vector.tolist(),     # convert numpy array to Python list
            payload={
                "text":         chunk["text"],
                "page_number":  chunk["page_number"],
                "chunk_index":  chunk["chunk_index"],
                "document_id":  DOCUMENT_ID,
                "user_id":      USER_ID,
                "filename":     PDF_PATH.split("/")[-1]
            }
        )
    )

# 3d. Upload to Qdrant
print(f"\n[INFO] Storing {len(points)} points in Qdrant...")
qdrant.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)
print("[INFO] All points stored successfully.")


# ─────────────────────────────────────────
# VERIFICATION: Check collection stats
# ─────────────────────────────────────────

info = qdrant.get_collection(COLLECTION_NAME)
print("\n" + "="*60)
print("INDEXING COMPLETE")
print("="*60)
print(f"Document ID    : {DOCUMENT_ID}")
print(f"Total points   : {info.points_count}")
print(f"Vector size    : {info.config.params.vectors.size}")
print(f"Distance metric: {info.config.params.vectors.distance}")
print("="*60)
print("\nYour chunks are now stored in Qdrant and ready for semantic search.")