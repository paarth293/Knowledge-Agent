"""
Document Management Routes
============================
POST /api/documents      — Upload a PDF
GET  /api/documents      — List user's documents
DELETE /api/documents/{id} — Delete a document
"""

import os
import uuid
import re
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber

from src.db.database import get_db
from src.db.models import Document, Chunk

router = APIRouter()

# Initialize clients once at module level
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
qdrant      = QdrantClient(path="./qdrant_db")
COLLECTION  = "document_chunks"
UPLOAD_DIR  = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ─── Helper: full ingestion pipeline ───

def ingest_document(doc_id: str, file_path: str, user_id: str, db: Session):
    """Runs the full parse → chunk → embed → store pipeline in the background."""
    try:
        # Parse PDF
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 20:
                    pages.append({"page_number": i+1, "text": text})

        # Chunk
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = []
        idx = 0
        for page in pages:
            text = re.sub(r'-\n', '', page["text"])
            text = re.sub(r'[ \t]+', ' ', text).strip()
            for chunk_text in splitter.split_text(text):
                chunks.append({"text": chunk_text, "page_number": page["page_number"], "idx": idx})
                idx += 1

        # Embed
        vectors = embed_model.encode([c["text"] for c in chunks], batch_size=32)

        # Store in Qdrant + PostgreSQL
        points = []
        filename = os.path.basename(file_path)
        for chunk, vector in zip(chunks, vectors):
            qdrant_id = str(uuid.uuid4())
            points.append(PointStruct(
                id=qdrant_id,
                vector=vector.tolist(),
                payload={
                    "text": chunk["text"], "page_number": chunk["page_number"],
                    "document_id": doc_id, "user_id": user_id, "filename": filename
                }
            ))
            db.add(Chunk(
                document_id=doc_id, chunk_index=chunk["idx"],
                text=chunk["text"], page_number=chunk["page_number"], qdrant_id=qdrant_id
            ))

        qdrant.upsert(collection_name=COLLECTION, points=points)

        # Update document status
        doc = db.query(Document).filter(Document.id == doc_id).first()
        doc.status = "ready"
        doc.chunk_count = len(chunks)
        db.commit()

    except Exception as e:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.status = "failed"
            db.commit()
        print(f"[ERROR] Ingestion failed: {e}")


# ─── Endpoints ───

@router.post("")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """Upload a PDF. Returns immediately. Processing happens in background."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    user_id = "user-paarth-001"  # hardcoded — replace with auth later
    doc_id  = str(uuid.uuid4())

    # Save file to disk
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Create document record in DB
    doc = Document(
        id=doc_id, user_id=user_id,
        filename=file.filename, file_type="pdf",
        file_size=os.path.getsize(file_path), status="processing"
    )
    db.add(doc)
    db.commit()

    # Process in background (user doesn't wait)
    background_tasks.add_task(ingest_document, doc_id, file_path, user_id, db)

    return {"document_id": doc_id, "status": "processing", "filename": file.filename}


@router.get("")
def list_documents(db: Session = Depends(get_db)):
    """List all documents for the current user."""
    user_id = "user-paarth-001"
    docs = db.query(Document).filter(Document.user_id == user_id).all()
    return [
        {"id": d.id, "filename": d.filename, "status": d.status,
         "chunk_count": d.chunk_count, "uploaded_at": d.uploaded_at}
        for d in docs
    ]


@router.delete("/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Delete a document and all its chunks from DB and Qdrant."""
    user_id = "user-paarth-001"
    doc = db.query(Document).filter(
        Document.id == document_id, Document.user_id == user_id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Delete from Qdrant
    qdrant.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        )
    )

    # Delete from DB (cascades to chunks)
    db.delete(doc)
    db.commit()
    return {"success": True, "deleted_document_id": document_id}