"""
FastAPI Application Entry Point
================================
Registers all routes and starts the application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.db.database import engine, Base
from src.api.routes import documents, chat

# Create all database tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Personal Knowledge Agent API",
    description="AI-powered document Q&A system",
    version="1.0.0"
)

# Allow Streamlit frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Register routes
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(chat.router,      prefix="/api/chat",      tags=["Chat"])


@app.get("/")
def root():
    return {"status": "running", "message": "Personal Knowledge Agent API"}


# Run with: uvicorn src.api.main:app --reload --port 8000