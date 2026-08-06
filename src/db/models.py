"""
SQLAlchemy ORM Models
=======================
Defines all database tables as Python classes.
Each class = one table in the database.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from src.db.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class Document(Base):
    __tablename__ = "documents"

    id          = Column(String, primary_key=True, default=generate_uuid)
    user_id     = Column(String, nullable=False, index=True)
    filename    = Column(String, nullable=False)
    file_size   = Column(Integer)
    file_type   = Column(String)
    status      = Column(String, default="processing")  # processing | ready | failed
    chunk_count = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete")


class Chunk(Base):
    __tablename__ = "chunks"

    id          = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    text        = Column(Text, nullable=False)
    page_number = Column(Integer)
    qdrant_id   = Column(String)   # links to the Qdrant point ID
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    document = relationship("Document", back_populates="chunks")


class Conversation(Base):
    __tablename__ = "conversations"

    id         = Column(String, primary_key=True, default=generate_uuid)
    user_id    = Column(String, nullable=False, index=True)
    title      = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    messages = relationship("Message", back_populates="conversation", cascade="all, delete")


class Message(Base):
    __tablename__ = "messages"

    id              = Column(String, primary_key=True, default=generate_uuid)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False, index=True)
    role            = Column(String, nullable=False)   # "user" or "assistant"
    content         = Column(Text, nullable=False)
    created_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    conversation = relationship("Conversation", back_populates="messages")