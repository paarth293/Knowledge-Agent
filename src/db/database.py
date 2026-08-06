"""
Database Connection Setup
===========================
Creates the SQLite/PostgreSQL engine and session factory.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# SQLite for development (file-based, no server needed)
# Switch to PostgreSQL URL for production:
# "postgresql://user:password@localhost:5432/knowledge_agent"
DATABASE_URL = "sqlite:///./knowledge_agent.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # needed for SQLite with FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: provides a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()