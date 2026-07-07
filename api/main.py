import sys
import os
import time
import sqlite3
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag_pipeline import RAGPipeline

# ─────────────────────────────────────────
# DATABASE SETUP
# We log every query to SQLite for observability
# This is how you monitor a RAG system in production
# ─────────────────────────────────────────

DB_PATH = Path("logs/query_logs.db")
DB_PATH.parent.mkdir(exist_ok=True)

def init_db():
    """Create the logs table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            answer TEXT,
            num_sources INTEGER,
            latency_ms REAL,
            retrieved_count INTEGER
        )
    """)
    conn.commit()
    conn.close()

def log_query(question: str, answer: str, num_sources: int, latency_ms: float, retrieved_count: int):
    """Log a query and its result to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO query_logs 
        (timestamp, question, answer, num_sources, latency_ms, retrieved_count)
        VALUES (datetime('now'), ?, ?, ?, ?, ?)
    """, (question, answer, num_sources, latency_ms, retrieved_count))
    conn.commit()
    conn.close()


# ─────────────────────────────────────────
# LIFESPAN
# FastAPI lifespan loads the RAG pipeline once
# when the server starts — not on every request
# This is critical for performance since loading
# models takes ~10 seconds
# ─────────────────────────────────────────

pipeline = RAGPipeline()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline on startup, clean up on shutdown."""
    print("Starting up — loading RAG pipeline...")
    init_db()
    pipeline.load()
    print("Server ready.")
    yield
    print("Shutting down...")


# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────

app = FastAPI(
    title="F1 RAG Chatbot API",
    description="Retrieval Augmented Generation API for Formula 1 questions",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — allows Streamlit (running on port 8501) to call this API (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# Pydantic validates incoming and outgoing data
# ─────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

class SourceModel(BaseModel):
    source: str
    metadata: dict
    text_preview: str
    rerank_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceModel]
    retrieved_count: int
    latency_ms: float


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "running", "service": "F1 RAG Chatbot API"}


@app.get("/health")
def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline.is_loaded,
        "db_path": str(DB_PATH)
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main endpoint — takes a question, returns an answer with sources.
    
    Flow:
    1. Validate request (Pydantic does this automatically)
    2. Run RAG pipeline
    3. Log to SQLite
    4. Return response
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Time the full pipeline
    start = time.time()

    try:
        result = pipeline.query(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    latency_ms = (time.time() - start) * 1000

    # Log to SQLite
    log_query(
        question=request.question,
        answer=result["answer"],
        num_sources=len(result["sources"]),
        latency_ms=latency_ms,
        retrieved_count=result["retrieved_count"]
    )

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        retrieved_count=result["retrieved_count"],
        latency_ms=round(latency_ms, 2)
    )


@app.get("/logs")
def get_logs(limit: int = 20):
    """
    Return recent query logs.
    This powers the observability dashboard in the Streamlit UI.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        SELECT id, timestamp, question, answer, num_sources, latency_ms
        FROM query_logs
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()

    return {
        "logs": [
            {
                "id": r[0],
                "timestamp": r[1],
                "question": r[2],
                "answer": r[3],
                "num_sources": r[4],
                "latency_ms": r[5]
            }
            for r in rows
        ]
    }


@app.get("/stats")
def get_stats():
    """Return aggregate stats about system usage."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        SELECT 
            COUNT(*) as total_queries,
            AVG(latency_ms) as avg_latency_ms,
            MIN(latency_ms) as min_latency_ms,
            MAX(latency_ms) as max_latency_ms,
            AVG(num_sources) as avg_sources
        FROM query_logs
    """)
    row = cursor.fetchone()
    conn.close()

    return {
        "total_queries": row[0],
        "avg_latency_ms": round(row[1] or 0, 2),
        "min_latency_ms": round(row[2] or 0, 2),
        "max_latency_ms": round(row[3] or 0, 2),
        "avg_sources_used": round(row[4] or 0, 2)
    }