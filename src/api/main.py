"""
TaxGPT Financial Chatbot — FastAPI Application

Endpoints:
  GET  /              → Chat UI (HTML)
  GET  /api/health    → System health check
  POST /api/chat      → Main chat endpoint
  POST /api/ingest    → Trigger re-ingestion (admin)
  GET  /docs          → Auto-generated OpenAPI docs
"""

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.api.models import ChatRequest, ChatResponse, HealthResponse, IngestRequest, SourceReference
from src.api.llm import LLMClient
from src.retrieval.vector_store import VectorStore
from src.retrieval.graph_store import GraphStore
from src.retrieval.hybrid_retriever import HybridRetriever

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Globals (initialized on startup) ───────────────────────────────────────
vector_store: VectorStore = None
graph_store: GraphStore = None
retriever: HybridRetriever = None
llm: LLMClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup, clean up on shutdown."""
    global vector_store, graph_store, retriever, llm

    logger.info("Initializing TaxGPT components...")

    # Vector store
    vector_store = VectorStore(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        collection_name=os.getenv("CHROMA_COLLECTION", "financial_data"),
    )

    # Graph store
    graph_store = GraphStore(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password123"),
    )

    # Hybrid retriever
    retriever = HybridRetriever(vector_store, graph_store)

    # LLM
    llm = LLMClient(
        model=os.getenv("OLLAMA_MODEL", "llama3"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    logger.info("TaxGPT ready.")
    yield

    # Cleanup
    if graph_store:
        graph_store.close()
    logger.info("TaxGPT shutdown complete.")


app = FastAPI(
    title="TaxGPT Financial Chatbot",
    description="Q&A chatbot over financial CSV, PDF, and PPT data using hybrid RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat UI."""
    html_path = Path("frontend/index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>TaxGPT API</h1><p>Visit <a href='/docs'>/docs</a></p>")


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check — reports component status."""
    try:
        v_count = vector_store.count() if vector_store else 0
        g_count = graph_store.node_count() if graph_store else 0
    except Exception as e:
        logger.warning(f"Health check error: {e}")
        v_count, g_count = -1, -1

    return HealthResponse(
        status="ok",
        vector_docs=v_count,
        graph_nodes=g_count,
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3"),
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    
    Retrieves relevant context from ChromaDB + Neo4j,
    then generates an answer via Ollama.
    """
    if not retriever or not llm:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Retrieve context
    context = retriever.retrieve(request.message, n_vector=request.n_results)
    graph_used = "[Graph Database" in context

    # Count context chunks
    context_chunks = context.count("[") - (1 if graph_used else 0)

    # Generate answer
    answer = llm.chat(request.message, context)

    # Extract source references from vector hits for transparency
    vector_hits = vector_store.query(request.message, n_results=request.n_results)
    sources = [
        SourceReference(
            source=h["metadata"].get("source", "unknown"),
            page=str(h["metadata"].get("page", "")) or None,
            slide=h["metadata"].get("slide"),
            score=h["score"],
            source_type=h["metadata"].get("source_type"),
        )
        for h in vector_hits[:5]
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        graph_used=graph_used,
        context_chunks=context_chunks,
    )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint — returns tokens via Server-Sent Events.
    Connect with EventSource in the frontend for real-time output.
    """
    if not retriever or not llm:
        raise HTTPException(status_code=503, detail="Service not initialized")

    context = retriever.retrieve(request.message, n_vector=request.n_results)

    async def generate():
        for token in llm.chat_stream(request.message, context):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/ingest")
async def trigger_ingest(request: IngestRequest):
    """
    Admin endpoint: re-run the ingestion pipeline.
    WARNING: set reset=true to wipe ChromaDB before re-ingesting.
    """
    import subprocess
    cmd = ["python", "scripts/ingest.py", "--data-dir", request.data_dir]
    if request.reset:
        cmd.append("--reset")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)

    return {"status": "ingestion complete", "output": result.stdout[-2000:]}
