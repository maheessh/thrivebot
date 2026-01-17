"""
ThriveBot - FastAPI Main Application
Combines REST API with Slack Bot integration
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slack_bolt.adapter.fastapi import SlackRequestHandler
import structlog
import threading

from app.config import settings
from app.ingestion import GeminiEmbedder
from app.retrieval import FAISSVectorStore, RAGRetriever
from app.generation import GeminiLLM
from app.slack_bot import ThriveSlackBot

import logging

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer() if settings.is_development else structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, settings.log_level.upper(), logging.INFO)
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Global instances
embedder: Optional[GeminiEmbedder] = None
vector_store: Optional[FAISSVectorStore] = None
retriever: Optional[RAGRetriever] = None
llm: Optional[GeminiLLM] = None
slack_bot: Optional[ThriveSlackBot] = None
slack_handler: Optional[SlackRequestHandler] = None


def initialize_components():
    """Initialize all RAG components"""
    global embedder, vector_store, retriever, llm, slack_bot, slack_handler
    
    logger.info("Initializing ThriveBot components...")
    
    try:
        # Initialize embedder
        embedder = GeminiEmbedder(api_key=settings.gemini_api_key)
        
        # Initialize vector store and load existing index
        vector_store = FAISSVectorStore(
            dimension=embedder.dimension,
            store_path=settings.vector_store_path
        )
        vector_store.load()
        
        # Initialize retriever
        retriever = RAGRetriever(
            embedder=embedder,
            vector_store=vector_store,
            top_k=settings.top_k
        )
        
        # Initialize LLM
        llm = GeminiLLM(api_key=settings.gemini_api_key)
        
        # Initialize Slack bot
        slack_bot = ThriveSlackBot(
            retriever=retriever,
            llm=llm
        )
        slack_handler = SlackRequestHandler(slack_bot.get_bolt_app())
        
        logger.info(
            "ThriveBot initialized successfully",
            vector_store_size=vector_store.size
        )
        
    except Exception as e:
        logger.error("Failed to initialize components", error=str(e))
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for FastAPI"""
    # Startup
    initialize_components()
    logger.info("ThriveBot API started", host=settings.api_host, port=settings.api_port)
    
    yield
    
    # Shutdown
    logger.info("ThriveBot API shutting down...")


# Create FastAPI app
app = FastAPI(
    title="ThriveBot API",
    description="RAG-Based Slack FAQ Bot for Thrive Scholars",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    sources: list
    query: str


class HealthResponse(BaseModel):
    """Response model for health endpoint"""
    status: str
    vector_store_size: int
    environment: str


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to ThriveBot API! 🎓",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        vector_store_size=vector_store.size if vector_store else 0,
        environment=settings.app_env
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Query the knowledge base and get an AI-generated answer.
    
    This endpoint is useful for testing without Slack integration.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Retrieve context
        context, sources = retriever.retrieve_and_format(
            request.query,
            top_k=request.top_k
        )
        
        # Generate response
        answer = llm.generate(request.query, context)
        
        return QueryResponse(
            answer=answer,
            sources=[{
                "source": s.get("source", "Unknown"),
                "score": s.get("score", 0)
            } for s in sources],
            query=request.query
        )
        
    except Exception as e:
        logger.error("Query failed", error=str(e), query=request.query[:50])
        raise HTTPException(status_code=500, detail="Failed to process query")


@app.get("/stats", tags=["Admin"])
async def get_stats():
    """Get vector store statistics"""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    return vector_store.get_stats()


# Slack event endpoints
@app.post("/slack/events", tags=["Slack"])
async def slack_events(request):
    """Handle Slack events (for HTTP mode)"""
    return await slack_handler.handle(request)


@app.post("/slack/commands", tags=["Slack"])
async def slack_commands(request):
    """Handle Slack slash commands"""
    return await slack_handler.handle(request)


@app.post("/slack/interactions", tags=["Slack"])
async def slack_interactions(request):
    """Handle Slack interactions (buttons, modals, etc.)"""
    return await slack_handler.handle(request)


def run_socket_mode():
    """Run bot in Socket Mode (for development)"""
    initialize_components()
    slack_bot.start_socket_mode()


if __name__ == "__main__":
    import uvicorn
    import sys
    
    if "--socket-mode" in sys.argv:
        # Run in Socket Mode for development
        run_socket_mode()
    else:
        # Run FastAPI server
        uvicorn.run(
            "app.main:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=settings.is_development
        )
