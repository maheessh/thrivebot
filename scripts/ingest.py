#!/usr/bin/env python
"""
Document Ingestion Script
Run this to ingest documents into the vector store
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings
from app.ingestion import DocumentLoader, TextChunker, GeminiEmbedder
from app.retrieval import FAISSVectorStore
import structlog

import logging

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()


def ingest_documents(
    documents_path: str = None,
    vector_store_path: str = None,
    clear_existing: bool = False
):
    """
    Ingest documents from the specified path into the vector store.
    
    Args:
        documents_path: Path to documents directory
        vector_store_path: Path to save vector store
        clear_existing: Whether to clear existing index before ingesting
    """
    documents_path = documents_path or settings.documents_path
    vector_store_path = vector_store_path or settings.vector_store_path
    
    logger.info("Starting document ingestion", documents_path=documents_path)
    
    # Step 1: Load documents
    logger.info("Step 1: Loading documents...")
    loader = DocumentLoader(documents_path)
    documents = loader.load_all()
    
    if not documents:
        logger.warning("No documents found! Add documents to the data/documents folder.")
        return
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Step 2: Chunk documents
    logger.info("Step 2: Chunking documents...")
    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_text(
            doc.content,
            doc.source,
            doc.metadata
        )
        all_chunks.extend(chunks)
    
    logger.info(f"Created {len(all_chunks)} chunks")
    
    # Step 3: Generate embeddings
    logger.info("Step 3: Generating embeddings with Gemini...")
    embedder = GeminiEmbedder(api_key=settings.gemini_api_key)
    
    chunk_texts = [chunk.content for chunk in all_chunks]
    embeddings = embedder.embed_texts(chunk_texts)
    
    logger.info(f"Generated {len(embeddings)} embeddings")
    
    # Step 4: Store in vector database
    logger.info("Step 4: Storing in vector database...")
    vector_store = FAISSVectorStore(
        dimension=embedder.dimension,
        store_path=vector_store_path
    )
    
    if not clear_existing:
        # Try to load existing index
        vector_store.load()
    
    # Prepare documents for storage
    docs_for_storage = [
        {
            "content": chunk.content,
            "source": chunk.source,
            "chunk_id": chunk.id,
            "metadata": chunk.metadata
        }
        for chunk in all_chunks
    ]
    
    vector_store.add(embeddings, docs_for_storage)
    
    # Step 5: Save index
    logger.info("Step 5: Saving vector store...")
    vector_store.save()
    
    logger.info(
        "✅ Ingestion complete!",
        total_documents=len(documents),
        total_chunks=len(all_chunks),
        vector_store_size=vector_store.size
    )
    
    return vector_store.get_stats()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents into ThriveBot")
    parser.add_argument(
        "--documents",
        type=str,
        default=None,
        help="Path to documents directory"
    )
    parser.add_argument(
        "--vector-store",
        type=str,
        default=None,
        help="Path to vector store"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index before ingesting"
    )
    
    args = parser.parse_args()
    
    try:
        stats = ingest_documents(
            documents_path=args.documents,
            vector_store_path=args.vector_store,
            clear_existing=args.clear
        )
        if stats:
            print("\nVector Store Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    except Exception as e:
        logger.error("Ingestion failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
