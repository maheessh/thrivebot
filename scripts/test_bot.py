#!/usr/bin/env python
"""
Test Script - Test ThriveBot locally without Slack
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings
from app.ingestion import GeminiEmbedder
from app.retrieval import FAISSVectorStore, RAGRetriever
from app.generation import GeminiLLM
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


def test_query(query: str):
    """Test a query against the knowledge base"""
    
    logger.info("Initializing components...")
    
    # Initialize embedder
    embedder = GeminiEmbedder(api_key=settings.gemini_api_key)
    
    # Load vector store
    vector_store = FAISSVectorStore(
        dimension=embedder.dimension,
        store_path=settings.vector_store_path
    )
    
    if not vector_store.load():
        logger.error("No vector store found! Run 'python scripts/ingest.py' first.")
        return
    
    logger.info(f"Vector store loaded with {vector_store.size} documents")
    
    # Initialize retriever
    retriever = RAGRetriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=settings.top_k
    )
    
    # Initialize LLM
    llm = GeminiLLM(api_key=settings.gemini_api_key)
    
    # Retrieve context
    logger.info(f"Query: {query}")
    context, sources = retriever.retrieve_and_format(query)
    
    print("\n" + "="*60)
    print("RETRIEVED SOURCES:")
    print("="*60)
    for i, source in enumerate(sources, 1):
        print(f"\n[{i}] {source.get('source', 'Unknown')} (Score: {source.get('score', 0):.2f})")
        print(f"    {source.get('content', '')[:200]}...")
    
    # Generate response
    print("\n" + "="*60)
    print("GENERATED RESPONSE:")
    print("="*60)
    response = llm.generate(query, context)
    print(f"\n{response}")
    print("\n" + "="*60)


def interactive_mode():
    """Run in interactive mode"""
    print("\n🎓 ThriveBot Test Console")
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            query = input("\nYour question: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            if query:
                test_query(query)
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ThriveBot locally")
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        help="Question to ask (or run interactive mode if not provided)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    try:
        if args.query:
            test_query(args.query)
        else:
            interactive_mode()
    except Exception as e:
        logger.error("Test failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
