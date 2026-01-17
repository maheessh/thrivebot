"""
RAG Retriever - Orchestrates the retrieval pipeline
Combines embedding, search, and context formatting
"""

from typing import List, Dict, Any, Optional, Tuple
import structlog

from app.ingestion.embedder import GeminiEmbedder
from app.retrieval.vector_store import FAISSVectorStore

logger = structlog.get_logger()


class RAGRetriever:
    """
    Retrieval-Augmented Generation retriever.
    Handles query embedding and context retrieval.
    """
    
    def __init__(
        self,
        embedder: GeminiEmbedder,
        vector_store: FAISSVectorStore,
        top_k: int = 5,
        score_threshold: float = 0.3
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User's question or search query
            top_k: Override default number of results
            
        Returns:
            List of relevant document chunks with scores
        """
        if not query or not query.strip():
            return []
        
        k = top_k or self.top_k
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_query(query)
            
            # Search vector store
            results = self.vector_store.search(query_embedding, k)
            
            # Filter by score threshold and format results
            formatted_results = []
            for doc, score in results:
                if score >= self.score_threshold:
                    formatted_results.append({
                        "content": doc.get("content", ""),
                        "source": doc.get("source", "unknown"),
                        "score": score,
                        "metadata": doc.get("metadata", {})
                    })
            
            logger.info(
                "Retrieved documents",
                query=query[:50],
                num_results=len(formatted_results)
            )
            
            return formatted_results
            
        except Exception as e:
            logger.error("Retrieval failed", error=str(e), query=query[:50])
            return []
    
    def format_context(
        self,
        results: List[Dict[str, Any]],
        max_tokens: int = 3000
    ) -> str:
        """
        Format retrieved results into a context string for the LLM.
        
        Args:
            results: List of retrieved documents
            max_tokens: Maximum approximate tokens for context
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results, 1):
            content = result["content"]
            source = result.get("source", "Unknown")
            score = result.get("score", 0)
            
            # Format each context piece
            context_piece = f"""
[Source {i}: {source}]
Relevance: {score:.2f}
Content: {content}
"""
            
            # Approximate token count (4 chars per token)
            piece_tokens = len(context_piece) // 4
            
            if current_length + piece_tokens > max_tokens:
                break
                
            context_parts.append(context_piece)
            current_length += piece_tokens
        
        return "\n---\n".join(context_parts)
    
    def retrieve_and_format(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_tokens: int = 3000
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Convenience method to retrieve and format in one call.
        
        Returns:
            Tuple of (formatted_context, raw_results)
        """
        results = self.retrieve(query, top_k)
        context = self.format_context(results, max_tokens)
        return context, results
