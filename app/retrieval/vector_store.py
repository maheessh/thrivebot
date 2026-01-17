"""
FAISS Vector Store - Efficient similarity search using Facebook's FAISS
Supports persistence and incremental updates
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import structlog

logger = structlog.get_logger()


class FAISSVectorStore:
    """
    Vector store using FAISS for efficient similarity search.
    Supports persistence and metadata storage.
    """
    
    def __init__(
        self,
        dimension: int = 768,
        store_path: str = "data/vector_store"
    ):
        self.dimension = dimension
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.documents: List[Dict[str, Any]] = []
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        try:
            import faiss
            # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
            # For larger datasets, consider IndexIVFFlat or IndexHNSWFlat
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("FAISS index initialized", dimension=self.dimension)
        except ImportError:
            logger.error("FAISS not installed. Run: pip install faiss-cpu")
            raise
    
    def add(
        self,
        embeddings: np.ndarray,
        documents: List[Dict[str, Any]]
    ) -> None:
        """
        Add embeddings and their associated documents to the index.
        
        Args:
            embeddings: numpy array of shape (n, dimension)
            documents: list of document dicts with 'content', 'source', 'metadata'
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        if len(embeddings) == 0:
            return
        
        # Normalize embeddings for cosine similarity
        embeddings = self._normalize(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store document metadata
        self.documents.extend(documents)
        
        logger.info(
            "Added documents to vector store",
            num_added=len(documents),
            total_documents=len(self.documents)
        )
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: numpy array of shape (dimension,)
            top_k: number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Normalize query embedding
        query_embedding = self._normalize(query_embedding.reshape(1, -1))
        
        # Limit top_k to available documents
        top_k = min(top_k, self.index.ntotal)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        return embeddings / norms
    
    def save(self, name: str = "index") -> None:
        """Save the index and documents to disk"""
        import faiss
        
        index_path = self.store_path / f"{name}.faiss"
        docs_path = self.store_path / f"{name}.pkl"
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save documents
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        logger.info(
            "Vector store saved",
            index_path=str(index_path),
            num_documents=len(self.documents)
        )
    
    def load(self, name: str = "index") -> bool:
        """Load the index and documents from disk"""
        import faiss
        
        index_path = self.store_path / f"{name}.faiss"
        docs_path = self.store_path / f"{name}.pkl"
        
        if not index_path.exists() or not docs_path.exists():
            logger.warning("No saved index found", path=str(self.store_path))
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load documents
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            logger.info(
                "Vector store loaded",
                num_documents=len(self.documents)
            )
            return True
        except Exception as e:
            logger.error("Failed to load vector store", error=str(e))
            return False
    
    def clear(self) -> None:
        """Clear all documents from the index"""
        self._initialize_index()
        self.documents = []
        logger.info("Vector store cleared")
    
    @property
    def size(self) -> int:
        """Return number of documents in the store"""
        return len(self.documents)
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the vector store"""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "store_path": str(self.store_path)
        }
