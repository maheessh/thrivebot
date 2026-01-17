"""
Gemini Embedder - Generate embeddings using Google's Gemini API
Uses text-embedding-004 model for high-quality embeddings
"""

from typing import List, Optional
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

logger = structlog.get_logger()


class GeminiEmbedder:
    """
    Generate embeddings using Google's Gemini embedding model.
    Handles batching and rate limiting automatically.
    """
    
    # Gemini embedding model
    MODEL_NAME = "models/text-embedding-004"
    
    # Embedding dimension for text-embedding-004
    EMBEDDING_DIMENSION = 768
    
    # Max batch size for embedding requests
    MAX_BATCH_SIZE = 100
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None
        self._configure_client()
    
    def _configure_client(self):
        """Configure the Gemini client"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai
            logger.info("Gemini client configured successfully")
        except Exception as e:
            logger.error("Failed to configure Gemini client", error=str(e))
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        Returns numpy array of shape (768,)
        """
        try:
            result = self._client.embed_content(
                model=self.MODEL_NAME,
                content=text,
                task_type="retrieval_document"
            )
            return np.array(result['embedding'], dtype=np.float32)
        except Exception as e:
            logger.error("Embedding generation failed", error=str(e))
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        Uses different task_type for better retrieval performance.
        """
        try:
            result = self._client.embed_content(
                model=self.MODEL_NAME,
                content=query,
                task_type="retrieval_query"
            )
            return np.array(result['embedding'], dtype=np.float32)
        except Exception as e:
            logger.error("Query embedding failed", error=str(e))
            raise
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        Handles batching automatically.
        Returns numpy array of shape (n_texts, 768)
        """
        if not texts:
            return np.array([])
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[i:i + self.MAX_BATCH_SIZE]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
            
            logger.info(
                "Embedded batch",
                batch_num=i // self.MAX_BATCH_SIZE + 1,
                total_batches=(len(texts) - 1) // self.MAX_BATCH_SIZE + 1
            )
        
        return np.array(all_embeddings, dtype=np.float32)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts"""
        try:
            # Gemini supports batch embedding
            result = self._client.embed_content(
                model=self.MODEL_NAME,
                content=texts,
                task_type="retrieval_document"
            )
            return [np.array(emb, dtype=np.float32) for emb in result['embedding']]
        except Exception as e:
            logger.error("Batch embedding failed", error=str(e), batch_size=len(texts))
            # Fallback to individual embeddings
            return [self.embed_text(text) for text in texts]
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension"""
        return self.EMBEDDING_DIMENSION
