"""
Text Chunker - Splits documents into optimal chunks for embedding
Uses tiktoken for accurate token counting
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field
import re
import structlog

logger = structlog.get_logger()


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    content: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def id(self) -> str:
        """Generate unique ID for this chunk"""
        return f"{self.source}::chunk_{self.chunk_index}"


class TextChunker:
    """
    Splits documents into chunks optimized for embedding and retrieval.
    Uses semantic-aware splitting with overlap for context preservation.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name
        self._tokenizer = None
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer"""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding(self.encoding_name)
            except ImportError:
                logger.warning("tiktoken not available, using approximate counting")
                self._tokenizer = None
        return self._tokenizer
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Fallback: approximate 4 chars per token
        return len(text) // 4
    
    def chunk_text(self, text: str, source: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """
        Split text into chunks with overlap.
        Uses semantic boundaries (paragraphs, sentences) when possible.
        """
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # First, split by paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            # If single paragraph exceeds chunk size, split by sentences
            if para_tokens > self.chunk_size:
                # Flush current chunk if exists
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, source, len(chunks), metadata
                    ))
                    current_chunk = []
                    current_tokens = 0
                
                # Split large paragraph by sentences
                sentence_chunks = self._split_large_paragraph(para, source, len(chunks), metadata)
                for chunk in sentence_chunks:
                    chunk.chunk_index = len(chunks)
                    chunks.append(chunk)
            
            # Check if adding this paragraph exceeds chunk size
            elif current_tokens + para_tokens > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, source, len(chunks), metadata
                    ))
                
                # Start new chunk with overlap from previous
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_tokens = self.count_tokens(" ".join(current_chunk))
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk, source, len(chunks), metadata
            ))
        
        logger.info(
            "Chunking complete",
            source=source,
            num_chunks=len(chunks),
            avg_tokens=sum(self.count_tokens(c.content) for c in chunks) // max(len(chunks), 1)
        )
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split on double newlines or multiple newlines
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - handles common cases
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_large_paragraph(
        self,
        paragraph: str,
        source: str,
        start_index: int,
        metadata: Dict[str, Any]
    ) -> List[TextChunk]:
        """Split a large paragraph into smaller chunks by sentences"""
        sentences = self._split_into_sentences(paragraph)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, source, start_index + len(chunks), metadata
                    ))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk, source, start_index + len(chunks), metadata
            ))
        
        return chunks
    
    def _create_chunk(
        self,
        text_parts: List[str],
        source: str,
        chunk_index: int,
        metadata: Dict[str, Any]
    ) -> TextChunk:
        """Create a TextChunk from text parts"""
        content = " ".join(text_parts)
        return TextChunk(
            content=content,
            source=source,
            chunk_index=chunk_index,
            metadata={
                **metadata,
                "token_count": self.count_tokens(content)
            }
        )
    
    def _get_overlap_text(self, chunk_parts: List[str]) -> str:
        """Get overlap text from the end of a chunk"""
        if not chunk_parts:
            return ""
        
        full_text = " ".join(chunk_parts)
        
        if self.tokenizer:
            tokens = self.tokenizer.encode(full_text)
            if len(tokens) <= self.chunk_overlap:
                return full_text
            overlap_tokens = tokens[-self.chunk_overlap:]
            return self.tokenizer.decode(overlap_tokens)
        else:
            # Fallback: use character approximation
            chars = self.chunk_overlap * 4
            return full_text[-chars:] if len(full_text) > chars else full_text
