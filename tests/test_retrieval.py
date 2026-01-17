"""
Tests for the retrieval module
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestFAISSVectorStore:
    """Tests for FAISS vector store"""
    
    def test_initialization(self):
        """Test vector store initializes correctly"""
        from app.retrieval import FAISSVectorStore
        
        store = FAISSVectorStore(dimension=768, store_path="data/test_store")
        assert store.dimension == 768
        assert store.size == 0
    
    def test_add_and_search(self):
        """Test adding and searching documents"""
        from app.retrieval import FAISSVectorStore
        
        store = FAISSVectorStore(dimension=4, store_path="data/test_store")
        
        # Create test embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        
        documents = [
            {"content": "Document about cats", "source": "doc1.txt"},
            {"content": "Document about dogs", "source": "doc2.txt"},
            {"content": "Document about birds", "source": "doc3.txt"},
        ]
        
        store.add(embeddings, documents)
        
        assert store.size == 3
        
        # Search for similar document
        query = np.array([1.0, 0.1, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=2)
        
        assert len(results) == 2
        # First result should be the cats document (most similar)
        assert "cats" in results[0][0]["content"]
    
    def test_empty_search(self):
        """Test searching empty store returns empty results"""
        from app.retrieval import FAISSVectorStore
        
        store = FAISSVectorStore(dimension=4, store_path="data/test_store")
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        results = store.search(query, top_k=5)
        assert len(results) == 0


class TestTextChunker:
    """Tests for text chunking"""
    
    def test_basic_chunking(self):
        """Test basic text chunking"""
        from app.ingestion import TextChunker
        
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        
        text = """This is the first paragraph with some content.

        This is the second paragraph with different content.

        This is the third paragraph with more information."""
        
        chunks = chunker.chunk_text(text, "test.txt")
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.content
            assert chunk.source == "test.txt"
    
    def test_empty_text(self):
        """Test chunking empty text"""
        from app.ingestion import TextChunker
        
        chunker = TextChunker()
        chunks = chunker.chunk_text("", "test.txt")
        
        assert len(chunks) == 0
    
    def test_chunk_ids_are_unique(self):
        """Test that chunk IDs are unique"""
        from app.ingestion import TextChunker
        
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        
        text = "A " * 100  # Create text that will be split into multiple chunks
        chunks = chunker.chunk_text(text, "test.txt")
        
        ids = [chunk.id for chunk in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs should be unique"


class TestDocumentLoader:
    """Tests for document loading"""
    
    def test_text_file_loading(self, tmp_path):
        """Test loading a text file"""
        from app.ingestion import DocumentLoader
        
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is test content.")
        
        loader = DocumentLoader(str(tmp_path))
        documents = loader.load_all()
        
        assert len(documents) == 1
        assert documents[0].content == "This is test content."
    
    def test_unsupported_format(self, tmp_path):
        """Test that unsupported formats are skipped"""
        from app.ingestion import DocumentLoader
        
        # Create a file with unsupported extension
        test_file = tmp_path / "test.xyz"
        test_file.write_text("This should be skipped.")
        
        loader = DocumentLoader(str(tmp_path))
        documents = loader.load_all()
        
        assert len(documents) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
