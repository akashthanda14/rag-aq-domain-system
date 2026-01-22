"""
Unit Tests for Document Ingestion Module

Tests the functionality of document loading, preprocessing, and chunking.

Author: CSE435 Project Team
"""

import pytest
from pathlib import Path
from src.ingestion import DocumentIngestion, validate_document_format


class TestDocumentIngestion:
    """Test cases for DocumentIngestion class."""
    
    def test_initialization(self):
        """Test that DocumentIngestion initializes correctly."""
        ingestion = DocumentIngestion(data_dir="data/raw")
        assert ingestion.data_dir == Path("data/raw")
        assert ingestion.chunk_size == 1000
        assert ingestion.chunk_overlap == 200
        assert '.txt' in ingestion.supported_formats
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        ingestion = DocumentIngestion(
            data_dir="custom/path",
            chunk_size=500,
            chunk_overlap=100
        )
        assert ingestion.chunk_size == 500
        assert ingestion.chunk_overlap == 100
    
    # TODO: Add more tests once implementation is complete
    # def test_load_documents(self):
    #     """Test document loading functionality."""
    #     pass
    # 
    # def test_chunk_documents(self):
    #     """Test document chunking."""
    #     pass
    # 
    # def test_preprocess_text(self):
    #     """Test text preprocessing."""
    #     pass


class TestValidateDocumentFormat:
    """Test cases for document format validation."""
    
    def test_validate_format(self):
        """Test format validation function."""
        # TODO: Implement once function is complete
        assert validate_document_format("test.txt") == True


if __name__ == "__main__":
    pytest.main([__file__])
