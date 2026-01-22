"""
RAG System for Domain-Specific Question Answering

This package provides core components for building a Retrieval-Augmented Generation system:
- Document ingestion and preprocessing
- Embedding generation
- Vector-based retrieval
- Response generation using LLMs

Author: Akash Thanda
Course: CSE435 - Information Retrieval
"""

__version__ = "1.0.0"
__author__ = "Akash Thanda"

# Import main components for easier access
from .ingestion import DocumentIngestion
from .embeddings import EmbeddingGenerator
from .retrieval import VectorRetriever
from .response_generation import ResponseGenerator

__all__ = [
    'DocumentIngestion',
    'EmbeddingGenerator',
    'VectorRetriever',
    'ResponseGenerator',
]
