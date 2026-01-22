"""
RAG System for Domain-Specific Question Answering
CSE435 Project

This package implements a Retrieval-Augmented Generation system with modular components
for document ingestion, embedding generation, retrieval, and response generation.
"""

__version__ = "1.0.0"
__author__ = "CSE435 Project Team"

from .ingestion import DocumentIngestion
from .embeddings import EmbeddingGenerator
from .retrieval import DocumentRetriever
from .generation import ResponseGenerator

__all__ = [
    "DocumentIngestion",
    "EmbeddingGenerator",
    "DocumentRetriever",
    "ResponseGenerator",
]
