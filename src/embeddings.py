"""
Embeddings Module

This module handles the generation of vector embeddings for text chunks.
Embeddings convert text into high-dimensional vectors that capture semantic meaning,
enabling similarity-based retrieval.

Key Responsibilities:
1. Initialize and manage embedding models
2. Convert text to vector representations
3. Batch processing for efficiency
4. Cache embeddings to avoid redundant computation
5. Support multiple embedding model backends

Design Principles:
- Model-agnostic interface for flexibility
- Efficient batch processing
- Memory-conscious operations
- Support for both local and API-based models
"""

import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path
import json


class EmbeddingModel:
    """
    Abstract base class for embedding models.
    This allows easy swapping between different embedding backends.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name or path of the model to use
        """
        self.model_name = model_name
        self.embedding_dim = None  # Will be set by specific implementations
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        raise NotImplementedError("Subclasses must implement embed method")
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        Some models use different prompts for queries vs documents.
        
        Args:
            query: Query text to embed
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        # Default implementation - can be overridden for query-specific handling
        return self.embed([query])[0]


class SentenceTransformerModel(EmbeddingModel):
    """
    Embedding model using sentence-transformers library.
    Supports various pre-trained models from HuggingFace.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize sentence-transformer model.
        
        Args:
            model_name: Name of the sentence-transformers model
                       Popular options:
                       - all-MiniLM-L6-v2 (fast, 384 dim)
                       - all-mpnet-base-v2 (high quality, 768 dim)
                       - multi-qa-mpnet-base-dot-v1 (for QA tasks)
        """
        super().__init__(model_name)
        # TODO: Initialize sentence-transformers model
        # from sentence_transformers import SentenceTransformer
        # self.model = SentenceTransformer(model_name)
        # self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Placeholder values
        self.model = None
        self.embedding_dim = 384  # Default for MiniLM
        print(f"[Placeholder] Initialized SentenceTransformer: {model_name}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using sentence-transformers.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of embeddings
        """
        # TODO: Implement actual embedding generation
        # return self.model.encode(texts, convert_to_numpy=True)
        
        # Placeholder: return random embeddings
        return np.random.rand(len(texts), self.embedding_dim)


class OpenAIEmbeddingModel(EmbeddingModel):
    """
    Embedding model using OpenAI's API.
    Requires API key and makes remote calls.
    """
    
    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: Optional[str] = None):
        """
        Initialize OpenAI embedding model.
        
        Args:
            model_name: OpenAI model name (e.g., text-embedding-ada-002)
            api_key: OpenAI API key (or set OPENAI_API_KEY environment variable)
        """
        super().__init__(model_name)
        # TODO: Initialize OpenAI client
        # import openai
        # self.client = openai.OpenAI(api_key=api_key)
        # self.embedding_dim = 1536  # for ada-002
        
        self.api_key = api_key
        self.embedding_dim = 1536
        print(f"[Placeholder] Initialized OpenAI embeddings: {model_name}")
    
    def embed(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Generate embeddings using OpenAI API.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process per API call
            
        Returns:
            numpy array of embeddings
        """
        # TODO: Implement OpenAI API calls with batching
        # embeddings = []
        # for i in range(0, len(texts), batch_size):
        #     batch = texts[i:i+batch_size]
        #     response = self.client.embeddings.create(
        #         model=self.model_name,
        #         input=batch
        #     )
        #     batch_embeddings = [item.embedding for item in response.data]
        #     embeddings.extend(batch_embeddings)
        # return np.array(embeddings)
        
        # Placeholder: return random embeddings
        return np.random.rand(len(texts), self.embedding_dim)


class EmbeddingGenerator:
    """
    Main class for generating and managing embeddings.
    Provides high-level interface for embedding generation with caching.
    """
    
    def __init__(
        self,
        model_type: str = "sentence-transformer",
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the EmbeddingGenerator.
        
        Args:
            model_type: Type of embedding model ("sentence-transformer" or "openai")
            model_name: Specific model name (uses defaults if None)
            cache_dir: Directory to cache embeddings (None to disable caching)
        """
        self.model_type = model_type
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache = {}
        
        # Initialize the appropriate model
        if model_type == "sentence-transformer":
            model_name = model_name or "all-MiniLM-L6-v2"
            self.model = SentenceTransformerModel(model_name)
        elif model_type == "openai":
            model_name = model_name or "text-embedding-ada-002"
            self.model = OpenAIEmbeddingModel(model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load cache if available
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()
    
    def generate_embeddings(
        self,
        texts: Union[List[str], List],
        show_progress: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts or document chunks.
        
        Args:
            texts: List of text strings or DocumentChunk objects
            show_progress: Whether to show progress bar
            batch_size: Number of texts to process at once
            
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        # TODO: Implement batched embedding generation with progress tracking
        # Extract text content if DocumentChunk objects are provided
        text_strings = []
        for item in texts:
            if isinstance(item, str):
                text_strings.append(item)
            else:
                # Assume it's a DocumentChunk or similar object with .text attribute
                text_strings.append(item.text)
        
        # Check cache for existing embeddings
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        for i, text in enumerate(text_strings):
            cached_embedding = self._get_cached_embedding(text)
            if cached_embedding is not None:
                embeddings.append((i, cached_embedding))
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Generate embeddings for uncached texts
        if texts_to_embed:
            if show_progress:
                print(f"Generating embeddings for {len(texts_to_embed)} texts...")
            
            # Process in batches
            new_embeddings = []
            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i:i+batch_size]
                batch_embeddings = self.model.embed(batch)
                new_embeddings.append(batch_embeddings)
                
                if show_progress:
                    progress = min(i + batch_size, len(texts_to_embed))
                    print(f"Progress: {progress}/{len(texts_to_embed)}")
            
            # Combine batch results
            if new_embeddings:
                new_embeddings = np.vstack(new_embeddings)
                
                # Cache new embeddings
                for text, embedding in zip(texts_to_embed, new_embeddings):
                    self._cache_embedding(text, embedding)
                    embeddings.append((indices_to_embed[len(embeddings) - len(embeddings)], embedding))
        
        # Sort by original indices and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        result = np.array([emb for _, emb in embeddings])
        
        # If no cached embeddings were used, just generate all at once
        if len(embeddings) == 0:
            result = self.model.embed(text_strings)
            # Cache all
            for text, embedding in zip(text_strings, result):
                self._cache_embedding(text, embedding)
        
        return result
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        Args:
            query: Query text
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        # Check cache first
        cached = self._get_cached_embedding(query)
        if cached is not None:
            return cached
        
        # Generate new embedding
        embedding = self.model.embed_query(query)
        
        # Cache it
        self._cache_embedding(query, embedding)
        
        return embedding
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Retrieve cached embedding for a text.
        
        Args:
            text: Text to look up
            
        Returns:
            Cached embedding or None if not found
        """
        # TODO: Implement efficient caching mechanism
        # Options:
        # 1. In-memory dict with text hash as key
        # 2. Disk-based cache using pickle or joblib
        # 3. Database (SQLite, Redis) for larger caches
        
        text_hash = self._hash_text(text)
        return self.cache.get(text_hash)
    
    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """
        Store embedding in cache.
        
        Args:
            text: Text that was embedded
            embedding: Generated embedding vector
        """
        text_hash = self._hash_text(text)
        self.cache[text_hash] = embedding
        
        # Optionally save to disk
        if self.cache_dir:
            cache_file = self.cache_dir / f"{text_hash}.npy"
            np.save(cache_file, embedding)
    
    def _hash_text(self, text: str) -> str:
        """
        Generate hash for text to use as cache key.
        
        Args:
            text: Text to hash
            
        Returns:
            Hash string
        """
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _load_cache(self):
        """Load cached embeddings from disk."""
        if not self.cache_dir or not self.cache_dir.exists():
            return
        
        # TODO: Implement cache loading from disk
        # for cache_file in self.cache_dir.glob("*.npy"):
        #     text_hash = cache_file.stem
        #     embedding = np.load(cache_file)
        #     self.cache[text_hash] = embedding
        
        print(f"[Placeholder] Cache directory: {self.cache_dir}")
    
    def save_embeddings(self, embeddings: np.ndarray, output_path: str, metadata: Optional[Dict] = None):
        """
        Save embeddings to disk for later use.
        
        Args:
            embeddings: numpy array of embeddings
            output_path: Path to save embeddings
            metadata: Optional metadata to save with embeddings
        """
        # TODO: Implement saving embeddings with metadata
        # Options:
        # 1. numpy .npz format for embeddings + JSON for metadata
        # 2. HDF5 for large-scale storage
        # 3. Pickle for quick prototyping
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        np.save(output_path, embeddings)
        
        # Save metadata if provided
        if metadata:
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Saved embeddings to {output_path}")
    
    def load_embeddings(self, input_path: str) -> tuple:
        """
        Load embeddings from disk.
        
        Args:
            input_path: Path to load embeddings from
            
        Returns:
            Tuple of (embeddings, metadata)
        """
        # TODO: Implement loading embeddings with metadata
        input_path = Path(input_path)
        
        # Load embeddings
        embeddings = np.load(input_path)
        
        # Load metadata if available
        metadata = None
        metadata_path = input_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        print(f"Loaded embeddings from {input_path}")
        return embeddings, metadata
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.model.embedding_dim


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the EmbeddingGenerator module.
    This demonstrates how to generate embeddings for text.
    """
    
    # Initialize the embedding generator
    embedder = EmbeddingGenerator(
        model_type="sentence-transformer",
        model_name="all-MiniLM-L6-v2",
        cache_dir="cache/embeddings"
    )
    
    # Example texts
    texts = [
        "What is information retrieval?",
        "Information retrieval is the process of obtaining relevant information.",
        "RAG systems combine retrieval with generation.",
        "Vector embeddings capture semantic meaning of text."
    ]
    
    # Generate embeddings
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = embedder.generate_embeddings(texts)
    
    print(f"\nGenerated embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embedder.embedding_dimension}")
    
    # Generate query embedding
    query = "How does RAG work?"
    query_embedding = embedder.generate_query_embedding(query)
    print(f"\nQuery embedding shape: {query_embedding.shape}")
    
    # Calculate similarity (example)
    # Cosine similarity between query and first document
    similarity = np.dot(query_embedding, embeddings[0]) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(embeddings[0])
    )
    print(f"\nSimilarity between query and first text: {similarity:.4f}")
