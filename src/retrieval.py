"""
Retrieval Module

This module handles the storage and retrieval of document embeddings.
It implements vector similarity search to find relevant documents for a given query.

Key Responsibilities:
1. Build and manage vector indices for efficient similarity search
2. Perform k-nearest neighbor search
3. Support multiple similarity metrics (cosine, dot product, euclidean)
4. Implement hybrid search strategies
5. Re-rank results for improved relevance

Design Principles:
- Efficient indexing for fast retrieval
- Support multiple vector database backends
- Flexible similarity metrics
- Scalable to large document collections
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json


class VectorIndex:
    """
    Base class for vector indices.
    Provides interface for different vector database backends.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize vector index.
        
        Args:
            dimension: Dimensionality of the vectors
        """
        self.dimension = dimension
        self.num_vectors = 0
    
    def add(self, vectors: np.ndarray, metadata: Optional[List[Dict]] = None):
        """
        Add vectors to the index.
        
        Args:
            vectors: numpy array of shape (n, dimension)
            metadata: Optional list of metadata dicts for each vector
        """
        raise NotImplementedError("Subclasses must implement add method")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_vector: Query vector of shape (dimension,)
            k: Number of neighbors to return
            
        Returns:
            Tuple of (distances, indices)
        """
        raise NotImplementedError("Subclasses must implement search method")
    
    def save(self, path: str):
        """Save index to disk."""
        raise NotImplementedError("Subclasses must implement save method")
    
    def load(self, path: str):
        """Load index from disk."""
        raise NotImplementedError("Subclasses must implement load method")


class FAISSIndex(VectorIndex):
    """
    Vector index using FAISS (Facebook AI Similarity Search).
    Provides efficient similarity search for large-scale datasets.
    """
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Vector dimension
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
                       - flat: Exact search, best for small datasets
                       - ivf: Inverted file index, faster for large datasets
                       - hnsw: Hierarchical NSW, good balance of speed/accuracy
        """
        super().__init__(dimension)
        self.index_type = index_type
        # TODO: Initialize FAISS index
        # import faiss
        # if index_type == "flat":
        #     self.index = faiss.IndexFlatL2(dimension)
        # elif index_type == "ivf":
        #     quantizer = faiss.IndexFlatL2(dimension)
        #     self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        # elif index_type == "hnsw":
        #     self.index = faiss.IndexHNSWFlat(dimension, 32)
        
        self.index = None
        self.vectors = []  # Placeholder storage
        print(f"[Placeholder] Initialized FAISS index: {index_type}")
    
    def add(self, vectors: np.ndarray, metadata: Optional[List[Dict]] = None):
        """
        Add vectors to FAISS index.
        
        Args:
            vectors: numpy array of vectors
            metadata: Optional metadata (stored separately)
        """
        # TODO: Implement FAISS add
        # self.index.add(vectors.astype('float32'))
        
        # Placeholder
        self.vectors.extend(vectors)
        self.num_vectors += len(vectors)
        print(f"Added {len(vectors)} vectors to index (total: {self.num_vectors})")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search FAISS index for nearest neighbors.
        
        Args:
            query_vector: Query vector
            k: Number of results
            
        Returns:
            Tuple of (distances, indices)
        """
        # TODO: Implement FAISS search
        # query = query_vector.reshape(1, -1).astype('float32')
        # distances, indices = self.index.search(query, k)
        # return distances[0], indices[0]
        
        # Placeholder: random results
        distances = np.random.rand(k)
        indices = np.random.randint(0, max(1, self.num_vectors), k)
        return distances, indices


class SimpleVectorIndex(VectorIndex):
    """
    Simple in-memory vector index using numpy.
    Good for small datasets and prototyping.
    """
    
    def __init__(self, dimension: int, metric: str = "cosine"):
        """
        Initialize simple vector index.
        
        Args:
            dimension: Vector dimension
            metric: Similarity metric ("cosine", "euclidean", "dot")
        """
        super().__init__(dimension)
        self.metric = metric
        self.vectors = None
        self.metadata_list = []
    
    def add(self, vectors: np.ndarray, metadata: Optional[List[Dict]] = None):
        """
        Add vectors to the index.
        
        Args:
            vectors: numpy array of vectors
            metadata: Optional metadata for each vector
        """
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        
        if metadata:
            self.metadata_list.extend(metadata)
        else:
            self.metadata_list.extend([{}] * len(vectors))
        
        self.num_vectors = len(self.vectors)
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors using brute force.
        
        Args:
            query_vector: Query vector
            k: Number of results
            
        Returns:
            Tuple of (distances/similarities, indices)
        """
        if self.vectors is None or len(self.vectors) == 0:
            return np.array([]), np.array([])
        
        if self.metric == "cosine":
            # Cosine similarity
            query_norm = query_vector / np.linalg.norm(query_vector)
            vectors_norm = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
            similarities = np.dot(vectors_norm, query_norm)
            # Higher is better for similarity, so negate for distance
            distances = 1 - similarities
        elif self.metric == "dot":
            # Dot product
            similarities = np.dot(self.vectors, query_vector)
            distances = -similarities  # Negate so smaller is better
        else:  # euclidean
            # Euclidean distance
            distances = np.linalg.norm(self.vectors - query_vector, axis=1)
        
        # Get top k
        k = min(k, len(distances))
        top_k_indices = np.argsort(distances)[:k]
        top_k_distances = distances[top_k_indices]
        
        return top_k_distances, top_k_indices
    
    def save(self, path: str):
        """Save index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save vectors
        np.save(path.with_suffix('.npy'), self.vectors)
        
        # Save metadata
        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'metric': self.metric,
                'dimension': self.dimension,
                'metadata': self.metadata_list
            }, f)
        
        print(f"Saved index to {path}")
    
    def load(self, path: str):
        """Load index from disk."""
        path = Path(path)
        
        # Load vectors
        self.vectors = np.load(path.with_suffix('.npy'))
        self.num_vectors = len(self.vectors)
        
        # Load metadata
        metadata_path = path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                self.metric = data.get('metric', self.metric)
                self.metadata_list = data.get('metadata', [])
        
        print(f"Loaded index from {path}")


class VectorRetriever:
    """
    Main class for vector-based retrieval.
    Manages the vector index and provides high-level retrieval interface.
    """
    
    def __init__(
        self,
        index_type: str = "simple",
        similarity_metric: str = "cosine",
        embedding_dim: Optional[int] = None
    ):
        """
        Initialize the retriever.
        
        Args:
            index_type: Type of vector index ("simple", "faiss")
            similarity_metric: Similarity metric to use
            embedding_dim: Dimension of embeddings (required if building new index)
        """
        self.index_type = index_type
        self.similarity_metric = similarity_metric
        self.embedding_dim = embedding_dim
        self.index = None
        self.documents = []  # Store original documents/chunks
    
    def build_index(
        self,
        embeddings: np.ndarray,
        documents: List,
        metadata: Optional[List[Dict]] = None
    ):
        """
        Build vector index from embeddings and documents.
        
        Args:
            embeddings: numpy array of embeddings
            documents: List of document chunks or strings
            metadata: Optional metadata for each document
        """
        # TODO: Implement index building
        # Steps:
        # 1. Validate inputs
        # 2. Initialize appropriate index type
        # 3. Add embeddings to index
        # 4. Store documents for retrieval
        
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        # Infer dimension if not set
        if self.embedding_dim is None:
            self.embedding_dim = embeddings.shape[1]
        
        # Initialize index
        if self.index_type == "simple":
            self.index = SimpleVectorIndex(
                dimension=self.embedding_dim,
                metric=self.similarity_metric
            )
        elif self.index_type == "faiss":
            self.index = FAISSIndex(
                dimension=self.embedding_dim,
                index_type="flat"
            )
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Add to index
        self.index.add(embeddings, metadata)
        
        # Store documents
        self.documents = documents
        
        print(f"Built index with {len(documents)} documents")
    
    def retrieve(
        self,
        query: str,
        embedder,
        top_k: int = 5,
        return_scores: bool = False
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            embedder: EmbeddingGenerator instance to embed the query
            top_k: Number of documents to retrieve
            return_scores: Whether to include similarity scores
            
        Returns:
            List of retrieved documents with metadata
        """
        # TODO: Implement retrieval pipeline
        # Steps:
        # 1. Generate query embedding
        # 2. Search vector index
        # 3. Retrieve corresponding documents
        # 4. Format and return results
        
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Generate query embedding
        query_embedding = embedder.generate_query_embedding(query)
        
        # Search index
        distances, indices = self.index.search(query_embedding, k=top_k)
        
        # Retrieve documents
        results = []
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if idx < len(self.documents):
                doc = self.documents[idx]
                result = {
                    'rank': i + 1,
                    'document': doc,
                    'index': int(idx)
                }
                
                if return_scores:
                    # Convert distance to similarity score
                    if self.similarity_metric == "cosine":
                        result['score'] = 1 - dist
                    else:
                        result['score'] = float(dist)
                
                # Add document text if available
                if hasattr(doc, 'text'):
                    result['text'] = doc.text
                elif isinstance(doc, str):
                    result['text'] = doc
                
                # Add metadata if available
                if hasattr(doc, 'metadata'):
                    result['metadata'] = doc.metadata
                
                results.append(result)
        
        return results
    
    def retrieve_with_reranking(
        self,
        query: str,
        embedder,
        top_k: int = 5,
        rerank_top_n: int = 20
    ) -> List[Dict]:
        """
        Retrieve documents with re-ranking for improved relevance.
        
        Args:
            query: Query string
            embedder: EmbeddingGenerator instance
            top_k: Final number of documents to return
            rerank_top_n: Number of candidates to retrieve before re-ranking
            
        Returns:
            List of re-ranked documents
        """
        # TODO: Implement re-ranking
        # Steps:
        # 1. Retrieve top_n candidates using vector search
        # 2. Apply re-ranking model (e.g., cross-encoder)
        # 3. Return top_k after re-ranking
        
        # For now, just use regular retrieval
        return self.retrieve(query, embedder, top_k=top_k, return_scores=True)
    
    def hybrid_search(
        self,
        query: str,
        embedder,
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> List[Dict]:
        """
        Perform hybrid search combining dense (semantic) and sparse (keyword) retrieval.
        
        Args:
            query: Query string
            embedder: EmbeddingGenerator instance
            top_k: Number of documents to return
            dense_weight: Weight for dense retrieval scores
            sparse_weight: Weight for sparse retrieval scores
            
        Returns:
            List of documents from hybrid search
        """
        # TODO: Implement hybrid search
        # Steps:
        # 1. Perform dense retrieval (vector search)
        # 2. Perform sparse retrieval (BM25, TF-IDF)
        # 3. Combine scores with weights
        # 4. Re-rank and return top_k
        
        # For now, just use dense retrieval
        print("[Placeholder] Hybrid search not implemented, using dense retrieval only")
        return self.retrieve(query, embedder, top_k=top_k, return_scores=True)
    
    def save_index(self, path: str):
        """
        Save the vector index and documents to disk.
        
        Args:
            path: Path to save the index
        """
        if self.index is None:
            raise ValueError("No index to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save index
        self.index.save(path)
        
        # Save documents separately
        import pickle
        docs_path = path.with_suffix('.docs.pkl')
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        print(f"Saved retriever to {path}")
    
    def load_index(self, path: str):
        """
        Load vector index and documents from disk.
        
        Args:
            path: Path to load the index from
        """
        path = Path(path)
        
        # Initialize index
        if self.index_type == "simple":
            self.index = SimpleVectorIndex(
                dimension=self.embedding_dim or 384,  # Default dimension
                metric=self.similarity_metric
            )
        elif self.index_type == "faiss":
            self.index = FAISSIndex(
                dimension=self.embedding_dim or 384,
                index_type="flat"
            )
        
        # Load index
        self.index.load(path)
        
        # Load documents
        import pickle
        docs_path = path.with_suffix('.docs.pkl')
        if docs_path.exists():
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
        
        print(f"Loaded retriever from {path}")


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the VectorRetriever module.
    This demonstrates how to build an index and retrieve documents.
    """
    
    # Create sample documents and embeddings
    sample_docs = [
        "Information retrieval is finding relevant documents.",
        "RAG combines retrieval with text generation.",
        "Vector embeddings represent text semantically.",
        "Question answering systems use NLP techniques.",
        "Machine learning powers modern search engines."
    ]
    
    # Placeholder embeddings (in practice, use EmbeddingGenerator)
    sample_embeddings = np.random.rand(len(sample_docs), 384)
    
    # Initialize retriever
    retriever = VectorRetriever(
        index_type="simple",
        similarity_metric="cosine",
        embedding_dim=384
    )
    
    # Build index
    retriever.build_index(sample_embeddings, sample_docs)
    
    # Create a mock embedder for demonstration
    class MockEmbedder:
        def generate_query_embedding(self, query):
            return np.random.rand(384)
    
    embedder = MockEmbedder()
    
    # Retrieve documents
    query = "How does RAG work?"
    results = retriever.retrieve(query, embedder, top_k=3, return_scores=True)
    
    print(f"\nTop {len(results)} results for query: '{query}'")
    for result in results:
        print(f"\nRank {result['rank']}:")
        print(f"  Text: {result['text']}")
        print(f"  Score: {result.get('score', 'N/A')}")
