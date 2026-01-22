"""
Example Workflow for RAG System

This script demonstrates a complete end-to-end workflow for the RAG system:
1. Document ingestion and chunking
2. Embedding generation
3. Vector index building
4. Query processing and retrieval
5. Response generation

This is a simplified example showing the system architecture.
For production use, implement the TODO sections in each module.

Author: Akash Thanda
Course: CSE435 - Information Retrieval
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Check for required dependencies
try:
    import numpy as np
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("=" * 80)
    print("  NOTICE: Required dependencies not installed")
    print("=" * 80)
    print("\nTo run this example, please install dependencies:")
    print("  pip install -r requirements.txt")
    print("\nThis demo will show the workflow structure without actual execution.")
    print("=" * 80 + "\n")

if DEPENDENCIES_AVAILABLE:
    from src.ingestion import DocumentIngestion, Document
    from src.embeddings import EmbeddingGenerator
    from src.retrieval import VectorRetriever
    from src.response_generation import ResponseGenerator


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_workflow_overview():
    """Print workflow overview when dependencies are not available."""
    print_section("RAG System Workflow Overview")
    
    print("The RAG system follows this workflow:\n")
    
    print("1. DOCUMENT INGESTION")
    print("   - Load documents from various formats (PDF, TXT, DOCX, HTML)")
    print("   - Clean and preprocess text")
    print("   - Split into chunks with overlap for context preservation")
    print("   - Extract metadata (source, page numbers, etc.)\n")
    
    print("2. EMBEDDING GENERATION")
    print("   - Convert text chunks to vector representations")
    print("   - Use models like Sentence-BERT or OpenAI embeddings")
    print("   - Cache embeddings for efficiency")
    print("   - Typical dimensions: 384-1536\n")
    
    print("3. VECTOR INDEX BUILDING")
    print("   - Store embeddings in vector database")
    print("   - Options: FAISS, Pinecone, Chroma")
    print("   - Enable fast similarity search")
    print("   - Support millions of documents\n")
    
    print("4. QUERY PROCESSING")
    print("   - Convert user question to vector")
    print("   - Perform k-nearest neighbor search")
    print("   - Retrieve top-k most relevant chunks")
    print("   - Calculate similarity scores\n")
    
    print("5. RESPONSE GENERATION")
    print("   - Construct prompt with retrieved context")
    print("   - Send to LLM (GPT-4, GPT-3.5, etc.)")
    print("   - Generate contextual answer")
    print("   - Include source citations\n")
    
    print_section("Getting Started")
    
    print("To run the full demo:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. (Optional) Set up API keys in .env file")
    print("  3. Run: python example_workflow.py\n")
    
    print("To explore the codebase:")
    print("  - src/ingestion.py - Document loading and chunking")
    print("  - src/embeddings.py - Vector embedding generation")
    print("  - src/retrieval.py - Vector search and retrieval")
    print("  - src/response_generation.py - LLM-based answer generation\n")
    
    print("Each module contains detailed comments and TODO sections for implementation.\n")
    print("See README.md for complete documentation.")



def main():
    """
    Main workflow demonstrating the RAG system.
    """
    
    if not DEPENDENCIES_AVAILABLE:
        print_workflow_overview()
        return
    
    print_section("RAG System Demo - Complete Workflow")
    
    # =========================================================================
    # STEP 1: Document Ingestion
    # =========================================================================
    print_section("Step 1: Document Ingestion & Chunking")
    
    # Initialize ingestion system
    ingestion = DocumentIngestion(
        chunk_size=500,  # Characters per chunk
        chunk_overlap=50  # Overlap between chunks
    )
    
    # Create sample documents for demonstration
    # In production, you would load from files: ingestion.load_documents("data/")
    sample_documents = [
        Document(
            content="""
            Retrieval-Augmented Generation (RAG) is an advanced AI architecture that 
            combines the strengths of large language models with external knowledge retrieval. 
            The system works by first retrieving relevant information from a knowledge base, 
            then using that information to generate accurate, contextual responses.
            """,
            metadata={'source': 'rag_overview.txt', 'topic': 'architecture'}
        ),
        Document(
            content="""
            The RAG pipeline consists of several key components: document ingestion handles 
            loading and preprocessing of source documents, the embedding module converts text 
            into vector representations, the retrieval system finds relevant context using 
            similarity search, and the generation module produces answers using an LLM.
            """,
            metadata={'source': 'rag_components.txt', 'topic': 'components'}
        ),
        Document(
            content="""
            Information retrieval is the foundation of RAG systems. It uses techniques like 
            TF-IDF, BM25, and neural embeddings to find relevant documents. Vector databases 
            like FAISS enable efficient similarity search at scale. The quality of retrieval 
            directly impacts the accuracy of generated responses.
            """,
            metadata={'source': 'information_retrieval.txt', 'topic': 'retrieval'}
        ),
        Document(
            content="""
            Embeddings are dense vector representations of text that capture semantic meaning. 
            Models like Sentence-BERT and OpenAI's text-embedding-ada-002 are commonly used. 
            Good embeddings ensure that semantically similar texts have similar vector 
            representations, enabling effective retrieval.
            """,
            metadata={'source': 'embeddings_guide.txt', 'topic': 'embeddings'}
        ),
        Document(
            content="""
            Large Language Models (LLMs) like GPT-4 power the generation component of RAG. 
            They take retrieved context and user questions as input, then generate coherent, 
            contextual responses. Prompt engineering is crucial for controlling the quality 
            and format of generated answers.
            """,
            metadata={'source': 'llm_basics.txt', 'topic': 'generation'}
        )
    ]
    
    print(f"Created {len(sample_documents)} sample documents")
    
    # Chunk documents
    chunks = ingestion.chunk_documents(sample_documents)
    print(f"Generated {len(chunks)} chunks from documents")
    
    # Display sample chunk
    if chunks:
        print(f"\nSample chunk:")
        print(f"  ID: {chunks[0].chunk_id}")
        print(f"  Text: {chunks[0].text[:150]}...")
        print(f"  Metadata: {chunks[0].metadata}")
    
    # =========================================================================
    # STEP 2: Generate Embeddings
    # =========================================================================
    print_section("Step 2: Embedding Generation")
    
    # Initialize embedding generator
    # Note: This uses placeholder embeddings. In production, install sentence-transformers
    embedder = EmbeddingGenerator(
        model_type="sentence-transformer",
        model_name="all-MiniLM-L6-v2",
        cache_dir="cache/embeddings"
    )
    
    print(f"Embedding model: sentence-transformer/all-MiniLM-L6-v2")
    print(f"Embedding dimension: {embedder.embedding_dimension}")
    
    # Generate embeddings for all chunks
    print(f"\nGenerating embeddings for {len(chunks)} chunks...")
    embeddings = embedder.generate_embeddings(chunks, show_progress=True)
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # =========================================================================
    # STEP 3: Build Vector Index
    # =========================================================================
    print_section("Step 3: Building Vector Index")
    
    # Initialize retriever
    retriever = VectorRetriever(
        index_type="simple",  # Use "faiss" for larger datasets
        similarity_metric="cosine",
        embedding_dim=embedder.embedding_dimension
    )
    
    # Build index
    retriever.build_index(embeddings, chunks)
    print(f"Built vector index with {len(chunks)} documents")
    
    # Optionally save index for later use
    # retriever.save_index("cache/vector_index")
    
    # =========================================================================
    # STEP 4: Query Processing & Retrieval
    # =========================================================================
    print_section("Step 4: Query Processing & Retrieval")
    
    # Example queries
    queries = [
        "What is RAG and how does it work?",
        "Explain the role of embeddings in RAG systems",
        "What are the main components of a RAG pipeline?"
    ]
    
    # Process first query in detail
    query = queries[0]
    print(f"Query: {query}\n")
    
    # Retrieve relevant documents
    top_k = 3
    retrieved_docs = retriever.retrieve(
        query=query,
        embedder=embedder,
        top_k=top_k,
        return_scores=True
    )
    
    print(f"Retrieved top {top_k} documents:\n")
    for doc in retrieved_docs:
        print(f"Rank {doc['rank']}:")
        print(f"  Text: {doc['text'][:200]}...")
        print(f"  Score: {doc.get('score', 'N/A'):.4f}")
        print(f"  Source: {doc.get('metadata', {}).get('source', 'Unknown')}\n")
    
    # =========================================================================
    # STEP 5: Response Generation
    # =========================================================================
    print_section("Step 5: Response Generation")
    
    # Initialize response generator
    # Note: This uses placeholder responses. In production, set up OpenAI API key
    generator = ResponseGenerator(
        llm_type="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    
    print(f"LLM: {generator.llm_type} ({generator.llm.model_name})")
    print(f"Temperature: {generator.temperature}\n")
    
    # Generate response
    response = generator.generate(
        question=query,
        retrieved_docs=retrieved_docs,
        max_tokens=500,
        include_sources=True
    )
    
    # Display formatted response
    print(generator.format_response_for_display(response))
    
    # =========================================================================
    # STEP 6: Multiple Query Examples
    # =========================================================================
    print_section("Step 6: Additional Query Examples")
    
    for query in queries[1:]:
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        # Retrieve
        retrieved = retriever.retrieve(
            query=query,
            embedder=embedder,
            top_k=2,
            return_scores=True
        )
        
        # Generate
        response = generator.generate(
            question=query,
            retrieved_docs=retrieved,
            max_tokens=300,
            include_sources=False
        )
        
        print(f"Answer: {response['answer']}\n")
        print(f"Confidence: {response['confidence']}")
        print(f"Sources used: {response['num_sources']}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_section("Workflow Complete!")
    
    print("Summary of RAG System Workflow:")
    print(f"  1. Ingested {len(sample_documents)} documents")
    print(f"  2. Created {len(chunks)} text chunks")
    print(f"  3. Generated {embeddings.shape[0]} embeddings ({embeddings.shape[1]}-dim)")
    print(f"  4. Built vector index for retrieval")
    print(f"  5. Processed {len(queries)} queries")
    print("\nNext Steps:")
    print("  - Implement actual embedding models (sentence-transformers)")
    print("  - Set up LLM API (OpenAI, HuggingFace)")
    print("  - Add your own documents to data/")
    print("  - Customize prompts for your domain")
    print("  - Scale with FAISS for larger datasets")
    print("  - Add evaluation metrics")
    print("\nFor more details, see README.md")


if __name__ == "__main__":
    """
    Run the example workflow.
    
    This demonstrates the complete RAG pipeline with placeholder implementations.
    To use with real models:
    1. Install dependencies: pip install -r requirements.txt
    2. Set up API keys in .env file (if using OpenAI)
    3. Implement the TODO sections in each module
    """
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
    except Exception as e:
        print(f"\n\nError during workflow: {e}")
        import traceback
        traceback.print_exc()
