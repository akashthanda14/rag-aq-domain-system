"""
Document Ingestion Module

This module handles the ingestion and preprocessing of documents for the RAG system.
It supports multiple file formats and prepares text for embedding generation.

Key Responsibilities:
1. Load documents from various sources (PDF, TXT, DOCX, HTML, etc.)
2. Extract and clean text content
3. Split documents into manageable chunks with overlap
4. Extract and preserve metadata
5. Handle special formatting and structure

Design Principles:
- Support multiple document formats
- Preserve document structure and metadata
- Efficient chunking strategies for context preservation
- Error handling for corrupted or unsupported files
"""

import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class Document:
    """
    Represents a document with its content and metadata.
    
    Attributes:
        content (str): The text content of the document
        metadata (dict): Metadata including source, page numbers, timestamps, etc.
        doc_id (str): Unique identifier for the document
    """
    
    def __init__(self, content: str, metadata: Optional[Dict] = None, doc_id: Optional[str] = None):
        """
        Initialize a Document object.
        
        Args:
            content: The text content of the document
            metadata: Optional dictionary containing document metadata
            doc_id: Optional unique identifier
        """
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id or self._generate_id()
    
    def _generate_id(self) -> str:
        """
        Generate a unique ID for the document.
        
        Returns:
            A unique string identifier
        """
        # TODO: Implement unique ID generation (e.g., using UUID or hash)
        import hashlib
        return hashlib.md5(self.content.encode()).hexdigest()[:16]
    
    def __repr__(self):
        return f"Document(id={self.doc_id}, length={len(self.content)}, metadata={self.metadata})"


class DocumentChunk:
    """
    Represents a chunk of a document with context information.
    
    Attributes:
        text (str): The text content of the chunk
        metadata (dict): Metadata inherited from parent document plus chunk-specific info
        chunk_id (str): Unique identifier for this chunk
        parent_doc_id (str): ID of the parent document
    """
    
    def __init__(self, text: str, parent_doc_id: str, chunk_index: int, metadata: Optional[Dict] = None):
        """
        Initialize a DocumentChunk object.
        
        Args:
            text: The text content of the chunk
            parent_doc_id: ID of the parent document
            chunk_index: Index of this chunk in the parent document
            metadata: Optional metadata dictionary
        """
        self.text = text
        self.parent_doc_id = parent_doc_id
        self.chunk_index = chunk_index
        self.metadata = metadata or {}
        self.chunk_id = f"{parent_doc_id}_chunk_{chunk_index}"
    
    def __repr__(self):
        return f"DocumentChunk(id={self.chunk_id}, length={len(self.text)})"


class DocumentIngestion:
    """
    Main class for document ingestion and preprocessing.
    
    This class provides methods to load documents from various sources,
    preprocess them, and split them into chunks suitable for embedding.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the DocumentIngestion system.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_formats = ['.txt', '.pdf', '.docx', '.html', '.md']
    
    def load_documents(self, source_path: str) -> List[Document]:
        """
        Load documents from a file or directory.
        
        Args:
            source_path: Path to a file or directory containing documents
            
        Returns:
            List of Document objects
            
        Raises:
            FileNotFoundError: If the source path doesn't exist
            ValueError: If no supported documents are found
        """
        # TODO: Implement document loading logic
        # Steps:
        # 1. Check if source_path is file or directory
        # 2. For directory: recursively find all supported files
        # 3. For each file: call appropriate loader based on extension
        # 4. Extract text and metadata
        # 5. Create Document objects
        
        documents = []
        
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
        
        # Collect all files to process
        files_to_process = []
        if path.is_file():
            files_to_process.append(path)
        else:
            # Recursively find all supported files
            for ext in self.supported_formats:
                files_to_process.extend(path.rglob(f"*{ext}"))
        
        # Process each file
        for file_path in files_to_process:
            try:
                document = self._load_single_document(file_path)
                documents.append(document)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                # Continue processing other files
        
        if not documents:
            raise ValueError(f"No supported documents found in {source_path}")
        
        return documents
    
    def _load_single_document(self, file_path: Path) -> Document:
        """
        Load a single document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document object
        """
        # TODO: Implement file-specific loaders
        # For PDF: Use PyPDF2 or pdfplumber
        # For DOCX: Use python-docx
        # For HTML: Use BeautifulSoup
        # For TXT/MD: Direct read
        
        extension = file_path.suffix.lower()
        
        if extension == '.txt' or extension == '.md':
            content = self._load_text_file(file_path)
        elif extension == '.pdf':
            content = self._load_pdf_file(file_path)
        elif extension == '.docx':
            content = self._load_docx_file(file_path)
        elif extension == '.html':
            content = self._load_html_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
        
        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'format': extension,
            'size': file_path.stat().st_size
        }
        
        return Document(content=content, metadata=metadata)
    
    def _load_text_file(self, file_path: Path) -> str:
        """Load content from a text file."""
        # TODO: Implement text file loading with encoding detection
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_pdf_file(self, file_path: Path) -> str:
        """Load content from a PDF file."""
        # TODO: Implement PDF loading using PyPDF2 or pdfplumber
        # Example:
        # import PyPDF2
        # with open(file_path, 'rb') as f:
        #     pdf_reader = PyPDF2.PdfReader(f)
        #     text = ""
        #     for page in pdf_reader.pages:
        #         text += page.extract_text()
        #     return text
        raise NotImplementedError("PDF loading requires PyPDF2 or pdfplumber library")
    
    def _load_docx_file(self, file_path: Path) -> str:
        """Load content from a DOCX file."""
        # TODO: Implement DOCX loading using python-docx
        # Example:
        # from docx import Document as DocxDocument
        # doc = DocxDocument(file_path)
        # return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        raise NotImplementedError("DOCX loading requires python-docx library")
    
    def _load_html_file(self, file_path: Path) -> str:
        """Load content from an HTML file."""
        # TODO: Implement HTML loading using BeautifulSoup
        # Example:
        # from bs4 import BeautifulSoup
        # with open(file_path, 'r', encoding='utf-8') as f:
        #     soup = BeautifulSoup(f.read(), 'html.parser')
        #     return soup.get_text()
        raise NotImplementedError("HTML loading requires BeautifulSoup library")
    
    def chunk_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """
        Split documents into smaller chunks for embedding.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of DocumentChunk objects
        """
        # TODO: Implement intelligent chunking strategy
        # Considerations:
        # 1. Respect sentence boundaries
        # 2. Maintain context with overlap
        # 3. Handle special cases (code blocks, tables)
        # 4. Preserve metadata
        
        all_chunks = []
        
        for document in documents:
            chunks = self._chunk_single_document(document)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _chunk_single_document(self, document: Document) -> List[DocumentChunk]:
        """
        Split a single document into chunks.
        
        Args:
            document: Document object to chunk
            
        Returns:
            List of DocumentChunk objects
        """
        # TODO: Implement smart chunking algorithm
        # Options:
        # 1. Character-based with overlap
        # 2. Sentence-based chunking
        # 3. Paragraph-based chunking
        # 4. Semantic chunking using NLP
        
        text = document.content
        chunks = []
        
        # Simple character-based chunking with overlap
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                break_point = max(last_period, last_newline)
                
                if break_point > start:
                    end = break_point + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunk_metadata = document.metadata.copy()
                chunk_metadata['chunk_index'] = chunk_index
                chunk_metadata['start_char'] = start
                chunk_metadata['end_char'] = end
                
                chunk = DocumentChunk(
                    text=chunk_text,
                    parent_doc_id=document.doc_id,
                    chunk_index=chunk_index,
                    metadata=chunk_metadata
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text to clean and normalize it.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        # TODO: Implement text preprocessing
        # Steps:
        # 1. Remove extra whitespace
        # 2. Fix encoding issues
        # 3. Remove special characters (if needed)
        # 4. Normalize punctuation
        # 5. Handle case normalization (optional)
        
        # Basic preprocessing
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        # Remove multiple newlines
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        
        return text.strip()


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the DocumentIngestion module.
    This demonstrates how to load and chunk documents.
    """
    
    # Initialize the ingestion system
    ingestion = DocumentIngestion(chunk_size=500, chunk_overlap=50)
    
    # Example: Load documents from a directory
    # documents = ingestion.load_documents("data/")
    
    # Example: Process a single document
    sample_text = """
    This is a sample document for the RAG system.
    It demonstrates how documents are loaded and chunked.
    The system will split this into smaller pieces for embedding.
    Each chunk will maintain context through overlap.
    This helps the retrieval system find relevant information.
    """
    
    sample_doc = Document(
        content=sample_text,
        metadata={'source': 'example', 'type': 'demonstration'}
    )
    
    # Chunk the document
    chunks = ingestion.chunk_documents([sample_doc])
    
    print(f"Created {len(chunks)} chunks from the sample document")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}:")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  Text: {chunk.text[:100]}...")
        print(f"  Metadata: {chunk.metadata}")
