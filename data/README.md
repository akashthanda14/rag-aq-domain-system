# Sample Data Directory

This directory is for storing your domain-specific documents that will be ingested into the RAG system.

## Supported Formats

- `.txt` - Plain text files
- `.pdf` - PDF documents (requires PyPDF2)
- `.docx` - Microsoft Word documents (requires python-docx)
- `.html` - HTML files (requires BeautifulSoup)
- `.md` - Markdown files

## Usage

1. Place your documents in this directory
2. Run the ingestion script:
   ```python
   from src.ingestion import DocumentIngestion
   
   ingestion = DocumentIngestion()
   documents = ingestion.load_documents("data/")
   ```

3. The system will recursively find all supported files and process them

## Best Practices

- **Organize by topic**: Create subdirectories for different topics
- **Name files clearly**: Use descriptive filenames
- **Check permissions**: Ensure you have rights to use the documents
- **Quality over quantity**: Curated, high-quality sources produce better results
- **Version control**: Keep track of document versions and updates
- **Metadata**: Include source attribution in filenames or metadata

## Example Structure

```
data/
├── computer_science/
│   ├── algorithms.pdf
│   ├── data_structures.txt
│   └── machine_learning.pdf
├── information_retrieval/
│   ├── ir_basics.txt
│   ├── vector_search.md
│   └── embeddings.pdf
└── sample_document.txt
```

## Sample Document

A sample document (`sample_document.txt`) is provided to test the system without additional setup.
