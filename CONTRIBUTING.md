# Contributing to RAG Domain-Specific QA System

Thank you for your interest in contributing to this RAG (Retrieval-Augmented Generation) system! This document provides guidelines for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Academic Integrity](#academic-integrity)

---

## Code of Conduct

This project follows academic and professional standards of conduct:
- Be respectful and inclusive in all interactions
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other contributors
- Follow ethical AI development practices

## Getting Started

### Prerequisites
- Python 3.9 or higher
- Git for version control
- Basic understanding of RAG systems and NLP
- Familiarity with the technologies in the stack

### Setting Up Development Environment

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/rag-aq-domain-system.git
   cd rag-aq-domain-system
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys if needed
   ```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

1. **Code Implementations**
   - Implementing TODO sections in modules
   - Adding new features
   - Optimizing existing code
   - Bug fixes

2. **Documentation**
   - Improving README and guides
   - Adding code comments
   - Creating tutorials
   - Writing technical blogs

3. **Testing**
   - Writing unit tests
   - Integration testing
   - Performance benchmarking
   - Bug reporting

4. **Research**
   - Experimenting with new techniques
   - Benchmarking different approaches
   - Documenting findings

### Areas Needing Contribution

Current TODOs in the codebase:
- [ ] Implement PDF loading in `src/ingestion.py`
- [ ] Implement DOCX loading in `src/ingestion.py`
- [ ] Implement HTML parsing in `src/ingestion.py`
- [ ] Add actual embedding model integration in `src/embeddings.py`
- [ ] Implement FAISS index in `src/retrieval.py`
- [ ] Add OpenAI API integration in `src/response_generation.py`
- [ ] Implement streaming responses in `src/response_generation.py`
- [ ] Add evaluation metrics
- [ ] Create web interface (Streamlit/FastAPI)
- [ ] Add unit tests

## Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `test/` - Testing additions
- `refactor/` - Code refactoring

### 2. Make Changes
- Write clean, readable code
- Follow existing code structure
- Add comments for complex logic
- Update documentation as needed

### 3. Test Your Changes
```bash
# Run tests (when available)
pytest

# Test your specific module
python -m src.your_module

# Run the example workflow
python example_workflow.py
```

### 4. Commit Your Changes
```bash
git add .
git commit -m "Brief description of changes"
```

Commit message format:
```
Type: Brief description (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain what and why, not how.

- Bullet points are fine
- Use present tense ("Add feature" not "Added feature")
- Reference issues: Fixes #123
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title and description
- Reference to related issues
- Screenshots if UI changes
- Testing steps

## Coding Standards

### Python Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Maximum line length: 100 characters
- Use docstrings for all functions and classes

### Docstring Format
```python
def function_name(arg1: str, arg2: int) -> bool:
    """
    Brief description of function.
    
    More detailed description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When and why this is raised
    """
    pass
```

### Code Organization
- One class per file (unless closely related)
- Group related functions together
- Use type hints for function signatures
- Keep functions focused and small

### Comments
- Explain WHY, not WHAT
- Update comments when code changes
- Use TODO comments for planned improvements
- Add references for complex algorithms

## Testing

### Writing Tests
- Place tests in `tests/` directory (create if needed)
- Name test files `test_*.py`
- Use descriptive test names
- Test edge cases and error conditions

### Test Example
```python
import pytest
from src.ingestion import DocumentIngestion

def test_chunk_document():
    """Test that documents are chunked correctly."""
    ingestion = DocumentIngestion(chunk_size=100, chunk_overlap=10)
    # ... test implementation
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ingestion.py

# Run with coverage
pytest --cov=src
```

## Documentation

### Code Documentation
- Add docstrings to all public functions and classes
- Include usage examples in docstrings
- Document parameters, return values, and exceptions
- Keep docstrings up to date with code changes

### README Updates
When adding new features:
- Update relevant sections in README.md
- Add to the Table of Contents if needed
- Include usage examples
- Update the tech stack if adding dependencies

### Creating Tutorials
- Write step-by-step guides
- Include code examples
- Add screenshots where helpful
- Test all steps before submitting

## Academic Integrity

### For Students
If you're a student working on this project:
- Clearly indicate which parts are your contributions
- Cite any external resources or papers you reference
- Don't copy code without understanding it
- Discuss contributions with your instructor if for coursework

### For Researchers
- Cite relevant papers in code comments
- Document novel approaches
- Share experimental results
- Be transparent about limitations

### Ethical Considerations
Before contributing, ensure your changes:
- Don't introduce biases
- Respect privacy and data protection
- Follow responsible AI practices
- Are properly licensed
- Don't violate any terms of service

## Questions and Support

### Getting Help
- Check existing documentation first
- Search for similar issues on GitHub
- Ask in course forums (for students)
- Create a GitHub issue for bugs or feature requests

### Reporting Issues
When reporting bugs, include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Error messages and stack traces

### Feature Requests
For feature requests, explain:
- What problem it solves
- How it fits with project goals
- Proposed implementation approach
- Any alternatives considered

## Recognition

Contributors will be:
- Listed in the project contributors
- Mentioned in release notes for significant contributions
- Cited in academic papers if work is published

---

## Thank You!

Your contributions help make this project better for everyone. Whether you're fixing a typo, implementing a feature, or improving documentation, every contribution is valued!

For questions, contact:
- Course instructor (for students)
- Repository maintainer
- Open a GitHub issue

---

**Note:** This is an educational project for CSE435. All contributors should maintain academic integrity and follow course guidelines.
