"""
Utility Functions

This module contains helper functions and utilities used across the RAG system.

Author: CSE435 Project Team
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    TODO:
        - Implement YAML loading
        - Add config validation
        - Support environment variable substitution
    """
    config = {}
    
    # PLACEHOLDER: Implement config loading
    # 
    # import yaml
    # 
    # with open(config_path, 'r') as f:
    #     config = yaml.safe_load(f)
    
    return config


def get_api_key(service: str) -> Optional[str]:
    """
    Securely retrieve API keys from environment variables.
    
    Args:
        service: Service name (e.g., 'OPENAI', 'PINECONE')
        
    Returns:
        API key string or None if not found
        
    Example:
        >>> api_key = get_api_key('OPENAI')
        >>> if api_key:
        ...     # Use the API key
    """
    key_name = f"{service.upper()}_API_KEY"
    api_key = os.getenv(key_name)
    
    if not api_key:
        logger.warning(f"API key not found: {key_name}")
    
    return api_key


def setup_logging(
    log_file: Optional[str] = None,
    level: str = "INFO"
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        log_file: Optional path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
        
    TODO:
        - Implement file logging
        - Add log rotation
        - Configure log format
    """
    log_level = getattr(logging, level.upper())
    
    # PLACEHOLDER: Implement advanced logging setup
    # 
    # if log_file:
    #     from logging.handlers import RotatingFileHandler
    #     
    #     handler = RotatingFileHandler(
    #         log_file,
    #         maxBytes=10485760,  # 10MB
    #         backupCount=5
    #     )
    #     handler.setLevel(log_level)
    #     
    #     formatter = logging.Formatter(
    #         '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #     )
    #     handler.setFormatter(formatter)
    #     
    #     logger = logging.getLogger()
    #     logger.addHandler(handler)
    
    return logger


def create_directory(path: str, exist_ok: bool = True):
    """
    Create a directory if it doesn't exist.
    
    Args:
        path: Directory path to create
        exist_ok: Whether to ignore if directory exists
    """
    Path(path).mkdir(parents=True, exist_ok=exist_ok)
    logger.debug(f"Directory created/verified: {path}")


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Estimate token count for a text string.
    
    Important for staying within model token limits.
    
    Args:
        text: Input text
        model: Model name for tokenizer selection
        
    Returns:
        Approximate token count
        
    TODO:
        - Implement tiktoken for OpenAI models
        - Add support for other tokenizers
        - Cache tokenizer instances
    """
    # PLACEHOLDER: Implement token counting
    # 
    # try:
    #     import tiktoken
    #     
    #     encoding = tiktoken.encoding_for_model(model)
    #     tokens = encoding.encode(text)
    #     return len(tokens)
    # 
    # except ImportError:
    #     # Fallback: rough approximation
    #     return len(text.split()) * 1.3
    
    # Rough approximation
    return int(len(text.split()) * 1.3)


def truncate_text(
    text: str,
    max_tokens: int = 1000,
    model: str = "gpt-3.5-turbo"
) -> str:
    """
    Truncate text to fit within token limit.
    
    Args:
        text: Input text
        max_tokens: Maximum tokens allowed
        model: Model for tokenization
        
    Returns:
        Truncated text
        
    TODO:
        - Implement proper truncation using tokenizer
        - Preserve sentence boundaries
        - Add truncation indicator
    """
    # PLACEHOLDER: Implement smart truncation
    
    token_count = count_tokens(text, model)
    if token_count <= max_tokens:
        return text
    
    # Rough truncation
    ratio = max_tokens / token_count
    char_limit = int(len(text) * ratio * 0.9)  # Safety margin
    return text[:char_limit] + "..."


def validate_environment():
    """
    Validate that required environment variables and dependencies are set.
    
    Checks for:
    - Required environment variables
    - Required Python packages
    - File/directory structure
    
    Raises:
        EnvironmentError: If validation fails
        
    TODO:
        - Check for required API keys
        - Verify package installations
        - Validate directory structure
    """
    # PLACEHOLDER: Implement environment validation
    # 
    # required_vars = ['OPENAI_API_KEY']  # Add others as needed
    # missing_vars = []
    # 
    # for var in required_vars:
    #     if not os.getenv(var):
    #         missing_vars.append(var)
    # 
    # if missing_vars:
    #     raise EnvironmentError(
    #         f"Missing required environment variables: {', '.join(missing_vars)}"
    #     )
    
    pass


class PerformanceTimer:
    """
    Context manager for timing code execution.
    
    Example:
        >>> with PerformanceTimer("Document loading"):
        ...     load_documents()
        Document loading completed in 2.34 seconds
    """
    
    def __init__(self, operation_name: str = "Operation"):
        """
        Initialize timer.
        
        Args:
            operation_name: Name of the operation being timed
        """
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start the timer."""
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer and log the duration."""
        import time
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"{self.operation_name} completed in {duration:.2f} seconds")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time is None:
            import time
            return time.time() - self.start_time
        return self.end_time - self.start_time
