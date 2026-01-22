#!/usr/bin/env python3
"""
Quick Start Script for RAG System

This script helps you get started with the RAG system quickly by:
1. Checking your environment
2. Guiding you through setup
3. Running a simple demo

Author: Akash Thanda
Course: CSE435 - Information Retrieval
"""

import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def check_python_version():
    """Check if Python version is 3.9 or higher."""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9 or higher is required!")
        print("Please upgrade Python and try again.")
        return False
    else:
        print("✓ Python version is compatible")
        return True


def check_virtual_environment():
    """Check if running in a virtual environment."""
    print_header("Checking Virtual Environment")
    
    in_venv = (hasattr(sys, 'real_prefix') or 
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    
    if in_venv:
        print("✓ Running in virtual environment")
        return True
    else:
        print("⚠ Not running in a virtual environment")
        print("\nRecommendation: Create a virtual environment to avoid conflicts")
        print("  python -m venv venv")
        print("  source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        
        response = input("\nContinue anyway? (y/n): ").strip().lower()
        return response == 'y'


def check_dependencies():
    """Check if required dependencies are installed."""
    print_header("Checking Dependencies")
    
    try:
        import numpy
        print("✓ numpy is installed")
        has_numpy = True
    except ImportError:
        print("✗ numpy is not installed")
        has_numpy = False
    
    if not has_numpy:
        print("\n⚠ Required dependencies are missing")
        print("\nTo install dependencies:")
        print("  pip install -r requirements.txt")
        
        response = input("\nInstall now? (y/n): ").strip().lower()
        if response == 'y':
            print("\nInstalling dependencies...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
                print("✓ Dependencies installed successfully")
                return True
            except subprocess.CalledProcessError:
                print("❌ Failed to install dependencies")
                return False
        else:
            return False
    else:
        return True


def setup_env_file():
    """Guide user through setting up .env file."""
    print_header("Environment Setup")
    
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_file.exists():
        print("✓ .env file already exists")
        return True
    
    if not env_example.exists():
        print("⚠ .env.example not found")
        return False
    
    print("The system can use API-based models (OpenAI) or local models.")
    print("\nFor API-based models, you'll need:")
    print("  - OpenAI API key (for GPT models)")
    print("\nFor local models, you can skip this step.")
    
    response = input("\nDo you have API keys to configure? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\nCopying .env.example to .env...")
        with open(env_example, 'r') as src:
            content = src.read()
        
        print("\nPlease enter your API keys (or press Enter to skip):")
        
        openai_key = input("OpenAI API Key: ").strip()
        if openai_key:
            content = content.replace('your_openai_api_key_here', openai_key)
        
        with open(env_file, 'w') as dst:
            dst.write(content)
        
        print("✓ .env file created")
    else:
        print("Skipping .env setup - system will use placeholder implementations")
    
    return True


def show_next_steps():
    """Show user what to do next."""
    print_header("Setup Complete!")
    
    print("\n✅ Your RAG system is ready to explore!")
    
    print("\nNext Steps:")
    print("\n1. Run the example workflow:")
    print("     python example_workflow.py")
    
    print("\n2. Explore the codebase:")
    print("     - src/ingestion.py - Document loading and chunking")
    print("     - src/embeddings.py - Vector embedding generation")
    print("     - src/retrieval.py - Vector search and retrieval")
    print("     - src/response_generation.py - Answer generation")
    
    print("\n3. Add your own documents:")
    print("     - Place documents in the data/ directory")
    print("     - See data/README.md for supported formats")
    
    print("\n4. Customize the system:")
    print("     - Edit configuration in .env file")
    print("     - Implement TODO sections in the code")
    print("     - Adjust chunk sizes and parameters")
    
    print("\n5. Read the documentation:")
    print("     - README.md - Complete system overview")
    print("     - ETHICS.md - Ethical AI practices")
    print("     - CONTRIBUTING.md - Contribution guidelines")
    
    print("\n6. For help:")
    print("     - Check the documentation")
    print("     - Open an issue on GitHub")
    print("     - Contact course instructor (for students)")
    
    print("\nHappy coding! 🚀")


def main():
    """Main function to run setup checks."""
    print_header("RAG System Quick Start")
    print("\nWelcome to the RAG System for Domain-Specific Question Answering!")
    print("This script will help you get started.")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check virtual environment
    if not check_virtual_environment():
        print("\nSetup cancelled.")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install dependencies and run this script again.")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Setup .env file
    setup_env_file()
    
    # Show next steps
    show_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
