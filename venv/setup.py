"""Package setup configuration."""

from setuptools import setup, find_packages

setup(
    name="customer-support-ai",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "langchain>=0.0.300",
        "langchain-community>=0.0.10",
        "langchain-huggingface>=0.0.5",
        "chromadb>=0.4.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
        "python-dotenv>=1.0.0"
    ],
    python_requires=">=3.8",
)