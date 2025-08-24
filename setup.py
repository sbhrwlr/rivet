"""Setup configuration for Rivet framework."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rivet-ai",
    version="0.1.0",
    author="Rivet Team",
    author_email="hello@rivet.ai",
    description="A lightweight, developer-first framework to build AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rivet-ai/rivet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Minimal dependencies - keep it lightweight
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "async": ["aiofiles>=0.8.0"],
        "full": ["openai>=1.0.0", "aiofiles>=0.8.0", "tiktoken>=0.4.0"],
        "dev": ["pytest>=6.0", "pytest-asyncio>=0.21.0", "black", "flake8", "mypy"],
    },
)