"""
Setup script for EvoAttention package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")
else:
    long_description = "Evolutionary Discovery of Attention Mechanisms"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "datasets>=2.14.0",
        "matplotlib>=3.7.0",
    ]

setup(
    name="evo-attention",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Evolutionary Discovery of Attention Mechanisms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/evo-attention",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "pytest>=7.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "evo-attention=experiments.wikitext2_evolution:main",
        ],
    },
)
