"""Setup configuration for Collage Toolkit."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="collage-toolkit",
    version="0.1.0",
    author="Collage Lab",
    author_email="",
    description="Collage Toolkit — continual learning framework from Collage Lab",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.vincenzolomonaco.com/collage-lab",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "peft>=0.5.0",
        "datasets>=2.14.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "wandb>=0.15.0",
        "tensorboard>=2.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "isort>=5.12.0",
        ],
        "vision": [
            "timm>=0.9.0",
            "opencv-python>=4.8.0",
        ],
        "language": [
            "sentencepiece>=0.1.99",
            "tokenizers>=0.13.0",
        ],
    },
)
