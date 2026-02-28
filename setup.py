"""Setup script for the WELDE package."""
from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="welde",
    version="1.0.0",
    description="Weighted Ensemble Loss with Diversity Enhancement for imbalanced object detection in medical imaging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Farhat Masood",
    url="https://github.com/farhatmasood/welde",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "numpy>=1.22.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "tqdm>=4.64.0",
        "Pillow>=9.0.0",
    ],
    extras_require={
        "external": ["medmnist>=2.2.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="long-tailed learning, ensemble, medical imaging, spine, imbalanced classification",
)
