"""
SymSorter - A CLIP-based image classification and similarity tool

SymSorter provides intelligent image sorting and classification using CLIP embeddings
for semantic similarity analysis. Perfect for research workflows involving large
image collections.
"""

__version__ = "0.1.0"
__author__ = "Nick Mortimer"
__email__ = "nick.mortimer@csiro.au"

from .image_browser import ImageBrowser
from .clip_encode import load_existing_embeddings

__all__ = ["ImageBrowser", "load_existing_embeddings"]
