"""
EvoAttention: Evolutionary Discovery of Attention Mechanisms
"""

__version__ = "0.1.0"

from .search_space import AttentionGene
from .attention import EvolvedAttention
from .model import TinyTransformer
from .evolution import Evolution, Individual
from .config import Config

__all__ = [
    'AttentionGene',
    'EvolvedAttention', 
    'TinyTransformer',
    'Evolution',
    'Individual',
    'Config'
]
