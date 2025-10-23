"""
Configuration management for EvoAttention.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration for evolutionary attention search."""
    
    # Model architecture
    vocab_size: int = 10000
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 128
    dropout: float = 0.1
    
    # Training
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 3e-4
    train_steps: int = 5000
    eval_steps: int = 200
    warmup_steps: int = 500
    
    # Evolution
    population_size: int = 12
    n_generations: int = 10
    elite_size: int = 3
    mutation_rate: float = 0.3
    
    # System
    device: str = "cuda"  # Will auto-detect in code
    checkpoint_dir: str = "./results/generation_checkpoints"
    seed: int = 42
    
    def __post_init__(self):
        """Ensure checkpoint directory exists."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
