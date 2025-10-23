"""
Transformer model implementation with evolved attention.
"""

import torch
import torch.nn as nn

from .attention import EvolvedAttention
from .search_space import AttentionGene
from .config import Config


class TransformerBlock(nn.Module):
    """
    Single transformer block with evolved attention.
    """
    
    def __init__(self, d_model: int, n_heads: int, gene: AttentionGene, dropout: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            gene: Gene specifying attention mechanism
            dropout: Dropout probability
        """
        super().__init__()
        self.attn = EvolvedAttention(d_model, n_heads, gene)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with pre-norm architecture.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Self-attention with residual
        x = x + self.attn(self.ln1(x), mask)
        
        # Feed-forward with residual
        x = x + self.ffn(self.ln2(x))
        
        return x


class TinyTransformer(nn.Module):
    """
    Small transformer model for language modeling.
    
    This is designed to be small enough to train quickly for
    evolutionary search, while still being expressive enough
    to differentiate between attention mechanisms.
    """
    
    def __init__(self, config: Config, gene: AttentionGene):
        """
        Initialize transformer model.
        
        Args:
            config: Model configuration
            gene: Gene specifying attention mechanism for all layers
        """
        super().__init__()
        self.config = config
        self.gene = gene
        
        # Token and position embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks (all use the same attention mechanism)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, gene, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying (optional but helps small models)
        self.head.weight = self.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with small values."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for language modeling.
        
        Args:
            x: Input token ids [batch, seq_len]
            
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        bsz, seq_len = x.shape
        device = x.device
        
        # Get embeddings
        tok_emb = self.embedding(x)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        
        x = self.dropout(tok_emb + pos_emb)
        
        # Create causal mask (prevent attending to future tokens)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
