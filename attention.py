"""
Evolved attention module implementation.
Implements attention mechanisms based on gene specifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .search_space import AttentionGene


class EvolvedAttention(nn.Module):
    """
    Attention module that implements a gene-specified mechanism.
    
    This module can implement various attention mechanisms based on
    the provided gene, including different similarity functions,
    normalizations, gating mechanisms, and temperature modes.
    """
    
    def __init__(self, d_model: int, n_heads: int, gene: AttentionGene):
        """
        Initialize evolved attention module.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            gene: Gene specifying the attention mechanism
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.gene = gene
        
        # Standard Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=gene.use_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=gene.use_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=gene.use_bias)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Additive attention components
        if gene.similarity == 'additive':
            self.attn_fc = nn.Linear(2 * self.d_head, self.d_head)
            self.attn_v = nn.Linear(self.d_head, 1, bias=False)
        
        # Temperature modes
        if gene.temperature_mode == 'learned':
            self.temperature = nn.Parameter(torch.ones(1))
        elif gene.temperature_mode == 'adaptive':
            self.temp_proj = nn.Linear(d_model, 1)
        else:
            self.temperature = np.sqrt(self.d_head)
        
        # Gating mechanisms
        if gene.gating == 'input_gate':
            self.gate = nn.Linear(d_model, d_model)
        elif gene.gating == 'output_gate':
            self.out_gate = nn.Linear(d_model, d_model)
        elif gene.gating == 'highway':
            self.highway_gate = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(gene.attention_dropout)
    
    def compute_similarity(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores based on gene's similarity function.
        
        Args:
            q: Query tensor [batch, heads, seq_len, d_head]
            k: Key tensor [batch, heads, seq_len, d_head]
            
        Returns:
            Attention scores [batch, heads, seq_len, seq_len]
        """
        if self.gene.similarity == 'dot':
            scores = torch.matmul(q, k.transpose(-2, -1))
            
        elif self.gene.similarity == 'multiplicative':
            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = scores * torch.sigmoid(scores)
            
        elif self.gene.similarity == 'cosine':
            q_norm = F.normalize(q, dim=-1)
            k_norm = F.normalize(k, dim=-1)
            scores = torch.matmul(q_norm, k_norm.transpose(-2, -1))
            
        elif self.gene.similarity == 'additive':
            # Bahdanau-style attention
            bsz, n_heads, seq_len, d_head = q.shape
            _, _, seq_len_k, _ = k.shape
            
            q_expanded = q.unsqueeze(3).expand(-1, -1, -1, seq_len_k, -1)
            k_expanded = k.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
            combined = torch.cat([q_expanded, k_expanded], dim=-1)
            scores = self.attn_v(torch.tanh(self.attn_fc(combined))).squeeze(-1)
        
        return scores
    
    def apply_normalization(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to attention scores.
        
        Args:
            scores: Attention scores [batch, heads, seq_len, seq_len]
            
        Returns:
            Normalized attention weights
        """
        if self.gene.normalization == 'softmax':
            attn_weights = F.softmax(scores, dim=-1)
            
        elif self.gene.normalization == 'sigmoid':
            attn_weights = torch.sigmoid(scores)
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
            
        elif self.gene.normalization == 'relu_norm':
            attn_weights = F.relu(scores)
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
            
        elif self.gene.normalization == 'sparsemax':
            # Simplified sparsemax (top-k approximation)
            k = max(1, scores.size(-1) // 4)
            topk_vals, topk_idx = torch.topk(scores, k, dim=-1)
            attn_weights = torch.zeros_like(scores)
            attn_weights.scatter_(-1, topk_idx, F.softmax(topk_vals, dim=-1))
        
        return attn_weights
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of evolved attention.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask [batch, 1, seq_len, seq_len]
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        bsz, seq_len, d_model = x.shape
        
        # Optional input gating
        if self.gene.gating == 'input_gate':
            gate = torch.sigmoid(self.gate(x))
            x = x * gate
        
        # Project to Q, K, V
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Compute similarity scores
        scores = self.compute_similarity(q, k)
        
        # Apply temperature scaling
        if self.gene.temperature_mode == 'adaptive':
            temp = torch.sigmoid(self.temp_proj(x.mean(dim=1))) + 0.5
            scores = scores / temp.unsqueeze(1).unsqueeze(2)
        else:
            if isinstance(self.temperature, nn.Parameter):
                temp = torch.abs(self.temperature) + 0.5
            else:
                temp = self.temperature
            scores = scores / temp
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Normalize to get attention weights
        attn_weights = self.apply_normalization(scores)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        out = self.out_proj(out)
        
        # Optional output gating
        if self.gene.gating == 'output_gate':
            gate = torch.sigmoid(self.out_gate(x))
            out = out * gate
        elif self.gene.gating == 'highway':
            gate = torch.sigmoid(self.highway_gate(x))
            out = gate * out + (1 - gate) * x
        
        return out
