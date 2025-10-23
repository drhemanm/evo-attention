"""
Basic tests for EvoAttention components.

Run with: pytest tests/
"""

import pytest
import torch
from evo_attention import Config, AttentionGene, EvolvedAttention, TinyTransformer


def test_attention_gene_creation():
    """Test that we can create attention genes."""
    gene = AttentionGene()
    
    assert gene.similarity in AttentionGene.SIMILARITY_FUNCS
    assert gene.normalization in AttentionGene.NORMALIZATIONS
    assert gene.gating in AttentionGene.GATING
    assert gene.temperature_mode in AttentionGene.TEMPERATURE_MODES


def test_attention_gene_mutation():
    """Test gene mutation."""
    gene1 = AttentionGene()
    gene2 = gene1.mutate(mutation_rate=1.0)  # Always mutate
    
    # Should be different (with high probability)
    assert gene1.to_dict() != gene2.to_dict()


def test_attention_gene_crossover():
    """Test gene crossover."""
    gene1 = AttentionGene(gene_dict={
        'similarity': 'dot',
        'normalization': 'softmax',
        'gating': 'none',
        'temperature_mode': 'fixed',
        'use_bias': True,
        'attention_dropout': 0.1
    })
    
    gene2 = AttentionGene(gene_dict={
        'similarity': 'additive',
        'normalization': 'sparsemax',
        'gating': 'output_gate',
        'temperature_mode': 'learned',
        'use_bias': False,
        'attention_dropout': 0.2
    })
    
    child = gene1.crossover(gene2)
    
    # Child should have valid components
    assert child.similarity in AttentionGene.SIMILARITY_FUNCS
    assert child.normalization in AttentionGene.NORMALIZATIONS


def test_evolved_attention_forward():
    """Test that evolved attention can do a forward pass."""
    config = Config()
    gene = AttentionGene()
    
    attention = EvolvedAttention(config.d_model, config.n_heads, gene)
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    # Forward pass
    output = attention(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, config.d_model)


def test_transformer_forward():
    """Test that transformer can do a forward pass."""
    config = Config()
    gene = AttentionGene()
    
    model = TinyTransformer(config, gene)
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(x)
    
    # Check output shape
    assert logits.shape == (batch_size, seq_len, config.vocab_size)


def test_baseline_attention():
    """Test that baseline (vanilla) attention works."""
    config = Config()
    
    baseline_gene = AttentionGene(gene_dict={
        'similarity': 'dot',
        'normalization': 'softmax',
        'gating': 'none',
        'temperature_mode': 'fixed',
        'use_bias': True,
        'attention_dropout': 0.1
    })
    
    model = TinyTransformer(config, baseline_gene)
    
    batch_size = 2
    seq_len = 10
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits = model(x)
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size)


def test_different_similarity_functions():
    """Test that all similarity functions work."""
    config = Config()
    
    for similarity in AttentionGene.SIMILARITY_FUNCS:
        gene = AttentionGene(gene_dict={
            'similarity': similarity,
            'normalization': 'softmax',
            'gating': 'none',
            'temperature_mode': 'fixed',
            'use_bias': True,
            'attention_dropout': 0.1
        })
        
        attention = EvolvedAttention(config.d_model, config.n_heads, gene)
        x = torch.randn(2, 10, config.d_model)
        
        try:
            output = attention(x)
            assert output.shape == x.shape
        except Exception as e:
            pytest.fail(f"Similarity '{similarity}' failed: {e}")


def test_different_normalizations():
    """Test that all normalization functions work."""
    config = Config()
    
    for normalization in AttentionGene.NORMALIZATIONS:
        gene = AttentionGene(gene_dict={
            'similarity': 'dot',
            'normalization': normalization,
            'gating': 'none',
            'temperature_mode': 'fixed',
            'use_bias': True,
            'attention_dropout': 0.1
        })
        
        attention = EvolvedAttention(config.d_model, config.n_heads, gene)
        x = torch.randn(2, 10, config.d_model)
        
        try:
            output = attention(x)
            assert output.shape == x.shape
        except Exception as e:
            pytest.fail(f"Normalization '{normalization}' failed: {e}")


def test_config_creation():
    """Test that config can be created."""
    config = Config()
    
    assert config.vocab_size == 10000
    assert config.d_model == 128
    assert config.n_heads == 4
    assert config.population_size == 12


def test_model_parameter_count():
    """Test that we can count model parameters."""
    config = Config()
    gene = AttentionGene()
    model = TinyTransformer(config, gene)
    
    num_params = sum(p.numel() for p in model.parameters())
    
    # Should be around 500K parameters
    assert 400_000 < num_params < 600_000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
