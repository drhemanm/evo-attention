"""
Train and evaluate baseline transformer on WikiText-2.

This script trains a standard transformer with vanilla attention
to establish a baseline for comparison.

Usage:
    python experiments/wikitext2_baseline.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from evo_attention import Config
from evo_attention.model import TinyTransformer
from evo_attention.search_space import AttentionGene
from evo_attention.training import train_and_evaluate
from evo_attention.utils import set_seed, get_dataloaders, get_device, count_parameters


def main():
    """Run baseline experiment."""
    
    # Configuration
    config = Config(
        # Model
        vocab_size=10000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_seq_len=128,
        dropout=0.1,
        
        # Training
        batch_size=32,
        eval_batch_size=64,
        learning_rate=3e-4,
        train_steps=5000,
        eval_steps=200,
        
        # System
        device=get_device(),
        seed=42
    )
    
    # Set seed
    set_seed(config.seed)
    
    # Print info
    print("=" * 70)
    print("BASELINE TRANSFORMER - WIKITEXT-2")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Model: {config.n_layers} layers, {config.d_model}d, {config.n_heads} heads")
    print(f"   Training: {config.train_steps} steps, lr={config.learning_rate}")
    print(f"   Device: {config.device}")
    
    # Load data
    print(f"\n📚 Loading data...")
    train_loader, eval_loader = get_dataloaders(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        use_wikitext=True
    )
    
    # Create baseline model (vanilla transformer attention)
    print(f"\n🔧 Creating baseline model...")
    baseline_gene = AttentionGene(gene_dict={
        'similarity': 'dot',
        'normalization': 'softmax',
        'gating': 'none',
        'temperature_mode': 'fixed',
        'use_bias': True,
        'attention_dropout': 0.1
    })
    
    model = TinyTransformer(config, baseline_gene)
    num_params = count_parameters(model)
    
    print(f"   Architecture: {baseline_gene}")
    print(f"   Parameters: {num_params:,}")
    
    # Train and evaluate
    print(f"\n🚀 Training baseline model...")
    fitness, perplexity, train_loss = train_and_evaluate(
        model,
        train_loader,
        eval_loader,
        config,
        gene_id=0
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)
    print(f"   Perplexity: {perplexity:.2f}")
    print(f"   Train Loss: {train_loss:.4f}")
    print(f"   Fitness: {fitness:.4f}")
    
    # Save results
    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    baseline_results = {
        'gene': baseline_gene.to_dict(),
        'perplexity': float(perplexity),
        'train_loss': float(train_loss),
        'fitness': float(fitness),
        'num_parameters': num_params,
        'config': {
            'n_layers': config.n_layers,
            'd_model': config.d_model,
            'n_heads': config.n_heads,
            'train_steps': config.train_steps
        }
    }
    
    with open(results_dir / 'baseline_results.json', 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_dir / 'baseline_results.json'}")
    print("\n✅ Baseline experiment complete!")


if __name__ == "__main__":
    main()
