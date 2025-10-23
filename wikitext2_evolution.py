"""
Run evolutionary attention discovery on WikiText-2.

This is the main experiment script that runs the complete
evolutionary search for attention mechanisms.

Usage:
    python experiments/wikitext2_evolution.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from evo_attention import Config, Evolution
from evo_attention.utils import set_seed, get_dataloaders, get_device


def main():
    """Run the evolutionary attention discovery experiment."""
    
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
        
        # Evolution
        population_size=12,
        n_generations=10,
        elite_size=3,
        mutation_rate=0.3,
        
        # System
        device=get_device(),
        checkpoint_dir="./results/generation_checkpoints",
        seed=42
    )
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Print configuration
    print("=" * 70)
    print("EVOLUTIONARY ATTENTION DISCOVERY - WIKITEXT-2")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Model: {config.n_layers} layers, {config.d_model}d, {config.n_heads} heads")
    print(f"   Training: {config.train_steps} steps, lr={config.learning_rate}")
    print(f"   Evolution: {config.population_size} individuals, {config.n_generations} generations")
    print(f"   Device: {config.device}")
    print(f"   Seed: {config.seed}")
    
    # Create data loaders
    print(f"\n📚 Loading data...")
    train_loader, eval_loader = get_dataloaders(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        use_wikitext=True
    )
    
    # Check for existing checkpoints
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("gen_*.pkl"))
    
    if checkpoints:
        print(f"\n💾 Found {len(checkpoints)} existing checkpoints")
        response = input("Resume from last checkpoint? (y/n): ").strip().lower()
        
        if response == 'y':
            # Find latest checkpoint
            latest_gen = max([
                int(p.stem.split('_')[1]) 
                for p in checkpoints
            ])
            
            # Create evolution and load checkpoint
            evo = Evolution(config, train_loader, eval_loader, use_wikitext=True)
            evo.load_checkpoint(latest_gen)
            
            print(f"✅ Resumed from generation {latest_gen}")
            
            # Continue evolution
            for gen in range(latest_gen + 1, config.n_generations):
                evo.evolve()
                evo.evaluate_population()
                evo.save_checkpoint()
        else:
            print("🆕 Starting fresh evolution")
            evo = Evolution(config, train_loader, eval_loader, use_wikitext=True)
            evo.run()
    else:
        # No checkpoints, start fresh
        print("\n🆕 Starting fresh evolution")
        evo = Evolution(config, train_loader, eval_loader, use_wikitext=True)
        evo.run()
    
    # Save final results
    print("\n💾 Saving final results...")
    evo.save_results(output_dir="./results")
    
    print("\n✅ Experiment complete!")
    print(f"   Results saved to: ./results/")
    print(f"   Checkpoints saved to: {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
