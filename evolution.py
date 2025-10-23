"""
Evolutionary algorithm for discovering attention mechanisms.
"""

import random
import pickle
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np
import torch
import matplotlib.pyplot as plt

from .search_space import AttentionGene
from .model import TinyTransformer
from .training import train_and_evaluate
from .config import Config


@dataclass
class Individual:
    """
    Represents an individual in the evolutionary population.
    
    Each individual has a gene (attention mechanism), fitness score,
    and performance metrics.
    """
    gene: AttentionGene
    fitness: float = float('-inf')
    perplexity: float = float('inf')
    train_loss: float = float('inf')
    generation: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert individual to dictionary."""
        return {
            'gene': self.gene.to_dict(),
            'fitness': float(self.fitness),
            'perplexity': float(self.perplexity),
            'train_loss': float(self.train_loss),
            'generation': self.generation
        }


class Evolution:
    """
    Evolutionary algorithm for attention mechanism discovery.
    
    Uses genetic algorithm with:
    - Elitism: Keep top performers
    - Tournament selection: Choose parents from best individuals
    - Crossover: Combine parent genes
    - Mutation: Random changes to genes
    """
    
    def __init__(self, config: Config, train_loader, eval_loader, use_wikitext: bool = True):
        """
        Initialize evolution.
        
        Args:
            config: Evolution configuration
            train_loader: Training data loader
            eval_loader: Evaluation data loader
            use_wikitext: Whether using WikiText dataset (for display)
        """
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.use_wikitext = use_wikitext
        self.population: List[Individual] = []
        self.history: List[Dict[str, Any]] = []
        self.generation = 0
    
    def initialize_population(self):
        """Create initial random population."""
        print(f"🧬 Initializing population of {self.config.population_size}")
        
        for i in range(self.config.population_size):
            gene = AttentionGene()
            individual = Individual(gene=gene, generation=0)
            self.population.append(individual)
            print(f"  Individual {i}: {gene}")
    
    def evaluate_population(self):
        """Evaluate fitness of all individuals in population."""
        print(f"\n📊 Evaluating Generation {self.generation}")
        
        for i, individual in enumerate(self.population):
            # Skip if already evaluated
            if individual.fitness != float('-inf'):
                continue
            
            print(f"\n🧪 Testing Individual {i+1}/{len(self.population)}: {individual.gene}")
            
            # Create model with this gene
            model = TinyTransformer(self.config, individual.gene)
            
            # Train and evaluate
            fitness, perplexity, train_loss = train_and_evaluate(
                model,
                self.train_loader,
                self.eval_loader,
                self.config,
                gene_id=i
            )
            
            # Update individual
            individual.fitness = fitness
            individual.perplexity = perplexity
            individual.train_loss = train_loss
            
            print(f"   ✅ Fitness: {fitness:.4f} | Perplexity: {perplexity:.2f}")
            
            # Free memory
            del model
            torch.cuda.empty_cache()
        
        # Sort population by fitness (best first)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Log generation statistics
        best = self.population[0]
        avg_fitness = np.mean([ind.fitness for ind in self.population])
        
        print(f"\n🏆 Generation {self.generation} Results:")
        print(f"   Best: {best.gene}")
        print(f"   Fitness: {best.fitness:.4f} | Perplexity: {best.perplexity:.2f}")
        print(f"   Avg Fitness: {avg_fitness:.4f}")
        
        # Save to history
        self.history.append({
            'generation': self.generation,
            'best_fitness': float(best.fitness),
            'best_perplexity': float(best.perplexity),
            'avg_fitness': float(avg_fitness),
            'best_gene': best.gene.to_dict()
        })
    
    def evolve(self):
        """Create next generation through selection, crossover, and mutation."""
        print(f"\n🔄 Creating Generation {self.generation + 1}")
        
        # Keep elite individuals
        new_population = self.population[:self.config.elite_size].copy()
        print(f"   Keeping {self.config.elite_size} elite individuals")
        
        # Generate offspring to fill population
        while len(new_population) < self.config.population_size:
            # Tournament selection: choose from top performers
            parent1 = random.choice(self.population[:self.config.elite_size * 2])
            parent2 = random.choice(self.population[:self.config.elite_size * 2])
            
            # Crossover (70% chance) or clone (30% chance)
            if random.random() < 0.7:
                child_gene = parent1.gene.crossover(parent2.gene)
            else:
                child_gene = AttentionGene(gene_dict=parent1.gene.to_dict())
            
            # Mutation
            child_gene = child_gene.mutate(self.config.mutation_rate)
            
            # Create new individual
            child = Individual(gene=child_gene, generation=self.generation + 1)
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
    
    def run(self):
        """Run the complete evolutionary process."""
        print("=" * 70)
        print("🧬 STARTING EVOLUTIONARY ATTENTION DISCOVERY")
        print("=" * 70)
        
        self.initialize_population()
        
        for gen in range(self.config.n_generations):
            self.evaluate_population()
            self.save_checkpoint()
            
            # Create next generation (except on last iteration)
            if gen < self.config.n_generations - 1:
                self.evolve()
        
        print("\n" + "=" * 70)
        print("🎉 EVOLUTION COMPLETE!")
        print("=" * 70)
        self.print_summary()
    
    def save_checkpoint(self):
        """Save current generation state."""
        checkpoint = {
            'generation': self.generation,
            'population': [ind.to_dict() for ind in self.population],
            'history': self.history,
            'config': asdict(self.config)
        }
        
        checkpoint_path = Path(self.config.checkpoint_dir) / f"gen_{self.generation}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, generation: int):
        """
        Load from a saved checkpoint.
        
        Args:
            generation: Generation number to load
        """
        checkpoint_path = Path(self.config.checkpoint_dir) / f"gen_{generation}.pkl"
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.generation = checkpoint['generation']
        self.history = checkpoint['history']
        
        # Reconstruct population
        self.population = [
            Individual(
                gene=AttentionGene(gene_dict=ind['gene']),
                fitness=ind['fitness'],
                perplexity=ind['perplexity'],
                train_loss=ind.get('train_loss', float('inf')),
                generation=ind['generation']
            )
            for ind in checkpoint['population']
        ]
        
        print(f"📂 Loaded checkpoint from generation {generation}")
    
    def print_summary(self):
        """Print final results summary."""
        best = self.population[0]
        
        print(f"\n🥇 BEST ATTENTION MECHANISM FOUND:")
        print(f"   Gene: {best.gene}")
        print(f"   Fitness: {best.fitness:.4f}")
        print(f"   Perplexity: {best.perplexity:.2f}")
        print(f"   Found in Generation: {best.generation}")
        
        print(f"\n📈 Top 5 Individuals:")
        for i, ind in enumerate(self.population[:5], 1):
            print(f"   {i}. {ind.gene} | Perplexity: {ind.perplexity:.2f}")
    
    def plot_history(self, save_path: str = None):
        """
        Visualize evolution progress.
        
        Args:
            save_path: Optional path to save plot
        """
        if not self.history:
            print("⚠️  No history to plot")
            return
        
        generations = [h['generation'] for h in self.history]
        best_perplexity = [h['best_perplexity'] for h in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_perplexity, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Best Perplexity', fontsize=12)
        plt.title('Evolution Progress', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, output_dir: str = "./results"):
        """
        Save final results to disk.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save best individual
        best = self.population[0]
        results = {
            'best_gene': best.gene.to_dict(),
            'best_fitness': float(best.fitness),
            'best_perplexity': float(best.perplexity),
            'best_generation': best.generation,
            'history': self.history,
            'top_5': [
                {
                    'gene': ind.gene.to_dict(),
                    'perplexity': float(ind.perplexity),
                    'generation': ind.generation
                }
                for ind in self.population[:5]
            ]
        }
        
        results_path = output_path / 'final_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to {results_path}")
        
        # Save evolution plot
        plot_path = output_path / 'evolution_progress.png'
        self.plot_history(save_path=str(plot_path))
