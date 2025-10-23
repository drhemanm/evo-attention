"""
Search space definition for attention mechanisms.
Genes encode different attention mechanism variants.
"""

import random
from typing import Dict, Any


class AttentionGene:
    """
    Represents a gene encoding an attention mechanism.
    
    A gene consists of 4 components:
    - similarity: How to compute attention scores
    - normalization: How to normalize attention weights
    - gating: Optional gating mechanism
    - temperature_mode: How to scale attention scores
    """
    
    SIMILARITY_FUNCS = ['dot', 'additive', 'multiplicative', 'cosine']
    NORMALIZATIONS = ['softmax', 'sigmoid', 'relu_norm', 'sparsemax']
    GATING = ['none', 'input_gate', 'output_gate', 'highway']
    TEMPERATURE_MODES = ['fixed', 'learned', 'adaptive']
    
    def __init__(self, gene_dict: Dict[str, Any] = None):
        """
        Initialize a gene either randomly or from a dictionary.
        
        Args:
            gene_dict: Optional dictionary with gene parameters
        """
        if gene_dict is None:
            # Random initialization
            self.similarity = random.choice(self.SIMILARITY_FUNCS)
            self.normalization = random.choice(self.NORMALIZATIONS)
            self.gating = random.choice(self.GATING)
            self.temperature_mode = random.choice(self.TEMPERATURE_MODES)
            self.use_bias = random.choice([True, False])
            self.attention_dropout = random.uniform(0.0, 0.3)
        else:
            # Initialize from dictionary
            self.__dict__.update(gene_dict)
    
    def mutate(self, mutation_rate: float = 0.3) -> 'AttentionGene':
        """
        Create a mutated copy of this gene.
        
        Args:
            mutation_rate: Probability of mutating each component
            
        Returns:
            New mutated gene
        """
        new_gene = AttentionGene(gene_dict=self.__dict__.copy())
        
        if random.random() < mutation_rate:
            new_gene.similarity = random.choice(self.SIMILARITY_FUNCS)
        if random.random() < mutation_rate:
            new_gene.normalization = random.choice(self.NORMALIZATIONS)
        if random.random() < mutation_rate:
            new_gene.gating = random.choice(self.GATING)
        if random.random() < mutation_rate:
            new_gene.temperature_mode = random.choice(self.TEMPERATURE_MODES)
        if random.random() < mutation_rate:
            new_gene.use_bias = not new_gene.use_bias
        if random.random() < mutation_rate:
            new_gene.attention_dropout = random.uniform(0.0, 0.3)
        
        return new_gene
    
    def crossover(self, other: 'AttentionGene') -> 'AttentionGene':
        """
        Create offspring via crossover with another gene.
        
        Args:
            other: Another gene to crossover with
            
        Returns:
            New child gene
        """
        child_dict = {}
        for key in self.__dict__:
            child_dict[key] = random.choice([
                self.__dict__[key], 
                other.__dict__[key]
            ])
        return AttentionGene(gene_dict=child_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert gene to dictionary."""
        return self.__dict__.copy()
    
    def __str__(self) -> str:
        """String representation of gene."""
        return (f"Attn({self.similarity}|{self.normalization}|"
                f"{self.gating}|{self.temperature_mode})")
    
    def __repr__(self) -> str:
        return self.__str__()
