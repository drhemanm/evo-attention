"""
Training and evaluation utilities for evolved attention models.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

from .model import TinyTransformer
from .config import Config
from .search_space import AttentionGene


def train_and_evaluate(
    model: TinyTransformer,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    config: Config,
    gene_id: int = 0
) -> tuple[float, float, float]:
    """
    Train a model and evaluate its fitness.
    
    Args:
        model: Transformer model to train
        train_loader: Training data loader
        eval_loader: Evaluation data loader
        config: Training configuration
        gene_id: ID for progress bar display
        
    Returns:
        Tuple of (fitness, perplexity, train_loss)
        - fitness: Negative perplexity (higher is better)
        - perplexity: Evaluation perplexity
        - train_loss: Average training loss
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    model.train()
    train_losses = []
    step = 0
    
    pbar = tqdm(total=config.train_steps, desc=f"Gene {gene_id}", leave=False)
    
    while step < config.train_steps:
        for x, y in train_loader:
            if step >= config.train_steps:
                break
            
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                y.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            step += 1
            
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    pbar.close()
    
    # Evaluation
    model.eval()
    eval_losses = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(eval_loader):
            if i >= config.eval_steps:
                break
                
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                y.view(-1)
            )
            eval_losses.append(loss.item())
    
    # Calculate metrics
    eval_loss = np.mean(eval_losses)
    perplexity = np.exp(eval_loss)
    avg_train_loss = np.mean(train_losses[-100:])  # Last 100 steps
    
    # Fitness is negative perplexity (higher is better for evolution)
    fitness = -perplexity
    
    return fitness, perplexity, avg_train_loss


def create_baseline_model(config: Config) -> TinyTransformer:
    """
    Create a baseline transformer with standard attention.
    
    Args:
        config: Model configuration
        
    Returns:
        Transformer model with vanilla attention
    """
    baseline_gene = AttentionGene(gene_dict={
        'similarity': 'dot',
        'normalization': 'softmax',
        'gating': 'none',
        'temperature_mode': 'fixed',
        'use_bias': True,
        'attention_dropout': 0.1
    })
    
    return TinyTransformer(config, baseline_gene)
