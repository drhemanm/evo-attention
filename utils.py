"""
Utility functions for data loading, preprocessing, and helpers.
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class WikiTextDataset(Dataset):
    """
    WikiText-2 dataset for language modeling.
    """
    
    # Class-level vocabulary (shared across train/val/test)
    vocab = None
    inv_vocab = None
    
    def __init__(self, split: str = 'train', max_seq_len: int = 128, vocab_size: int = 10000):
        """
        Initialize WikiText-2 dataset.
        
        Args:
            split: Dataset split ('train', 'validation', or 'test')
            max_seq_len: Maximum sequence length
            vocab_size: Vocabulary size
        """
        print(f"📚 Loading WikiText-2 {split} split...")
        
        # Load dataset
        try:
            from datasets import load_dataset
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        except ImportError:
            print("⚠️  Installing datasets library...")
            import subprocess
            subprocess.check_call(['pip', 'install', '-q', 'datasets'])
            from datasets import load_dataset
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        
        # Build vocabulary (only once, from training data)
        if WikiTextDataset.vocab is None and split == 'train':
            print("🔤 Building vocabulary...")
            self._build_vocabulary(dataset, vocab_size)
            print(f"✅ Vocabulary built: {len(WikiTextDataset.vocab)} tokens")
        
        # Tokenize and create sequences
        self.sequences = []
        for item in dataset:
            text = item['text'].strip()
            if len(text) == 0:
                continue
            
            # Tokenize (simple whitespace tokenization)
            tokens = text.split()
            token_ids = [
                WikiTextDataset.vocab.get(t, WikiTextDataset.vocab['<UNK>']) 
                for t in tokens
            ]
            
            # Split into fixed-length sequences
            for i in range(0, len(token_ids) - max_seq_len, max_seq_len // 2):
                seq = token_ids[i:i + max_seq_len]
                if len(seq) == max_seq_len:
                    self.sequences.append(torch.tensor(seq, dtype=torch.long))
        
        print(f"✅ Created {len(self.sequences)} sequences")
    
    @staticmethod
    def _build_vocabulary(dataset, vocab_size: int):
        """Build vocabulary from dataset."""
        from collections import Counter
        
        # Collect all text
        all_text = ' '.join([
            item['text'] 
            for item in dataset 
            if len(item['text'].strip()) > 0
        ])
        
        # Tokenize and count
        tokens = all_text.split()
        token_counts = Counter(tokens)
        
        # Create vocabulary with special tokens
        WikiTextDataset.vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<EOS>': 2,
        }
        
        # Add most common tokens
        most_common = token_counts.most_common(vocab_size - 3)
        for idx, (token, _) in enumerate(most_common, start=3):
            WikiTextDataset.vocab[token] = idx
        
        # Create inverse mapping
        WikiTextDataset.inv_vocab = {v: k for k, v in WikiTextDataset.vocab.items()}
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence and its target.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (input_sequence, target_sequence)
        """
        x = self.sequences[idx]
        # Target is input shifted by one (predict next token)
        y = torch.cat([x[1:], torch.tensor([WikiTextDataset.vocab['<PAD>']])])
        return x, y


class SyntheticDataset(Dataset):
    """
    Simple synthetic dataset for testing.
    Generates random token sequences.
    """
    
    def __init__(self, vocab_size: int, seq_len: int, n_samples: int):
        """
        Initialize synthetic dataset.
        
        Args:
            vocab_size: Vocabulary size
            seq_len: Sequence length
            n_samples: Number of samples
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.data = torch.randint(1, vocab_size, (n_samples, seq_len))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence and its target."""
        x = self.data[idx]
        y = torch.cat([x[1:], torch.tensor([0])])
        return x, y


def get_dataloaders(
    vocab_size: int,
    max_seq_len: int,
    batch_size: int,
    eval_batch_size: int,
    use_wikitext: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and eval data loaders.
    
    Args:
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        use_wikitext: If True, use WikiText-2; else use synthetic data
        
    Returns:
        Tuple of (train_loader, eval_loader)
    """
    if use_wikitext:
        try:
            train_dataset = WikiTextDataset('train', max_seq_len, vocab_size)
            eval_dataset = WikiTextDataset('validation', max_seq_len, vocab_size)
            print("🎯 Using WikiText-2 dataset")
        except Exception as e:
            print(f"⚠️  Failed to load WikiText: {e}")
            print("📦 Falling back to synthetic data")
            train_dataset = SyntheticDataset(vocab_size, max_seq_len, 10000)
            eval_dataset = SyntheticDataset(vocab_size, max_seq_len, 1000)
    else:
        train_dataset = SyntheticDataset(vocab_size, max_seq_len, 10000)
        eval_dataset = SyntheticDataset(vocab_size, max_seq_len, 1000)
        print("🎲 Using synthetic data")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, eval_loader


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total number of trainable parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> str:
    """
    Get the best available device.
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
