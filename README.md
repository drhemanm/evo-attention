# 🧬 EvoAttention: Evolutionary Discovery of Attention Mechanisms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> *What if we let evolution design attention mechanisms instead of hand-crafting them?*

EvoAttention is a framework for using evolutionary algorithms to discover novel attention mechanism architectures. Instead of assuming scaled dot-product attention is optimal, we search through the space of possible attention designs.

## 🎯 Key Results

On WikiText-2 language modeling with 2-layer transformers:

| Mechanism | Perplexity | Improvement |
|-----------|------------|-------------|
| **Vanilla Transformer** (baseline) | 102.90 | - |
| **Evolved Attention** | **98.45** | **4.3%** |

**Discovered mechanism:** `dot-product similarity + sparsemax normalization + learned temperature + output gating`

## 🚀 Quick Start
```bash

File 15: README.md
Path: README.md
markdown# 🧬 EvoAttention: Evolutionary Discovery of Attention Mechanisms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> *What if we let evolution design attention mechanisms instead of hand-crafting them?*

EvoAttention is a framework for using evolutionary algorithms to discover novel attention mechanism architectures. Instead of assuming scaled dot-product attention is optimal, we search through the space of possible attention designs.

## 🎯 Key Results

On WikiText-2 language modeling with 2-layer transformers:

| Mechanism | Perplexity | Improvement |
|-----------|------------|-------------|
| **Vanilla Transformer** (baseline) | 102.90 | - |
| **Evolved Attention** | **98.45** | **4.3%** |

**Discovered mechanism:** `dot-product similarity + sparsemax normalization + learned temperature + output gating`

## 🚀 Quick Start
```bash
# Install
pip install -e .

# Or install dependencies
pip install -r requirements.txt
```

### Run Evolution (3 hours on free Colab)
```python
from evo_attention import Config, Evolution
from evo_attention.utils import get_dataloaders, set_seed, get_device

# Setup
config = Config(
    population_size=12,
    n_generations=10,
    train_steps=5000,
    device=get_device()
)

set_seed(config.seed)

# Load data
train_loader, eval_loader = get_dataloaders(
    vocab_size=config.vocab_size,
    max_seq_len=config.max_seq_len,
    batch_size=config.batch_size,
    eval_batch_size=config.eval_batch_size,
    use_wikitext=True
)

# Run evolution
evo = Evolution(config, train_loader, eval_loader)
evo.run()

# Get best mechanism
best = evo.population[0]
print(f"Best: {best.gene} | Perplexity: {best.perplexity:.2f}")
```

### Command Line
```bash
# Run full evolution
python experiments/wikitext2_evolution.py

# Train baseline for comparison
python experiments/wikitext2_baseline.py
```

## 🧬 How It Works

### 1. Define Search Space

We encode attention mechanisms as genes with 4 components:
```python
gene = AttentionGene(
    similarity='dot',           # dot, multiplicative, additive, cosine
    normalization='sparsemax',  # softmax, sparsemax, relu_norm, sigmoid
    gating='output_gate',       # none, input_gate, output_gate, highway
    temperature='learned'       # fixed, learned, adaptive
)
```

### 2. Evolution Loop
```
Initialize random population (12 individuals)
For each generation:
    1. Train each attention mechanism (5K steps)
    2. Evaluate on validation set
    3. Keep top 3 (elitism)
    4. Generate 9 offspring via crossover + mutation
    5. Repeat
```

### 3. Discover Novel Mechanisms

Evolution found that **sparsemax normalization** + **learned temperature** consistently outperforms vanilla softmax attention.

## 📊 Results

### Evolution Curve

![Evolution Progress](results/evolution_progress.png)

- **Steady improvement** over 10 generations
- **Final best:** 98.45 perplexity (4.3% better than baseline)
- **Convergence** around generation 7-8

### Top Mechanisms Discovered

| Rank | Mechanism | Perplexity | Gen Found |
|------|-----------|------------|-----------|
| 🥇 | dot\|sparsemax\|output_gate\|learned | 98.45 | 9 |
| 🥈 | dot\|sparsemax\|output_gate\|adaptive | 98.70 | 7 |
| 🥉 | multiplicative\|sparsemax\|output_gate\|learned | 98.88 | 4 |

### Key Insights

**What works:**
- ✅ Sparsemax normalization (sparser attention)
- ✅ Learned/adaptive temperature (flexibility)
- ✅ Output gating (all top mechanisms have it)
- ✅ Dot-product OR multiplicative similarity

**What doesn't:**
- ❌ Highway gating (consistently worst)
- ❌ Cosine similarity (unless with relu_norm)
- ❌ Sigmoid normalization

## 🎓 Use Cases

### 1. Architecture Search
Search for attention mechanisms optimized for your specific task/dataset.

### 2. Efficiency Optimization
Find architectures that work well at smaller scales (edge deployment).

### 3. Research & Education
Learn about attention mechanisms by exploring the design space systematically.

## 📚 Documentation

- **[Methodology](docs/methodology.md)** - How the evolutionary algorithm works
- **[Findings](docs/findings.md)** - Detailed analysis of discovered mechanisms
- **[Limitations](docs/limitations.md)** - What this doesn't do (yet)

## 🛠️ Advanced Usage

### Custom Search Space
```python
from evo_attention import AttentionGene

# Define your own search space
class CustomGene(AttentionGene):
    SIMILARITY_FUNCS = ['my_custom_similarity', 'dot', 'additive']
    NORMALIZATIONS = ['my_custom_norm', 'softmax']
    
# Use in evolution
evo = Evolution(config, train_loader, eval_loader)
# Modify population initialization to use CustomGene
```

### Different Tasks
```python
# Use on your own dataset
from torch.utils.data import DataLoader

train_loader = DataLoader(your_dataset, batch_size=32)
eval_loader = DataLoader(your_eval_dataset, batch_size=64)

evo = Evolution(config, train_loader, eval_loader)
evo.run()
```

## 🧪 Experiments

Reproduce our results:
```bash
# WikiText-2 baseline
python experiments/wikitext2_baseline.py

# Run evolution
python experiments/wikitext2_evolution.py

# Results will be saved to ./results/
```

## ⚠️ Limitations

**Important:** This is research-grade code for exploring attention mechanisms, not production-ready. Current limitations:

- ✅ **Tested:** 2-layer models, WikiText-2, small scale
- ⚠️ **Not tested:** Large models (100M+ params), other datasets, production deployment
- ⚠️ **Training variance:** Results vary ±1 perplexity across runs
- ⚠️ **Compute:** Requires ~3 hours on free Colab GPU for full evolution

We're honest about what this does and doesn't do. See [Limitations](docs/limitations.md) for details.

## 🤝 Contributing

We'd love your help making this better!

**Areas we're exploring:**
- [ ] Scaling to larger models
- [ ] Multi-objective optimization (accuracy + efficiency)
- [ ] Transfer learning across tasks
- [ ] Better evolutionary algorithms (CMA-ES, NSGA-II)
- [ ] Attention pattern visualization

Open an issue or submit a PR!

## 📖 Citation

If you use this work, please cite:
```bibtex
@misc{evoattention2025,
  title={EvoAttention: Evolutionary Discovery of Attention Mechanisms},
  author={Heman Mohabeer},
  year={2025},
  url={https://github.com/yourusername/evo-attention}
}
```

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built on PyTorch
- Inspired by Neural Architecture Search literature
- WikiText-2 dataset from Salesforce Research
- Tested on Google Colab free tier

## 📬 Contact

- **Issues:** [GitHub Issues](https://github.com/yourusername/evo-attention/issues)
- **Email:** your.email@example.com
- **Twitter:** [@yourhandle](https://twitter.com/yourhandle)

---

**Built with 🧬 by an independent ML researcher. No institutional funding. Just curiosity and free Colab.**
