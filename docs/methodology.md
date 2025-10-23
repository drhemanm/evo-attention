File 16: docs/methodology.md
Path: docs/methodology.md
markdown# Methodology

This document describes how EvoAttention's evolutionary algorithm works.

## Overview

EvoAttention uses a **genetic algorithm** to search the space of attention mechanisms. The core idea is simple: instead of hand-designing attention, we let evolution explore different designs and keep the best ones.

## Search Space

### Attention Gene Encoding

Each attention mechanism is encoded as a "gene" with 6 components:

| Component | Options | Description |
|-----------|---------|-------------|
| **Similarity** | dot, additive, multiplicative, cosine | How Q and K compute attention scores |
| **Normalization** | softmax, sparsemax, relu_norm, sigmoid | How scores are normalized to weights |
| **Gating** | none, input_gate, output_gate, highway | Optional gating mechanisms |
| **Temperature** | fixed, learned, adaptive | How to scale attention scores |
| **Use Bias** | true, false | Whether attention projections have bias |
| **Dropout** | 0.0-0.3 | Attention dropout rate |

This creates a search space of **4 × 4 × 4 × 3 × 2 × ∞ = 384+ possible mechanisms** (continuous for dropout).

### Why These Components?

These components represent fundamental design choices in attention:

1. **Similarity**: Different ways to measure query-key relevance
2. **Normalization**: Different ways to convert scores to probability-like weights
3. **Gating**: Optional mechanisms to control information flow
4. **Temperature**: Controls attention sharpness (how focused vs. diffuse)

## Evolutionary Algorithm

### Population Initialization

1. Create **12 random individuals** (attention mechanisms)
2. Each individual has a randomly initialized gene
3. All individuals start with `fitness = -∞` (unevaluated)

### Fitness Evaluation

For each individual:

1. **Create model**: Build a transformer with that attention mechanism
2. **Train**: Train for 5,000 steps on WikiText-2
3. **Evaluate**: Measure perplexity on validation set
4. **Assign fitness**: `fitness = -perplexity` (higher is better)

This is the most expensive step (~10 minutes per individual on free Colab).

### Selection

After evaluation, we rank all individuals by fitness:

- **Keep top 3** (elitism): Best individuals survive unchanged
- **Discard bottom 9**: Worst performers are removed

This ensures we never lose the best solutions found so far.

### Reproduction

To create the next generation:

1. **Start with elite 3** individuals
2. **Generate 9 offspring** through:

#### Crossover (70% probability)

Combine two parent genes:
```python
parent1 = top_performer_1
parent2 = top_performer_2

child.similarity = random.choice([parent1.similarity, parent2.similarity])
child.normalization = random.choice([parent1.normalization, parent2.normalization])
# ... for each component
```

This allows successful component combinations to be explored.

#### Mutation (30% rate per component)

Randomly change components:
```python
if random() < 0.3:
    child.similarity = random.choice(['dot', 'additive', 'multiplicative', 'cosine'])
```

This maintains diversity and enables exploration.

### Termination

Repeat for **10 generations** (configurable):
```
Gen 0: Random initialization → Evaluate 12 individuals
Gen 1: Keep top 3, evolve 9 → Evaluate 9 new individuals
Gen 2: Keep top 3, evolve 9 → Evaluate 9 new individuals
...
Gen 9: Keep top 3, evolve 9 → Evaluate 9 new individuals
```

**Total evaluations**: 12 + (9 × 9) = **93 trained models**

## Training Details

### Model Architecture

- **Layers**: 2 transformer blocks
- **Dimension**: 128
- **Heads**: 4
- **Parameters**: ~500K

Small by design to enable fast iteration.

### Training Procedure

Each individual is trained identically:

- **Optimizer**: AdamW (lr=3e-4)
- **Steps**: 5,000
- **Batch size**: 32
- **Gradient clipping**: Max norm 1.0
- **Dataset**: WikiText-2 (word-level)

### Why This Works

1. **Fast evaluation**: 5K steps takes ~10 minutes
2. **Sufficient signal**: 5K steps is enough to differentiate good/bad mechanisms
3. **Fair comparison**: All individuals trained identically

## Key Design Decisions

### Why 12 individuals?

- Small enough to run on free Colab (~2-3 hours total)
- Large enough to explore diverse mechanisms
- Divisible by 3 for clean elitism (top 25%)

### Why 10 generations?

- Empirically, convergence happens around generation 7-8
- More generations = diminishing returns
- 10 is a good balance of exploration vs. compute

### Why keep top 3?

- Ensures best solutions are never lost
- 25% elitism is standard in genetic algorithms
- Provides stable parents for reproduction

### Why 5,000 training steps?

- Enough to see performance differences
- Fast enough to iterate quickly
- Small models converge in 5-10K steps

## Limitations

### Search Space Coverage

With 93 evaluations across 384+ possible mechanisms, we only sample **~24%** of the discrete search space. Many potentially good mechanisms are never tried.

### Training Variance

Training the same mechanism multiple times can give different results (±1 perplexity). This means:

- Some "better" mechanisms might just be lucky
- True fitness requires multiple runs (not done here for compute reasons)

### No Multi-Objective Optimization

We only optimize perplexity. Real-world applications care about:

- Inference speed (FLOPs)
- Memory usage
- Training stability

These are not considered.

### Scale Unknown

Results are for 2-layer, 128d models. It's unclear if these mechanisms work at 100M+ parameters.

## Future Improvements

### Better Initialization

Instead of pure random, use informed priors:

- 50% proven components (dot+softmax baseline)
- 50% exploratory combinations

### Adaptive Mutation

Start with high mutation (exploration), decrease over time (exploitation):
```python
mutation_rate = 0.5 * (0.5 ** (generation / n_generations))
```

### Early Stopping

Stop training bad individuals early:
```python
if step > 1000 and loss > 2.0 * best_loss:
    break  # Don't waste compute
```

### Multi-Objective

Optimize for multiple criteria:

- Accuracy (perplexity)
- Efficiency (FLOPs)
- Memory footprint

Use Pareto front for selection.

## References

- **Genetic Algorithms**: Holland, J. H. (1992). Adaptation in natural and artificial systems
- **Neural Architecture Search**: Zoph & Le (2016). Neural Architecture Search with Reinforcement Learning
- **Attention Mechanisms**: Vaswani et al. (2017). Attention is All You Need
- **Sparsemax**: Martins & Astudillo (2016). From Softmax to Sparsemax

---

For implementation details, see the source code in `evo_attention/evolution.py`.
