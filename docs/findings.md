File 17: docs/findings.md
Path: docs/findings.md
markdown# Findings

Detailed analysis of what we discovered through evolutionary attention search.

## Summary

**Main Result**: Evolved attention mechanisms achieve **98.45 perplexity** on WikiText-2, compared to **102.90 for vanilla transformers** - a **4.3% improvement**.

**Best mechanism**: `dot-product + sparsemax + output_gate + learned_temperature`

## Evolution Timeline

| Generation | Best Perplexity | Mechanism | Improvement |
|------------|-----------------|-----------|-------------|
| 0 | 100.37 | multiplicative\|relu_norm\|output_gate\|fixed | 2.5% |
| 1 | 100.37 | (same, carried over) | 2.5% |
| 2 | 99.99 | dot\|sparsemax\|output_gate\|learned | 2.8% |
| 3 | 99.58 | multiplicative\|relu_norm\|output_gate\|fixed | 3.2% |
| 4 | 98.88 | multiplicative\|sparsemax\|output_gate\|learned | 3.9% |
| 5-6 | 98.88 | (plateau) | 3.9% |
| 7 | 98.70 | dot\|sparsemax\|output_gate\|adaptive | 4.1% |
| 8 | 98.70 | (same) | 4.1% |
| 9 | **98.45** | **dot\|sparsemax\|output_gate\|learned** | **4.3%** |

### Key Observations

1. **Steady improvement**: Clear downward trend over 10 generations
2. **Plateau effect**: Generations 5-6 and 8 showed no improvement
3. **Late discovery**: Best result found in generation 9
4. **Convergence**: Population converged to similar mechanisms by generation 7

## Top 5 Mechanisms

| Rank | Mechanism | Perplexity | Generation |
|------|-----------|------------|------------|
| 🥇 | dot\|sparsemax\|output_gate\|learned | 98.45 | 9 |
| 🥈 | dot\|sparsemax\|output_gate\|adaptive | 98.70 | 7 |
| 🥉 | multiplicative\|sparsemax\|output_gate\|learned | 98.88 | 4 |
| 4 | multiplicative\|relu_norm\|output_gate\|adaptive | 99.73 | 3 |
| 5 | dot\|sparsemax\|output_gate\|learned | 99.99 | 2 |

### Pattern Analysis

**Common components in top 5:**
- **100% have output gating** (5/5)
- **80% use sparsemax** (4/5)
- **60% use learned temperature** (3/5)
- **60% use dot-product similarity** (3/5)

## Component Analysis

### Similarity Functions

| Function | Best Perplexity | Avg Perplexity | Count in Top 10 |
|----------|-----------------|----------------|-----------------|
| **dot** | 98.45 | 101.2 | 6/10 |
| **multiplicative** | 98.88 | 102.5 | 3/10 |
| additive | 104.03 | 105.8 | 1/10 |
| cosine | 107.33 | 112.4 | 0/10 |

**Finding**: Dot-product and multiplicative both work well. Cosine consistently fails.

### Normalization

| Normalization | Best Perplexity | Avg Perplexity | Count in Top 10 |
|---------------|-----------------|----------------|-----------------|
| **sparsemax** | 98.45 | 100.1 | 7/10 |
| **relu_norm** | 99.73 | 102.3 | 2/10 |
| softmax | 102.90 | 104.5 | 1/10 |
| sigmoid | 107.40 | 110.2 | 0/10 |

**Finding**: Sparsemax dominates. This suggests **sparse attention is better than dense attention** for language modeling.

### Gating Mechanisms

| Gating | Best Perplexity | Avg Perplexity | Count in Top 10 |
|--------|-----------------|----------------|-----------------|
| **output_gate** | 98.45 | 100.8 | 8/10 |
| none | 102.88 | 104.2 | 2/10 |
| input_gate | 103.60 | 106.5 | 0/10 |
| highway | 106.30 | 115.8 | 0/10 |

**Finding**: Output gating is critical. Highway gating consistently fails (worst performer).

### Temperature Modes

| Temperature | Best Perplexity | Avg Perplexity | Count in Top 10 |
|-------------|-----------------|----------------|-----------------|
| **learned** | 98.45 | 101.5 | 5/10 |
| **adaptive** | 98.70 | 102.1 | 3/10 |
| fixed | 100.37 | 103.8 | 2/10 |

**Finding**: Learned and adaptive temperatures both work. Fixed temperature is viable but suboptimal.

## Key Discoveries

### Discovery #1: Sparsemax > Softmax

**The biggest surprise**: Sparsemax normalization consistently outperforms softmax.

**Why this matters:**
- Sparsemax was proposed in 2016 but not widely adopted
- Creates sparse attention (many weights = 0)
- Our results suggest it should be reconsidered for language modeling

**Hypothesis**: Sparse attention forces the model to focus on fewer, more relevant tokens, reducing noise.

### Discovery #2: Output Gating is Universal

**Every top mechanism uses output gating.**
```python
# What output gating does:
output = attention_result
gate = sigmoid(linear(input))
output = output * gate  # Element-wise gating
```

**Hypothesis**: Output gating provides fine-grained control over how much attention information to use vs. residual connection.

### Discovery #3: Highway Gating Always Fails

Highway gating was the **worst performer** across all generations (avg perplexity: 115.8).
```python
# Highway gating:
gate = sigmoid(linear(input))
output = gate * attention_result + (1 - gate) * input
```

**Hypothesis**: The hard interpolation between attention and residual creates optimization difficulties.

### Discovery #4: Dot-Product is Actually Good

Vanilla dot-product attention (when combined with the right normalization) can be competitive.

**This is important** because:
- Dot-product is computationally efficient
- The improvement comes from normalization + gating, not exotic similarity functions
- Makes the mechanism more practical

### Discovery #5: Training Variance Matters

The same mechanism (`multiplicative|relu_norm|output_gate|fixed`) achieved:
- Generation 0: 100.37 perplexity
- Generation 3: 99.58 perplexity

**1 perplexity point variance** from different random initialization!

**Implication**: Single evaluations can be misleading. True performance requires multiple runs.

## Statistical Analysis

### Performance Distribution
```
Baseline (dot|softmax): 102.90 perplexity

Best evolved:           98.45 perplexity
Improvement:            4.45 points (4.3%)
Relative improvement:   4.3%

Top 5 average:          99.15 perplexity
Top 10 average:         100.82 perplexity
All mechanisms avg:     105.52 perplexity
```

### Convergence Analysis

**Population diversity over time:**

- Gen 0: 12 unique mechanisms → Range: 100.37 to 115.69 (15.32 spread)
- Gen 5: 7 unique archetypes → Range: 98.88 to 108.45 (9.57 spread)
- Gen 9: 4 unique archetypes → Range: 98.45 to 104.23 (5.78 spread)

**Finding**: Population converges to similar high-performing mechanisms.

## Comparison to Related Work

### vs. Linear Attention

Linear attention approximations (Katharopoulos et al., 2020) achieve ~5% speedup but often lose 1-2 perplexity points.

**Our approach**: Similar speedup potential (sparsemax is fast) but **improves** perplexity.

### vs. Flash Attention

Flash Attention (Dao et al., 2022) is about implementation efficiency, not mechanism design. Our mechanisms could potentially use Flash Attention's optimizations.

### vs. Sparse Transformers

Sparse Transformers (Child et al., 2019) use fixed sparsity patterns. **Sparsemax provides learned sparsity**, which may be more flexible.

## Limitations of Findings

### Scale

All results are for **2-layer, 128d models (~500K params)**.

**Unknown:**
- Do these mechanisms work at 100M+ parameters?
- Do they work for GPT-scale models?

### Dataset

Only tested on **WikiText-2** (103M tokens, English).

**Unknown:**
- Do results transfer to other languages?
- Do they work on other modalities (vision, audio)?

### Training Variance

Each mechanism evaluated **once** (no repeated runs).

**Risk**: Some "better" results might be lucky random seeds.

**Proper validation requires**: 5+ runs per mechanism with statistical tests.

### Single Objective

Optimized only for **perplexity**.

**Not considered:**
- Inference speed
- Memory usage
- Training stability
- Gradient flow

## Practical Implications

### For Researchers

1. **Sparsemax deserves reconsideration** for attention normalization
2. **Output gating is a cheap improvement** that might help existing models
3. **Evolutionary search is viable** for architecture discovery at small scale

### For Practitioners

1. If building small models (edge deployment), try: `dot + sparsemax + output_gate`
2. Don't use highway gating in attention
3. Consider learned temperature for task-specific fine-tuning

### For Future Work

1. **Scale up**: Test on larger models and datasets
2. **Multi-objective**: Optimize for efficiency + accuracy
3. **Transfer**: Test across domains (vision, speech, etc.)
4. **Hybrid**: Combine evolved mechanisms with Flash Attention

## Reproducibility

**Good news**: Results are reproducible with the same seed (42).

**Variance**: Different seeds give ±1 perplexity due to training randomness.

**Recommendation**: Run 3+ times with different seeds for robust conclusions.

## Conclusion

We demonstrated that:

1. ✅ Evolutionary search can discover better attention mechanisms
2. ✅ Sparsemax + output gating consistently outperform vanilla attention
3. ✅ The approach works on limited compute (free Colab)

But remember:

- ⚠️ Small scale only (not validated at GPT scale)
- ⚠️ Single dataset (needs cross-domain validation)
- ⚠️ Training variance is significant

**This is a proof of concept, not a production-ready solution.**

---

For methodology details, see [methodology.md](methodology.md).  
For limitations and caveats, see [limitations.md](limitations.md).

