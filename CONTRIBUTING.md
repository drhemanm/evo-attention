File 20: CONTRIBUTING.md
Path: CONTRIBUTING.md
markdown# Contributing to EvoAttention

Thank you for your interest in contributing! This project is open to contributions from everyone.

## How to Contribute

### Reporting Issues

Found a bug or have a feature request?

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear title
   - Detailed description
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - Your environment (OS, Python version, GPU)

### Submitting Code

1. **Fork the repository**
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Test your changes**: `pytest tests/`
5. **Commit**: Use clear commit messages
6. **Push**: `git push origin feature/your-feature-name`
7. **Create a Pull Request**

## Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/evo-attention.git
cd evo-attention

# Install in development mode
pip install -e .

# Install dev dependencies
pip install pytest black flake8

# Run tests
pytest tests/
```

## Code Style

We follow standard Python conventions:

- **PEP 8** for code style
- **Black** for formatting (line length: 100)
- **Type hints** where appropriate
- **Docstrings** for all public functions/classes
```bash
# Format code
black evo_attention/ experiments/ tests/

# Check style
flake8 evo_attention/ --max-line-length=100
```

## What We're Looking For

### High Priority

- [ ] **Scaling experiments**: Test on larger models
- [ ] **New datasets**: WikiText-103, Penn Treebank, Enwik8
- [ ] **Efficiency metrics**: Add FLOPs counting, memory profiling
- [ ] **Multi-objective optimization**: Optimize for speed + accuracy
- [ ] **Visualization**: Attention pattern visualization tools
- [ ] **Documentation**: Jupyter notebooks, tutorials

### Medium Priority

- [ ] **Better evolutionary algorithms**: CMA-ES, NSGA-II
- [ ] **Transfer learning**: Test across domains
- [ ] **Ablation studies**: Isolate component contributions
- [ ] **Statistical validation**: Multiple runs, significance tests
- [ ] **Comparison baselines**: Linear attention, Performer, etc.

### Nice to Have

- [ ] **Web UI**: Interactive visualization of evolution
- [ ] **Pre-trained models**: Share best discovered mechanisms
- [ ] **Docker support**: Containerized experiments
- [ ] **Cloud integration**: AWS/GCP training scripts
- [ ] **Distributed training**: Multi-GPU support

## Contribution Guidelines

### Code Quality

- ✅ All tests pass
- ✅ New features have tests
- ✅ Code is documented
- ✅ No unnecessary dependencies

### Documentation

- ✅ Update README.md if needed
- ✅ Add docstrings to new functions
- ✅ Update docs/ if methodology changes
- ✅ Include examples for new features

### Experiments

If adding new experiments:

- ✅ Include configuration files
- ✅ Document hyperparameters
- ✅ Report results honestly (including failures)
- ✅ Save checkpoints for reproducibility

## Areas for Contribution

### 1. New Search Spaces

Add new attention components:
```python
class MyCustomGene(AttentionGene):
    SIMILARITY_FUNCS = AttentionGene.SIMILARITY_FUNCS + ['my_similarity']
    
    def compute_my_similarity(self, q, k):
        # Your implementation
        pass
```

### 2. Better Evolutionary Algorithms

Improve the evolution strategy:
```python
class AdvancedEvolution(Evolution):
    def evolve(self):
        # Your improved evolution strategy
        pass
```

### 3. New Tasks

Apply to different domains:
```python
# experiments/vision_evolution.py
# Evolve attention for Vision Transformers
```

### 4. Analysis Tools

Help understand results:
```python
# evo_attention/visualization.py
def plot_attention_patterns(model, data):
    # Visualize what evolved attention learned
    pass
```

### 5. Optimization

Make it faster:
```python
# evo_attention/efficient_attention.py
# Add Flash Attention integration
# Add CUDA kernels for custom operations
```

## Testing

All contributions should include tests:
```python
# tests/test_your_feature.py
def test_your_new_feature():
    # Test your code
    assert expected == actual
```

Run tests before submitting:
```bash
pytest tests/ -v
```

## Documentation

Update documentation for:

- New features → README.md
- Methodology changes → docs/methodology.md
- New findings → docs/findings.md
- Known issues → docs/limitations.md

## Review Process

1. **Automated checks** run on your PR
2. **Maintainer review** (usually within 1 week)
3. **Discussion** if changes needed
4. **Merge** once approved

## Communication

- **GitHub Issues**: Bug reports, feature requests
- **Pull Requests**: Code contributions
- **Discussions**: General questions, ideas

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be:

- Listed in README.md acknowledgments
- Credited in relevant documentation
- Co-authors on papers (if applicable)

## Questions?

Not sure where to start? Open an issue with the `question` label!

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to learn and build cool stuff.

---

**Thank you for contributing to EvoAttention!** 🧬
