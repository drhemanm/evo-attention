"""
Script to create the quickstart notebook.
Run this once: python create_notebook.py
Then delete this file.
"""

import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# EvoAttention Quickstart\n",
                "\n",
                "This notebook demonstrates how to use EvoAttention to discover attention mechanisms.\n",
                "\n",
                "**Time required:** ~3 hours on free Colab GPU\n",
                "\n",
                "**What you'll learn:**\n",
                "- How to configure evolutionary search\n",
                "- How to run evolution\n",
                "- How to analyze results"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Setup\n",
                "\n",
                "First, install the package (if running on Colab):"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Uncomment if running on Colab\n",
                "# !git clone https://github.com/yourusername/evo-attention.git\n",
                "# %cd evo-attention\n",
                "# !pip install -e ."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Import Dependencies"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "from evo_attention import Config, Evolution\n",
                "from evo_attention.utils import set_seed, get_dataloaders, get_device\n",
                "\n",
                "print(f\"PyTorch version: {torch.__version__}\")\n",
                "print(f\"Device: {get_device()}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Configuration\n",
                "\n",
                "Configure the evolutionary search. For a quick test, use fewer generations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "config = Config(\n",
                "    # Model architecture\n",
                "    vocab_size=10000,\n",
                "    d_model=128,\n",
                "    n_heads=4,\n",
                "    n_layers=2,\n",
                "    \n",
                "    # Training\n",
                "    batch_size=32,\n",
                "    train_steps=5000,  # Reduce to 1000 for quick test\n",
                "    \n",
                "    # Evolution\n",
                "    population_size=12,  # Reduce to 6 for quick test\n",
                "    n_generations=10,    # Reduce to 3 for quick test\n",
                "    elite_size=3,\n",
                "    mutation_rate=0.3,\n",
                "    \n",
                "    # System\n",
                "    device=get_device(),\n",
                "    checkpoint_dir=\"./results/checkpoints\",\n",
                "    seed=42\n",
                ")\n",
                "\n",
                "set_seed(config.seed)\n",
                "print(\"✅ Configuration complete\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Load Data\n",
                "\n",
                "Load WikiText-2 dataset (or use synthetic data for quick testing)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Set use_wikitext=False for quick synthetic data test\n",
                "train_loader, eval_loader = get_dataloaders(\n",
                "    vocab_size=config.vocab_size,\n",
                "    max_seq_len=config.max_seq_len,\n",
                "    batch_size=config.batch_size,\n",
                "    eval_batch_size=config.eval_batch_size,\n",
                "    use_wikitext=True  # Set to False for quick test\n",
                ")\n",
                "\n",
                "print(\"✅ Data loaded\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Run Evolution\n",
                "\n",
                "This will take ~3 hours on free Colab GPU.\n",
                "\n",
                "**Tip:** The code auto-saves checkpoints every generation, so you can stop and resume anytime."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create evolution instance\n",
                "evo = Evolution(config, train_loader, eval_loader, use_wikitext=True)\n",
                "\n",
                "# Run evolution\n",
                "evo.run()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Analyze Results\n",
                "\n",
                "Let's look at what evolution discovered!"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Print summary\n",
                "evo.print_summary()\n",
                "\n",
                "# Get best individual\n",
                "best = evo.population[0]\n",
                "print(f\"\\n🏆 Best Attention Mechanism:\")\n",
                "print(f\"   {best.gene}\")\n",
                "print(f\"   Perplexity: {best.perplexity:.2f}\")\n",
                "print(f\"   Generation: {best.generation}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Visualize Evolution Progress"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot evolution curve\n",
                "evo.plot_history()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Compare to Baseline"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from evo_attention import AttentionGene, TinyTransformer\n",
                "from evo_attention.training import train_and_evaluate\n",
                "\n",
                "# Create baseline (vanilla transformer)\n",
                "baseline_gene = AttentionGene(gene_dict={\n",
                "    'similarity': 'dot',\n",
                "    'normalization': 'softmax',\n",
                "    'gating': 'none',\n",
                "    'temperature_mode': 'fixed',\n",
                "    'use_bias': True,\n",
                "    'attention_dropout': 0.1\n",
                "})\n",
                "\n",
                "print(\"Training baseline transformer...\")\n",
                "baseline_model = TinyTransformer(config, baseline_gene)\n",
                "_, baseline_ppl, _ = train_and_evaluate(\n",
                "    baseline_model,\n",
                "    train_loader,\n",
                "    eval_loader,\n",
                "    config,\n",
                "    gene_id=0\n",
                ")\n",
                "\n",
                "print(f\"\\n📊 Comparison:\")\n",
                "print(f\"   Baseline: {baseline_ppl:.2f} perplexity\")\n",
                "print(f\"   Evolved:  {best.perplexity:.2f} perplexity\")\n",
                "print(f\"   Improvement: {((baseline_ppl - best.perplexity) / baseline_ppl * 100):.1f}%\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Save Results"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Save final results\n",
                "evo.save_results(output_dir=\"./results\")\n",
                "\n",
                "print(\"✅ Results saved to ./results/\")\n",
                "print(\"   - final_results.json\")\n",
                "print(\"   - evolution_progress.png\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Use the Best Mechanism\n",
                "\n",
                "Now you can use the discovered attention mechanism in your own models!"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create a model with the best discovered mechanism\n",
                "best_model = TinyTransformer(config, best.gene)\n",
                "\n",
                "print(f\"Created model with: {best.gene}\")\n",
                "print(f\"Parameters: {sum(p.numel() for p in best_model.parameters()):,}\")\n",
                "\n",
                "# You can now train this model further or use it for your task\n",
                "# model.train()\n",
                "# ..."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Next Steps\n",
                "\n",
                "1. **Validate**: Run multiple times with different seeds\n",
                "2. **Scale**: Test on larger models and datasets\n",
                "3. **Analyze**: Study why certain mechanisms work\n",
                "4. **Apply**: Use discovered mechanisms in your projects\n",
                "\n",
                "Check out the other notebooks for more advanced usage!"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Create notebooks directory
Path("notebooks").mkdir(exist_ok=True)

# Write notebook
with open("notebooks/01_quickstart.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("✅ Created notebooks/01_quickstart.ipynb")
print("You can now delete this script: rm create_notebook.py")
