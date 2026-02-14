# Assignment 1 – JAX vs PyTorch MLP Implementation (MNIST)

This project implements and compares a two-hidden-layer MLP on the MNIST dataset using:

- **Pure JAX (manual implementation from scratch with `jax.grad` and `jax.jit`)**
- **PyTorch (`nn.Module`, `CrossEntropyLoss`, Adam optimizer)**

The goal of this assignment is to compare:

- First-epoch (JIT compilation) overhead  
- Steady-state training performance  
- Final test accuracy  
- Effect of batch size  
- Reproducibility across seeds  

All experiments log metrics, save checkpoints, and generate visualization plots.

---

## Quickstart

### Option 1 – Automatic Setup (Recommended)

Run:

```bash
bash init.sh true
