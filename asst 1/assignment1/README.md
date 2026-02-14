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
```

This will:

- Create a virtual environment (dl_env)
- Install dependencies
- Create required folders
- Verify JAX backend
- Run both:
  - jax_train.py
  - pytorch_train.py
- Generate metrics and visualization plots
If dependencies are already installed:
```bash
bash init.sh true
```

---

### Option 2 - Manual Setup

1. Create Virtual Environment
```bash
    python3 -m venv dl_env
    source dl_env/bin/activate
    python -m pip install --upgrade pip
```
2. Install Dependencies
```bash
    pip install -r requirements.txt
```
3. PyTorch for CUDA 12
```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

4.Run training Scripts
```bash
    python -m src.jax_train
    python -m src.pytorch_train
    python -m src.viz_script
```

---
### Configuration:
Edit config.yaml to control:
- seeds: random seeds

- batch_size: list of batch sizes

- train.epochs: number of epochs

- train.lr: learning rate

- train.checkpoint_path: checkpoint directory

- train.logs_path: log directory

- train.metrics_out_dir: metrics directory

---

### Output:
Running the full pipeline produces:
- Logs
    - logs/jax_logs
    - logs/torch_logs
- checkpoint
    - half way model weight
    - final model weights
- Metrics
    - Assignment1_JAX_metrics.csv
    - Assignment1_JAX_metrics_epoch.csv
    - Assignment1_TORCH_metrics.csv
    - Assignment1_TORCH_metrics_epoch.csv
- Summary File
    - results/summary.csv
    - Columns Include
      - framework
      - seed
      - batch_size
      - first_epoch_time
      - steady_epoch_time
      - final_test_acc
      - final_training_acc
- Visualizations:
    - viz/
        - jax_iter_vs_seed_accuracy_mean_sd.png
        - torch_iter_vs_batch_accuracy_mean_se.png
        - first_vs_steady_jax_vs_torch.png
        - final_test_accuracy_comparison.png



--- 

## Project Structure
- src/
  - jax_train.py
  - pytorch_train.py
  - mlp_scratch_jax.py
  - data_mnist.py
  - reusable.py
  - viz_script.py
- config.yaml
- init.sh
- requirements.txt
- results/
- logs/
- metrics/
- Model_Checkpoints/
- viz/


--- 

## GPU verification
to Verify JAX Device
```bash
    python -c "import jax; print(jax.devices()); print(jax.default_backend())"
```

to verify PyTorch Device:  
```bash
    python -c "import torch; print(torch.cuda.is_available())"
```
