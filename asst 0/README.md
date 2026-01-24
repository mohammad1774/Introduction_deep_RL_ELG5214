# Assignment 0 - Reproducibility Check (MNIST MLP)

This project trains a simple MLP on MNIST across multiple seeds, logs
metrics, saves checkpoints at half and final weights, and generates learning-curve plots for
reproducibility analysis.

## Quickstart

1) Create the environment and install dependencies:
   - Run `init.sh` (PowerShell: `bash init.sh`) 
   This file is something which was not required for the assignment , but this will give a seamless experience.
   This script downloads the libraries using the requirements.txt and creates the required folders and checks for the required files.
2) Train and generate files of Logs, Checkpoints, Metric Dataset, Plots:
   - Run `python repro_check.py`


The script downloads MNIST into `./data` on first run.

## Configuration

Edit `config.yaml` to control:
- `seeds`: list of random seeds to run.
- `train.epochs`: number of epochs.
- `train.checkpoint_path`: where checkpoints are saved.
- `train.logs_path`: where log files are saved.
- `metric_path`: where plots are saved.

Notes on current code behavior (as implemented in `repro_check.py`):
- Batch size is fixed at 1024 (not read from `config.yaml`).
- Learning rate is fixed at 1e-3 with Adam (not read from `config.yaml`).

## Outputs

Running `repro_check.py` produces:
- Logs: `./logs/run<seed>.log`
- Checkpoints: `./Model_Checkpoints/run<seed>_half.pt` and `run<seed>_final.pt`
- Metrics CSVs: `./metrics/Assignment0_dataset_metrics.csv` and
  `./metrics/Assignment0_dataset_metrics_epochs.csv`
- Plots in `metric_path` (default: `./viz/`):
  - `mean_sd.png`
  - `mean_se.png`
  - `all_runs_accuracy.png`

## Project Layout

- `repro_check.py`: main training/metrics/plotting script.
- `config.yaml`: run configuration.
- `init.sh`: create venv, install deps, and generate environment metadata.
- `env_metadata.py`: writes `environment_info.txt`.
- `requirements.txt`: Python dependencies.
- `data/`: MNIST dataset download location.
- `logs/`, `metrics/`, `Model_Checkpoints/`, `viz/`: outputs.

## Troubleshooting

- If you see CUDA errors, set `device: cpu` in `config.yaml` and re-run.
- If you are on Windows without `bash`, run `init.sh` from Git Bash or WSL.
