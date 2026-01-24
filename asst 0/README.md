# Assignment 0 - Reproducibility Check (MNIST MLP)

This project trains a simple MLP on MNIST across multiple seeds, logs
metrics, saves checkpoints at half and final weights, and generates learning-curve plots for
reproducibility analysis.

## Quickstart

1) Create the environment and install dependencies:
   - Run `init.sh` (PowerShell: `bash init.sh`) 
   ```python
      $ bash init.sh
   ```

   This file is something which was not required for the assignment , but this will give a seamless experience.
   
   This script downloads the libraries using the requirements.txt and creates the required folders and checks for the required files.
   
   Expected outputs:
      - environment_info.txt, this is a file which documents the OS, GPU, Cuda Version, PyTorch Version 
        *Note: The exact version of PyTorch along with CUDA Wheel version is accordance with my laptop's GPU. So, there is an uncertainty for 
        the users GPU CUDA Version, if there is some indiscrepancy then GPU will be "N/A" and CPU will be used as the training device.*
      - logs folder and logs files from runs.
      - metrics folder containing the Assignment0_metric_dataset csv files.
      - Model_Checkpoints Folder with the weight of different runs of model training.
      - viz folder with the generated plots for the learning curves.

2) Create Env and installing dependenices and Train and generate files of Logs, Checkpoints, Metric Dataset, Plots manually:
   - Create an Virtual Environment 
      ```python
        $python -m venv .venv
      ```
   - Activate the Environment
      ```python
        $.venv/Scripts/activate
      ```
   - Installing Dependencies
      ```python
        $ python -m pip install --upgrage pip
        $ pip install -r requirements.txt
      ```
   - Running Metadata Script to generate environment info text file
      ```python
        $ python env_metadata.py
      ```
      This generates a environment_info.txt file which contains the os and other librarie information.
   - Creating Folders
      ```python
        $ /data
        $ /logs
        $ /metrics
        $ /Model_Checkpoints
        $ /viz
      ```
   - Running the main model training and plot generation script "repro_check.py"
      ```python
        $ Run python repro_check.py
      ```

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
