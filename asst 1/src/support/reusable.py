import os
import time
import random
import logging
from typing import Dict, List
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

import jax 
import jax.numpy as jnp 


class MetricsDataset:
    def __init__(self, proj_name: str):
        self.proj_name = proj_name
        self.records: List[Dict] = []
        self.epoch_records: List[Dict] = []

    def add(self, seed: int, iteration: int, total_iter: int, loss: float, accuracy: float, epoch: int, test_acc: float=0.0,val_acc: float=0.0,fst: float=0.0,steadyet: float=0.0,bs: int=0, epoch_end: bool=False ):
        if epoch_end:
            self.epoch_records.append({"Seed": seed, 
                                       "Epoch": epoch, 
                                       "Loss": loss, 
                                       "Accuracy": accuracy, 
                                       "Test_Accuracy": test_acc, 
                                       "Val_Accuracy":val_acc,
                                       "First_Epoch_Time":fst, 
                                       "Steady_Epoch_Time":steadyet,
                                       "Batch_Size": bs})
        else:
            self.records.append({
                "Seed": seed,
                "Epoch": epoch,
                "Loss": loss,
                "Accuracy": accuracy,
                "Iteration": iteration,
                "Total_iter": total_iter,
                "Validation_Acc": val_acc,
                "Batch_Size": bs

            })

    def save(self, output_dir: str="metrics", filename: str | None=None):
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            filename_iters = f"{self.proj_name}_dataset_metrics.csv"
            filename_epoch = f"{self.proj_name}_dataset_metrics_epochs.csv"
        else:
            filename_iters = filename
            filename_epoch = filename.replace(".csv", "_epochs.csv")

        path_iter = os.path.join(output_dir, filename_iters)
        path_epoch = os.path.join(output_dir, filename_epoch)

        pd.DataFrame(self.records).to_csv(path_iter, index=False)
        pd.DataFrame(self.epoch_records).to_csv(path_epoch, index=False)
        return {"iteration": path_iter, "epoch": path_epoch}


def load_config(path: str="config.yaml") -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Please add the config.yaml file")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must parse into a dictionary")
    return cfg


def set_global_seed(seed: int) -> jax.Array:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # JAX uses explicit PRNG keys; return one
    return jax.random.PRNGKey(seed)


def setup_logger(run_id: int, path: str):
    os.makedirs(path, exist_ok=True)
    logger = logging.getLogger(f"run{run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    log_path = f"{path}/run{run_id}.log"
    handler = logging.FileHandler(log_path, mode="w")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def save_run_summary_row(out_csv: str, row: Dict,):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.DataFrame([row])
    if os.path.exists(out_csv):
        df.to_csv(out_csv, model="a", header=False, index=False)
    else:
        df.to_csv(out_csv, mode="w", header=True, index=False)