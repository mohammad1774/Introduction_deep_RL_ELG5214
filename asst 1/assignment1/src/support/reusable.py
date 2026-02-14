import os
import time
import random
import logging
from typing import Dict, List
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

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
                #"Validation_Acc": val_acc,
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

class LearningCurvesComparer:
    """
    Expected Columns (Iteration Level df)
    - Iteration (int)
    - Seed (int)
    - bs (int) 
    - framework (str)  # "jax" or "torch"
    - Loss (float)
    - Accuracy (float)

    Expected columns (epoch-level df) for epoch time plots (if available):
      - Epoch (int)
      - epoch_time_s (float) 
        - First Epoch Time
        - Steady Epoch Time #In our case the second epoch time
      - Seed, bs, framework
    """

    def __init__(self, iter_csv_paths: dict, epoch_csv_paths: dict | None = None):
        """
        iter_csv_paths: {"jax": "path/to/jax_iter.csv", "torch": "path/to/torch_iter.csv"}
        epoch_csv_paths: {"jax": "path/to/jax_epoch.csv", "torch": "path/to/torch_epoch.csv"} 
        """
        self.iter_df = self._load_and_tag(iter_csv_paths)
        self.epoch_df = self._load_and_tag(epoch_csv_paths) if epoch_csv_paths else None


    def _load_and_tag(self,paths: dict) -> pd.DataFrame:
        dfs = []
        for fw, p in paths.items():
            df = pd.read_csv(p)
            df["framework"] = fw 
            dfs.append(df)
            
        return pd.concat(dfs, ignore_index=True)


    def _stats_mean_band(
            self,
            df: pd.DataFrame,
            x_col: str,
            y_col: str,
            group_cols: list[str],
            band: str = "sd",
    ) -> pd.DataFrame:
        """ 
        This is helper function which will calculate the mean standard error 
        mean standard deviation for each pair of X column and Y column, 
        which are chosen by us while calling this function
        """

        stats = (
            df.groupby(group_cols + [x_col])[y_col]
            .agg(["mean", "std","count"])
            .reset_index()
            .sort_values(group_cols+ [x_col])
        )

        if band == "se":
            stats["band"] = stats["std"] / np.sqrt(stats["count"].clip(lower=1))
        else:
            stats["band"] = stats["std"]
        return stats 
    
    def _plot_mean_band(
            self,
            df: pd.DataFrame,
            x_col: str,
            y_col: str,
            group_cols: list[str],
            title: str,
            out_path: str,
            band: str="sd",
    ): 
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        stats = self._stats_mean_band(df, x_col, y_col, group_cols, band=band)

        fig, ax = plt.subplots(figsize=(8,4))
        for key, g in stats.groupby(group_cols):
            if not isinstance(key, tuple):
                key = (key,)
            label = "|".join([f"{c}={v}" for c,v in zip(group_cols, key)])

            ax.plot(g[x_col],g["mean"], label=label)
            ax.fill_between(g[x_col],g["mean"]-g["band"], g["mean"]+g["band"],alpha=0.25)
        
        ax.set_title(title + (" (Mean ± SD)" if band == "sd" else " (Mean ± SE)"))
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

    def _plot_all_curves(
            self,
            df: pd.DataFrame,
            x_col: str,
            y_col: str,
            group_cols: list[str],
            title: str,
            out_path: str,
            max_legend: int=20,
    ):
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        fig, ax = plt.subplots(figsize=(10,10))
        for i , (key,g) in enumerate(df.groupby(group_cols)):
            g = g.sort_values(x_col)
            if not isinstance(key, tuple):
                key = (key,)
            label = " | ".join([f"{c} = {v}" for c,v in zip(group_cols,key)])
            ax.plot(g[x_col],g[y_col],label=label)

        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)

        # Avoid giant legends
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) <= max_legend:
            ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

    def plot_iteration_vs_col(self,framework: str, metric: str, ref_col: str,out_dir: str):
        """  
        Each Iteration Metric value vs The seed for each framework 

        Metric : "Loss" OR "Accuracy" 
        Framework: "JAX" or "PyTorch"
        """
        df = self.iter_df[self.iter_df["framework"] == framework].copy()

        self._plot_all_curves(
            df = df,
            x_col = "Iteration",
            y_col = metric,
            group_cols=[ref_col],
            title=f"{framework.upper()} | {metric} vs Iteration (per {ref_col})",
            out_path=os.path.join(out_dir, f"{framework}_iter_vs_seed_{metric.lower()}_all.png"),
        )

        self._plot_mean_band(
            df= df,
            x_col="Iteration",
            y_col=metric,
            group_cols = [ref_col],
            title=f"{framework.upper()} | {metric} vs Iteration groupbed by {ref_col}",
            out_path=os.path.join(out_dir, f"{framework}_iter_vs_{ref_col}_{metric.lower()}_mean_sd.png"),
            band="sd"
        )

        self._plot_mean_band(
            df=df,
            x_col="Iteration",
            y_col=metric,
            group_cols=[ref_col],
            title=f"{framework.upper()} | {metric} vs Iteration grouped by {ref_col}",
            out_path=os.path.join(out_dir, f"{framework}_iter_vs_{ref_col}_{metric.lower()}_mean_se.png"),
            band="se",
        )

def epoch_time_comparison(summary_csv: str="results/summary.csv", output_dir: str="viz"):
    df = pd.read_csv(summary_csv)

    # normalize column name
    if "steady_epoch_time_s" not in df.columns and "steady_epoch_time" in df.columns:
        df = df.rename(columns={"steady_epoch_time": "steady_epoch_time_s"})

    # wide -> long
    df_long = df.melt(
        id_vars=["framework", "batch_size"],
        value_vars=["first_epoch_time_s", "steady_epoch_time_s"],
        var_name="epoch_type",
        value_name="time_s"
    )

    df_long["epoch_type"] = df_long["epoch_type"].map({
        "first_epoch_time_s": "first_epoch_time",
        "steady_epoch_time_s": "steady_epoch_time",
    })

    # combined hue key
    df_long["fw_bs"] = df_long["framework"] + "_bs" + df_long["batch_size"].astype(str)

    plt.figure(figsize=(8, 4))
    sns.barplot(
        data=df_long,
        x="epoch_type",
        y="time_s",
        hue="fw_bs"
    )

    plt.ylabel("Training Time (seconds)")
    plt.title("First vs Steady Epoch Time (JAX vs PyTorch)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/first_vs_steady_jax_vs_torch.png", dpi=200)
    plt.show()

def framework_vs_final_acc(summary_csv: str="results/summary.csv", output_dir: str="viz"):
    df = pd.read_csv(summary_csv)

    df["batch_size"] = df["batch_size"].astype(int)
    df["final_test_acc"] = pd.to_numeric(df["final_test_acc"], errors="coerce")

    # if accuracies stored as 0-1, convert to %
    if df["final_test_acc"].max() <= 1.0:
        df["final_test_acc"] = 100.0 * df["final_test_acc"]

    plt.figure(figsize=(7, 4))
    sns.barplot(
    data=df,
    x="batch_size",
    y="final_test_acc",
    hue="framework"
    )
    plt.xlabel("Batch Size")
    plt.ylabel("Final Test Accuracy (%)")
    plt.title("Final Test Accuracy vs Batch Size (JAX vs PyTorch)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_test_acc_jax_vs_torch.png", dpi=200)
    plt.show()