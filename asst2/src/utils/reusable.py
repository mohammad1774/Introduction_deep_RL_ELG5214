import os
import logging
import pandas as pd

class RLMetricsDataset:
    def __init__(self, proj_name: str):
        self.proj_name = proj_name
        self.episode_records = []
        self.summary_records = []

    def add_episode(
            self,
            seed: int,
            episode: int,
            reward: float,
            episode_length: int,
            algorithm: str,
            lr: float,
            gamma: float,
            loss: float = 0.0,
            eval_success_rate: float = -1.0,
    ):
        self.episode_records.append({
            "seed": seed,
            "episode": episode,
            "reward": reward,
            "episode_length": episode_length,
            "loss": loss,
            "eval_success_rate": eval_success_rate,
            "algorithm": algorithm,
            "learning_rate": lr,
            "gamma": gamma
        })

    def add_summary(
            self,
            seed: int,
            algorithm: str,
            lr: float,
            gamma: float,
            final_mean_reward: float,
            final_success_rate: float,
            backend: str,
            devices: str,
            action: str = "stochastic",
            mean_length: float = 0):
        self.summary_records.append({
            "seed": seed,
            "algorithm": algorithm,
            "learning_rate": lr,
            "gamma": gamma,
            "final_mean_reward": final_mean_reward,
            "final_success_rate": final_success_rate,
            "backend": backend,
            "devices": devices,
            "action": action, 
            "mean_length": mean_length
        })

    def save(self, output_dir: str="metrics", filename: str | None=None):
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            filename_iters = f"{self.proj_name}_dataset_metrics.csv"
            filename_summ = f"{self.proj_name}_dataset_metrics_summary.csv"
        else:
            filename_iters = filename
            filename_summ = filename.replace(".csv", "_summary.csv")

        path_iter = os.path.join(output_dir, filename_iters)
        path_summ = os.path.join(output_dir, filename_summ)

        pd.DataFrame(self.episode_records).to_csv(path_iter, index=False)
        pd.DataFrame(self.summary_records).to_csv(path_summ, index=False)
        return {"iteration": path_iter, "summary": path_summ}


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