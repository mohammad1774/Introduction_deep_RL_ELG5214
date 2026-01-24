import os 
import math 
import time 
import random 
import logging 
from typing import Dict, List, Tuple 
from datetime import datetime 
import platform 
import sys 
import pandas as pd
import numpy as np 
import yaml 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from typing import Dict, List 



class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self, x):
        return self.net(x)
    

class MetricsDataset:
    """
    Collects metrics data per iteration or epoch and exports them as a DataFrame.
    """
    def __init__(self, proj_name: str):

        self.proj_name = proj_name 

        self.records: List[Dict] = []
        self.epoch_records: List[Dict] = []
    
    def add(self, seed: int, iteration: int, total_iter: int, loss: float, accuracy: float, epoch: int, epoch_end: bool=False):
        """
        Call this per iteration at the end of the run of an iteration to document the metrics. 
        
        If it is called at the end of epoch with bool epoch_end = True then the data is saved into a Dataset of Epochs to compare between different epochs.

        """

        if epoch_end:
            self.epoch_records.append(
                {
                    "Seed": seed,
                    "Epoch": epoch,
                    "Loss": loss,
                    "Accuracy": accuracy
                }
            )
        else:
            self.records.append(
                {
                    "Seed": seed,
                    "Epoch": epoch,
                    "Loss": loss,
                    "Accuracy": accuracy,
                    "Iteration": iteration, 
                    "Total_iter": total_iter,
                }
            )

    def save(self, output_dir: str="metrics",filename: str | None=None):
        os.makedirs(output_dir, exist_ok=True)

        if filename is None:
            filename_iters = f"{self.proj_name}_dataset_metrics.csv"
            filename_epoch = f"{self.proj_name}_dataset_metrics_epochs.csv"

        path_iter = os.path.join(output_dir, filename_iters)
        path_epoch = os.path.join(output_dir, filename_epoch)
        df = pd.DataFrame(self.records)
        epoch_df = pd.DataFrame(self.epoch_records)
        df.to_csv(path_iter, index=False)
        epoch_df.to_csv(path_epoch, index=False)

        return {"iteration": path_iter, "epoch": path_epoch}


def load_config(path: str="config.yaml") -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Please add the config.yaml file")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    if not isinstance(cfg,dict):
        raise ValueError("config.yaml must parse into a dictionary")
    return cfg 


def set_global_seed(seed: int, deterministic: bool=True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_logger(run_id: int, path: str):
    os.makedirs(path, exist_ok=True)

    logger = logging.getLogger(f"run{run_id}")
    logger.setLevel(logging.INFO)

    logger.handlers.clear()
    logger.propagate = False 

    log_path = f"{path}/run{run_id}.log"
    handler = logging.FileHandler(log_path, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger 


def main(config,seed,met_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    logger = setup_logger(seed, config['train']['logs_path'])

    logger.info(f"Starting run {seed} with seed = {seed}")
    logger.info(f"Total Epochs = {config['train']['epochs']}")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root = "./data",
        train = True,
        download=True,
        transform = transform 
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = 1024,
        shuffle = True 
    )

    model = MLP().to(device)

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters() , lr=1e-3)

    epochs = config['train']['epochs']
    model.train()
    checkpoint_path = config['train']['checkpoint_path']
    global_iter = 0
    total_iters = epochs * len(train_loader)
    half_iter = total_iters // 2
    logger.info(f"Total iterations = {total_iters}")

    for epoch in range(1, epochs+1):
        running_loss = 0.0
        running_correct = 0
        running_total = 0


        for batch_idx, (x,y) in enumerate(train_loader, start=1):
            global_iter += 1
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits,dim=1)

            iter_acc = (preds==y).float().mean().item()
            logger.info(
                    f"| An Iteration is Completed"
                    f"| epoch={epoch}/{epochs} "
                    f"| Iteration = {global_iter}/{total_iters}"
                    f"| loss={loss.item():.6f} "
                    f"| acc={iter_acc*100:.2f}%",
                    )    

            running_correct += (preds == y).sum().item()
            running_total += x.size(0)
            met_df.add(seed=seed,iteration=global_iter,total_iter=total_iters, epoch=epoch, loss=loss.item(),accuracy=iter_acc*100)

            if global_iter == half_iter:
                torch.save(model.state_dict(), f"{checkpoint_path}/run{seed}_half.pt")
                logger.info(f"saving halfway model checkpoints at the path : {checkpoint_path}/run{seed}_half.pt")
            
        epoch_loss = running_loss/ running_total 
        epoch_acc = running_correct/running_total 
        print(f"Epoch {epoch} | loss = {epoch_loss:.4f} | acc={epoch_acc:.4f}")
        logger.info(
                    f"| The Epoch is Completed"
                    f"| epoch={epoch}/{epochs}"
                    f"| loss={epoch_loss:.6f}"
                    f"| acc={epoch_acc*100:.2f}%"
                    )
        met_df.add(seed=seed,iteration=0,total_iter=0, epoch=epoch, loss=epoch_loss,accuracy=epoch_acc*100,epoch_end=True)
    
    torch.save(model.state_dict(), f"{checkpoint_path}/run{seed}_final.pt")
    logger.info(f"Saving the final model checkpoints with seed : {seed} at location: {checkpoint_path}/run{seed}_final.pt")

    return met_df


class learning_curves:
    """
    This python class will help us build various learning curves with the metrics dataset created. 

    It is expected for the dataset to have the following columns: seed, epoch, iteration, total_iterations, accuracy, loss
    """
    def __init__(self, file_path: str):
        print(file_path)
        self.df = pd.read_csv(file_path)

    def mean_std_dev(self, output_path):
        n_runs = self.df["Seed"].nunique()

        metrics = ["Loss", "Accuracy"]

        fig, axes = plt.subplots(1,2,figsize=(12,4))
        for i,metric in enumerate(metrics):
            stats = (
                self.df.groupby("Iteration")[metric]
                .agg(["mean","std"])
                .reset_index()
            )

            x = stats["Iteration"]

            ax = axes[i]
            ax.plot(x,stats["mean"], label="Mean Loss")
            ax.fill_between(
                x,
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                alpha=0.3,
                label = "+- 1 SD"
            )

            ax.set_title(f"Mean +- SD - {metric}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(f"Training {metric}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_path}/mean_sd.png",dpi = 2000)
        plt.show()
    
    def mean_se_dev(self, output_path):
        n_runs = self.df["Seed"].nunique()

        metrics = ["Loss", "Accuracy"]

        fig, axes = plt.subplots(1,2,figsize=(12,4))
        for i,metric in enumerate(metrics):
            stats = (
                self.df.groupby("Iteration")[metric]
                .agg(["mean","std"])
                .reset_index()
            )
            stats["ste"] = stats["std"] / (n_runs ** 0.5)

            x = stats["Iteration"]

            ax = axes[i]
            ax.plot(x,stats["mean"], label="Mean Loss")
            ax.fill_between(
                x,
                stats["mean"] - stats["ste"],
                stats["mean"] + stats["ste"],
                alpha=0.3,
                label = "+- 1 SE"
            )

            ax.set_title(f"Mean +- SE - {metric}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(f"Training {metric}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_path}/mean_se.png",dpi = 2000)
        plt.show()
    
    def all_curves(self,output_path):
        plt.figure(figsize = (7,4))
        n_runs = self.df["Seed"].nunique()

        for seed, dfr in self.df.groupby("Seed"):
            dfr = dfr.sort_values("Iteration")
            plt.plot(dfr["Iteration"], dfr["Accuracy"], label=f"Seed {seed}")
        
        plt.xlabel("Iteration")
        plt.ylabel("Training Accuracy (%)")
        plt.title("All Individual Learning Curves (Accuracy)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_path}/all_runs_accuracy.png", dpi=200)
        plt.show()


        

if __name__ == "__main__":
    config = load_config()
    print(config)
    met_df = MetricsDataset("Assignment0")
    for seed in config['seeds']:
        print(seed)
        set_global_seed(seed)
        met_df = main(config, seed , met_df)
    file_locs = met_df.save()

    plot = learning_curves(file_locs["iteration"])
    print("The Mean +- Standard Deviation Graphs for the training and Loss.")
    plot.mean_std_dev(config['metric_path'])
    print("The Mean +- Standard Error Graphs for the training and Loss.")
    plot.mean_se_dev(config['metric_path'])
    print("All runs super imposed Graphs for the training and Loss.")
    plot.all_curves(config['metric_path'])