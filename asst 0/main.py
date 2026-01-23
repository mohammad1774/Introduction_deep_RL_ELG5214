import os 
import math 
import time 
import random 
import logging 
from typing import Dict, List, Tuple 
from datetime import datetime 
import platform 
import sys 

import numpy as np 
import yaml 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

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

def main(config,seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    logger = setup_logger(seed)

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
        batch_size = 128,
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
            running_correct += (preds == y).sum().item()
            running_total += x.size(0)

            if global_iter == half_iter:
                torch.save(model.state_dict(), f"{checkpoint_path}/run{seed}_half.pt")
                logger.info(f"saving halfway model checkpoints at the path : {checkpoint_path}/run{seed}_half.pt")
            
        epoch_loss = running_loss/ running_total 
        epoch_acc = running_correct/running_total 
        print(f"Epoch {epoch} | loss = {epoch_loss:.4f} | acc={epoch_acc:.4f}")
    
    torch.save(model.state_dict(), f"{checkpoint_path}/run{seed}_final.pt")
    logger.info(f"Saving the final model checkpoints with seed : {seed} at location: {checkpoint_path}/run{seed}_final.pt")

if __name__ == "__main__":
    config = load_config()
    print(config)
    for seed in config['seeds']:
        print(seed)
        set_global_seed(seed)
        main(config, seed)
    