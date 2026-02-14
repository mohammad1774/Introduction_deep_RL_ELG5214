import os 
import time 
import random 
import logging 
from typing import Dict, List 
import numpy as np 
import pandas as pd 
import yaml 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms

from src.support.data_mnist import load_mnist
from src.support.reusable import * 

class NumpyMNIST(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.x = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x,y 


class MLP(nn.Module):
    def __init__(self, input_dim=784, h1=256,h2=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
           # nn.Flatten(),
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2,num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
def append_summary(path: str, row: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)

def main(config, seed, met_df, batch_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    run_id = seed * 100000 + batch_size 
    logger = setup_logger(run_id, config['train']['logs_path_torch'])


    logger.info(f"Starting run with seed = {seed} and Batch Size = {batch_size}")
    logger.info(f"The Device is : {device}")
    logger.info(f"Total Epochs = {config['train']['epochs']}")

    set_global_seed(seed)

    X_train, y_train, X_test, y_test, X_val, y_val = load_mnist(
        test_size=config["data"].get("test_size",0.1),
        val_size=config["data"].get("val_size",0.1),
        random_state = seed 
    )

    train_loader = DataLoader(NumpyMNIST(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(NumpyMNIST(X_test, y_test), batch_size=2048, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(NumpyMNIST(X_val, y_val), batch_size=2048, shuffle=False, num_workers=2, pin_memory=True)

    model = MLP(
        input_dim=X_train.shape[1],
        h1=config["model"]["hidden1"],
        h2=config["model"]["hidden2"],
        num_classes=int(np.max(y_train)+1),
    ).to(device)

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters() , lr=float(config["train"]["lr"]))
    

    epochs = config['train']['epochs']
    model.train()
    checkpoint_path = config['train']['checkpoint_path']
    global_iter = 0
    total_iters = epochs * len(train_loader)
    half_iter = total_iters // 2
    logger.info(f"Total iterations = {total_iters}")
    epoch_times=[]

    def eval_acc(loader):
        model.eval()
        correct=0 
        total=0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        model.train() 
        return correct / max(1,total)
    

    for epoch in range(1, epochs+1):
        t0 = time.time()
        running_loss = 0.0
        running_correct = 0
        running_total = 0


        for xb,yb in train_loader:
            global_iter += 1

            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                iter_acc = (pred == yb).float().mean().item()
            
            batch_loss = loss.item()
            bs = xb.size(0)

            running_loss += batch_loss * bs
            running_correct += (pred == yb).sum().item()
            running_total += bs 


            # preds = torch.argmax(logits,dim=1)

            # iter_acc = (preds==y).float().mean().item()
            logger.info(
                    f"| An Iteration is Completed"
                    f"| epoch={epoch}/{epochs} "
                    f"| Iteration = {global_iter}/{total_iters}"
                    f"| loss={batch_loss:.6f}"
                    f"| acc={iter_acc*100:.2f}%",
                    )    


            met_df.add(seed=seed,
                       iteration=global_iter,
                       total_iter=total_iters, 
                       epoch=epoch, 
                       loss=loss.item(),
                       accuracy=iter_acc*100,
                       bs=batch_size)

            if global_iter == half_iter:
                ckpt = os.path.join(checkpoint_path, f"torch_seed{seed}_bs{batch_size}_half.pt")
                torch.save(model.state_dict(), ckpt)
                logger.info(f"Saved halfway checkpoint: {ckpt}")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        epoch_times.append(t1-t0)


        epoch_loss = running_loss/ running_total 
        epoch_acc = running_correct/running_total 
        
        val_acc = eval_acc(val_loader)
        test_acc = eval_acc(test_loader)

        first_epoch_time = epoch_times[0] 
        steady_epoch_time = epoch_times[1] if (len(epoch_times) >= 2 and epoch>=1) else None 

        print(f"Epoch Completed | Epoch {epoch}"
              f"| seed={seed} | bs={batch_size}" 
              f"| loss = {epoch_loss:.4f}"
              f"| train_acc={epoch_acc*100:.2f}% | val_acc={epoch_acc*100:.2f}%"
              f"|  test_acc={test_acc*100:.2f}%"
              f"| t_first={first_epoch_time:.4f}s | t_steady={steady_epoch_time}"
              )
        logger.info(
            f"| Epoch Done | seed={seed} | bs={batch_size} "
            f"| epoch={epoch}/{epochs} | loss={epoch_loss:.6f} "
            f"| train_acc={epoch_acc*100:.2f}% | val_acc={val_acc*100:.2f}% | test_acc={test_acc*100:.2f}% "
            f"| t_first={first_epoch_time:.4f}s | t_steady={steady_epoch_time}"
        )
        met_df.add(
            seed=seed,
            iteration=0,
            total_iter=0,
            epoch=epoch,
            loss=epoch_loss,
            accuracy=epoch_acc * 100,
            epoch_end=True,
            bs=batch_size,
            val_acc=val_acc * 100,
            test_acc=test_acc * 100,
            fst=first_epoch_time,
            steadyet=steady_epoch_time if steady_epoch_time is not None else -1.0,
        )

    ckpt_final = os.path.join(checkpoint_path, f"torch_seed{seed}_bs{batch_size}_final.pt")
    torch.save(model.state_dict(), ckpt_final)
    logger.info(f"Saved final checkpoint: {ckpt_final}")
    
    summary_path = config["train"].get("summary_csv", "results/summary.csv")
    append_summary(summary_path, {
        "framework": "torch",
        "seed": seed,
        "batch_size": batch_size,
        "first_epoch_time_s": epoch_times[0],
        "steady_epoch_time_s": epoch_times[1] if len(epoch_times) >= 2 else None,
        "final_test_acc": test_acc,
        "final_training_acc": epoch_acc*100
    })

    return met_df

if __name__ == "__main__":
    config = load_config()
    met_df = MetricsDataset("Assignment1_torch")

    for seed in config["seeds"]:
        for batch in config["batch_size"]:
            met_df = main(config, seed, met_df, batch)

    file_locs = met_df.save(output_dir=config["train"]["metrics_out_dir"])
    print("Saved metrics:", file_locs)
