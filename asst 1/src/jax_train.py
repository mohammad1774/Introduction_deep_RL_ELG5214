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

from src.support.data_mnist import load_mnist
from src.jax_model.mlp_scratch_jax import *
from src.support.reusable import * 

def train_step(params: Params, Xb: jnp.ndarray, Yb: jnp.ndarray, lr: float):
    grads = jax.grad(loss_function)(params, Xb, Yb)
    loss_val = loss_function(params, Xb, Yb)
    new_params = update_params_sgd(params, grads, lr)
    return new_params, loss_val 


jit_train_step = jax.jit(train_step)

def save_checkpoint_npz(params: Dict[str, jnp.ndarray], path: str):
    """
    Saves JAX arrays as .npz (portable). This is torch.save equivalent.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    params_np = {k: np.array(v) for k, v in params.items()}
    np.savez(path, **params_np)

def main(config: Dict, seed: int, met_df: MetricsDataset,batch):
    run_id = seed * 100000 + batch
    logger = setup_logger(run_id, config["train"]["logs_path"])
    logger.info(f"MLP Implementation with JAX From scratch.")
    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"JAX backend: {jax.default_backend()}")
    logger.info(f"Straring run {seed} with Seed = {seed} and Batch Size={batch}")
    logger.info(f"Total Epochs = {config['train']['epochs']}")

    X_train_np, y_train_np,  X_test, y_test, X_val, y_val = load_mnist(test_size=config["data"].get("test_size", 0.1),
                                                                       val_size=config["data"].get("val_size",0.1),
                                                                       random_state = config["data"].get("random_state",42),
                                                                       )
    
    X_train = jnp.array(X_train_np, dtype=jnp.float32)
    y_train = jnp.array(y_train_np, dtype=jnp.int32)
    num_features = X_train.shape[1]
    num_classes = int(np.max(y_train_np) + 1)
    Y_train_oh = one_hot_encode(y_train, num_classes)

    X_test = jnp.array(X_test, dtype=jnp.float32)
    y_test = jnp.array(y_test, dtype=jnp.int32)
    Y_test_oh = one_hot_encode(y_test, num_classes)

    X_val = jnp.array(X_val, dtype=jnp.float32)
    y_val = jnp.array(y_val, dtype=jnp.int32)
    Y_val_oh = one_hot_encode(y_val, num_classes)


    key = set_global_seed(seed) 
    params = init_params(
        key=key,
        input_dim=int(num_features),
        hidden1= config["model"]["hidden1"],
        hidden2=config["model"]["hidden2"],
        output_dim = int(num_classes)
    )

    epochs = config["train"]["epochs"]
    batch_size = batch 
    lr = float(config["train"]["lr"])

    checkpoint_path = config["train"]["checkpoint_path"]
    global_iter = 0

    total_iters = epochs * int(np.ceil(X_train.shape[0] / batch_size))
    half_iter = total_iters // 2
    logger.info(f"Total iterations = {total_iters}")

    epoch_times = []

    for epoch in range(1, epochs+1):
        t0 = time.time()
        running_loss = 0.0 
        running_correct = 0 
        running_total = 0

        key, k_epoch = jax.random.split(key,2)
        for (Xb, Yb) in make_batches(k_epoch, X_train, Y_train_oh, batch_size):

            global_iter += 1 

            params , loss_val = jit_train_step(params, Xb, Yb, lr)

            loss_val = jax.block_until_ready(loss_val)

            probs, _ = forward(Xb, params)
            preds = jnp.argmax(probs, axis=1)
            true = jnp.argmax(Yb, axis=1)
            iter_acc = jnp.mean(preds == true)
            iter_acc = float(jax.device_get(iter_acc))
            #val_acc = float(jax.device_get(accuracy(params, X_val, Y_val_oh)))

            batch_loss = float(jax.device_get(loss_val))
            bs = Xb.shape[0]

            running_loss += batch_loss * bs
            running_correct += int(jax.device_get(jnp.sum(preds == true)))
            running_total += int(bs)

            logger.info(
                f"| An Iteration is Completed"
                f"| epoch={epoch}/{epochs} "
                f"| Iteration = {global_iter}/{total_iters}"
                f"| loss={batch_loss:.6f} "
                f"| acc={iter_acc*100:.2f}%"
                #f"| val acc={val_acc*100:.2f}"
            )

            met_df.add(
                seed=seed,
                iteration=global_iter,
                total_iter=total_iters,
                epoch=epoch,
                loss=batch_loss,
                accuracy=iter_acc * 100,
                #val_acc = val_acc * 100,
                bs=batch_size
            )

            if global_iter == half_iter:
                ckpt = f"{checkpoint_path}/run_seed{seed}_batch{batch}_half.npz"
                save_checkpoint_npz(params, ckpt)
                logger.info(f"saving halfway model checkpoints at the path : {ckpt}")

        jax.block_until_ready(params["W1"])
        t1 = time.time()
        epoch_times.append(t1 - t0)
        epoch_loss = running_loss / max(1, running_total)
        epoch_acc = running_correct / max(1, running_total)
        test_acc = float(jax.device_get(accuracy(params, X_test, Y_test_oh)))
        val_acc = float(jax.device_get(accuracy(params, X_val, Y_val_oh)))
        first_epoch_time = epoch_times[0]
        steady_epoch_time = epoch_times[1] if len(epoch_times) >=2 else None

        print(f"Epoch {epoch} | loss = {epoch_loss:.4f} | acc={epoch_acc*100:.4f} | seed={seed} | batchSize={batch_size}")
        logger.info(
            f"| The Epoch is Completed"
            f"seed={seed} | batch={batch_size}"
            f"| epoch={epoch}/{epochs}"
            f"| loss={epoch_loss:.6f}"
            f"| Acc={epoch_acc*100:.2f}%"
            f"| Test_Acc={test_acc*100:.2f}%"
            f"| Val Acc={val_acc*100:.2f}%"
            f"| First Steady Time={first_epoch_time}"
            f"| Steady Epoch Time={steady_epoch_time}"
        )

        met_df.add(
            seed=seed,
            iteration=0,
            total_iter=0,
            epoch=epoch,
            loss=epoch_loss,
            accuracy=epoch_acc * 100,
            epoch_end=True,
            test_acc=test_acc *100,
            val_acc=val_acc*100,
            fst = first_epoch_time,
            steadyet = steady_epoch_time,
            bs=batch_size
        )

    ckpt_final = f"{checkpoint_path}/run_seed{seed}_batch{batch}_final.npz"
    save_checkpoint_npz(params, ckpt_final)
    logger.info(f"Saving the final model checkpoints with seed : {seed} at location: {ckpt_final}")

    append_summary(config["train"].get("summary_csv", "results/summary.csv"), {
        "framework":"jax",
        "seed": seed,
        "batch_size": batch_size,
        "first_epoch_time_s": first_epoch_time,
        "steady_epoch_time": steady_epoch_time,
        "final_test_acc": test_acc,
        "final_training_acc": epoch_acc *100
    })

    return met_df

def append_summary(path: str, row: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)



if __name__ == "__main__":
    config = load_config()
    met_df = MetricsDataset("Assignment1_JAX")

    for seed in config["seeds"]:
        for batch in config["batch_size"]:
            met_df = main(config, seed, met_df, batch)

    file_locs = met_df.save(output_dir=config["train"]["metrics_out_dir"])
