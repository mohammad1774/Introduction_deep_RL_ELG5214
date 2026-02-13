from __future__ import annotations
from typing import Dict, Tuple, Any 
import time 

import jax 
import jax.numpy as jnp 

Params = Dict[str, jnp.ndarray]

def one_hot_encode(y: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    y = jnp.asarray(y).astype(jnp.int32)
    return jnp.eye(num_classes, dtype=jnp.float32)[y]

def init_params(key: jax.Array, input_dim: int, hidden1: int, hidden2: int, output_dim: int) -> Params:
    k1, k2, k3 = jax.random.split(key, 3) 

    W1 = jax.random.normal(k1, (input_dim, hidden1), dtype=jnp.float32) * jnp.sqrt(2.0/input_dim)
    b1 = jnp.zeros((1, hidden1), dtype=jnp.float32)

    W2 = jax.random.normal(k2, (hidden1, hidden2 ), dtype=jnp.float32) * jnp.sqrt(2.0/hidden1) 
    b2 = jnp.zeros((1, hidden2), dtype=jnp.float32)
    
    W3 = jax.random.normal(k3, (hidden2, output_dim), dtype=jnp.float32) * jnp.sqrt(2.0/hidden2) 
    b3 = jnp.zeros((1, output_dim), dtype=jnp.float32)
    
    return {"W1": W1, "b1":b1, "W2":W2,"b2":b2, "W3":W3,"b3":b3 }

def relu(z: jnp.ndarray) -> jnp.ndarray: 
    return jnp.maximum(z, 0.0)

def log_softmax(logits: jnp.ndarray) -> jnp.ndarray:
    z = logits - jnp.max(logits, axis=1, keepdims=True)
    return z - jnp.log(jnp.sum(jnp.exp(z), axis=1, keepdims=True))

def forward(X: jnp.ndarray, params: Params) -> Tuple[jnp.ndarray, Tuple[Any, ...]]:
    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)

    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = relu(Z2) 

    Z3 = A2 @ params["W3"] + params["b3"] 
    A3 = jnp.exp(log_softmax(Z3))

    cache = (X,Z1,A1,Z2, A2, Z3,A3)

    return A3 , cache 

def compute_loss(y_true_onehot: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray: 
    eps = 1e-9
    return -jnp.mean(jnp.sum(y_true_onehot*jnp.log(y_pred + eps), axis=1))

def loss_function(params: Params, X: jnp.ndarray, y_true_onehot: jnp.ndarray) -> jnp.ndarray:
    y_pred, _ = forward(X,params)
    return compute_loss(y_true_onehot, y_pred)

def accuracy(params: Params, X: jnp.ndarray, y_true_onehot: jnp.ndarray) -> jnp.ndarray:
    y_pred, _ = forward(X, params)
    pred = jnp.argmax(y_pred,axis=1)
    true = jnp.argmax(y_true_onehot, axis=1)
    return jnp.mean(pred == true)


def update_params_sgd(params: Params, grads: Params, lr: float) -> Params:
    # Explicit, no tree_map
    return {
        "W1": params["W1"] - lr * grads["W1"],
        "b1": params["b1"] - lr * grads["b1"],
        "W2": params["W2"] - lr * grads["W2"],
        "b2": params["b2"] - lr * grads["b2"],
        "W3": params["W3"] - lr * grads["W3"],
        "b3": params["b3"] - lr * grads["b3"],
    }

def make_batches(key: jax.Array, X: jnp.ndarray, Y: jnp.ndarray, batch_size: int):
    n = X.shape[0]
    perm = jax.random.permutation(key, n)
    Xs = X[perm]
    Ys = Y[perm]
    for i in range(0, n, batch_size):
        yield Xs[i : i + batch_size], Ys[i : i + batch_size]

def train_step(params: Params, Xb: jnp.ndarray, Yb: jnp.ndarray, lr: float):
    grads = jax.grad(loss_function)(params, Xb, Yb)
    loss_val = loss_function(params, Xb, Yb)
    new_params = update_params_sgd(params, grads, lr)
    return new_params, loss_val 

# jit_train_step = jax.jit(train_step)

# def train(
#         key: jax.Array,
#         X: jnp.ndarray,
#         Y: jnp.ndarray,
#         params: Params,
#         epochs: int=10,
#         lr: float = 0.01,
#         batch_size: int = 128,
#         use_jit: bool =True 
# ): 
#     step_fn = jit_train_step if use_jit else train_step 

#     first_epoch_time = None 
#     second_epoch_time = None 

#     for epoch in range(epochs):
#         key, k_epoch = jax.random.split(key, 2)
#         t0 = time.time()

#         losses = []
#         for Xb, Yb in make_batches(k_epoch,X, Y, batch_size):
#             params, loss_val = step_fn(params, Xb, Yb, lr)
#             losses.append(loss_val)
        