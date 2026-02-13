from __future__ import annotations
import numpy as np 
from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split

def load_mnist(test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42 ):

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"].astype(np.float32) / 255.0 
    y = mnist["target"].astype(np.int32)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    #train_size = 1 - (test_size + val_size)

    val_size = val_size / (1 - test_size)

    X_train, X_val, y_train,y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_val
    )

    return X_train, y_train, X_test, y_test, X_val, y_val

 