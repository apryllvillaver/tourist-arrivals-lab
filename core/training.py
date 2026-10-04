from __future__ import annotations

import itertools
import os
from typing import Callable

import numpy as np

from core.pipeline import PipelineError, safe


def tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    import tensorflow as tf
    return tf


def network(lookback: int, features: int, units: int, dropout: float):
    tf = tensorflow()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, features)),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss="mse")
    return model


def predict(model, X: np.ndarray) -> np.ndarray:
    # Direct inference avoids constructing a new tf.data thread pool for every SHAP call.
    chunks = [model(np.asarray(X[i:i+512], dtype=np.float32), training=False).numpy() for i in range(0, len(X), 512)]
    return np.concatenate(chunks).reshape(-1, 1)


def train(bundle: dict, epochs: int = 100, search: str = "full", progress: Callable | None = None):
    if not 5 <= epochs <= 150:
        raise PipelineError("Choose between 5 and 150 maximum epochs.")
    if search not in {"quick", "full"}:
        raise PipelineError("Choose the quick or full search.")
    tf = tensorflow()
    seed = bundle["config"].seed
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    grid = list(itertools.product([32, 64], [.1, .3], [16, 32])) if search == "full" else [(32, .1, 16), (64, .3, 32)]
    best_loss, best_params, best_weights, best_history = float("inf"), None, None, None
    trials = []
    lookback, features = bundle["X_train"].shape[1:]
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    for trial, (units, dropout, batch_size) in enumerate(grid):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = network(lookback, features, units, dropout)
        fit_data = tf.data.Dataset.from_tensor_slices((bundle["X_train"], bundle["y_train"])).batch(batch_size).with_options(options)
        val_data = tf.data.Dataset.from_tensor_slices((bundle["X_validation"], bundle["y_validation"])).batch(batch_size).with_options(options)

        class Update(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if progress:
                    progress((trial + (epoch + 1) / epochs) / len(grid), f"Trial {trial+1}/{len(grid)} · epoch {epoch+1}/{epochs}")

        history = model.fit(fit_data, validation_data=val_data, epochs=epochs, shuffle=False, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True), Update()]).history
        loss = float(min(history["val_loss"]))
        if not np.isfinite(loss):
            raise PipelineError("Training produced a non-finite loss. Review the prepared data and source outliers.")
        params = {"units": units, "dropout": dropout, "batch_size": batch_size}
        trials.append({**params, "validation_mse": loss, "epochs_run": len(history["loss"]), "best_epoch": int(np.argmin(history["val_loss"])) + 1})
        if loss < best_loss:
            best_loss, best_params, best_history = loss, params, history
            best_weights = [w.copy() for w in model.get_weights()]
        if progress:
            progress((trial + 1) / len(grid), f"Completed trial {trial+1}/{len(grid)}")
    tf.keras.backend.clear_session()
    model = network(lookback, features, best_params["units"], best_params["dropout"])
    model.set_weights(best_weights)
    report = {"best_params": best_params, "validation_mse": best_loss, "search": search, "seed": seed,
        "max_epochs": epochs, "trials": sorted(trials, key=lambda t: t["validation_mse"]),
        "history": [{"epoch": i+1, "training": a, "validation": b} for i, (a, b) in enumerate(zip(best_history["loss"], best_history["val_loss"]))],
        "policy": "No shuffling. Early stopping and model selection use validation MSE only. The test set is not read during training."}
    return model, safe(report)
