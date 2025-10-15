"""TFX Tuner module untuk model HeartDisease dengan KerasTuner BayesianOptimization.

Menyediakan:
- _build_model(): membangun arsitektur model dengan hyperparameters
- tuner_fn(): entry-point yang mengembalikan objek tuner dan fit_kwargs
"""

from types import SimpleNamespace
from typing import Dict, Any

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow import keras
import keras_tuner as kt

from .heart_common import (
    build_feature_block,
    make_datasets,
    make_metrics,
)


def _build_model(hp: kt.HyperParameters, tf_transform_output) -> keras.Model:
    """Bangun model tabular; memanfaatkan blok fitur bersama."""
    inputs, x, _ = build_feature_block(tf_transform_output)

    # ----- hyperparameters (pakai dict untuk mengurangi variabel lokal) -----
    hp_vals: Dict[str, Any] = {
        "units1": hp.Int("units1", 32, 128, step=32),
        "units2": hp.Int("units2", 16, 64, step=16),
        "dropout": hp.Float("dropout", 0.2, 0.6, step=0.1),
        "l2": hp.Float("l2", 1e-5, 1e-3, sampling="log"),
        "lr": hp.Choice("lr", [1e-3, 5e-4, 3e-4]),
        "label_smoothing": hp.Float("label_smoothing", 0.0, 0.05, step=0.01),
    }

    x = keras.layers.Dense(
        hp_vals["units1"],
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(hp_vals["l2"]),
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(hp_vals["dropout"])(x)

    x = keras.layers.Dense(
        hp_vals["units2"],
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(hp_vals["l2"]),
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(hp_vals["dropout"])(x)

    outputs = keras.layers.Dense(1, activation="sigmoid", name="prob")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(hp_vals["lr"]),
        loss=keras.losses.BinaryCrossentropy(
            label_smoothing=hp_vals["label_smoothing"]
        ),
        metrics=make_metrics(),
    )
    return model


def tuner_fn(fn_args):
    """Entry point TFX Tuner: siapkan BayesianOptimization dan argumen fit."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_ds, eval_ds = make_datasets(fn_args, tf_transform_output, batch_size=256)

    tuner = kt.BayesianOptimization(
        lambda hp: _build_model(hp, tf_transform_output),
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=25,
        num_initial_points=8,
        alpha=1e-4,
        beta=2.6,
        overwrite=True,
        directory=fn_args.working_dir,
        project_name="heart_tuning_bayes",
        seed=42,
    )

    fit_kwargs = {
        "x": train_ds,
        "validation_data": eval_ds,
        "steps_per_epoch": int(fn_args.train_steps),
        "validation_steps": int(fn_args.eval_steps),
        "epochs": 20,
        "callbacks": [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=3,
                restore_best_weights=True,
            )
        ],
        "verbose": 1,
    }

    return SimpleNamespace(tuner=tuner, fit_kwargs=fit_kwargs)
