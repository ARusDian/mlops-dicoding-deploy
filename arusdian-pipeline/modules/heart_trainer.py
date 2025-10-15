"""Trainer TFX untuk model klasifikasi penyakit jantung (HeartDisease)."""

import json
from typing import Any, Dict, Optional

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow import keras

from .heart_common import (
    build_feature_block,
    make_datasets,
    make_metrics,
    LABEL_KEY,
)


def _get_hparam(hparams: Any, name: str, default: Any) -> Any:
    """Ambil hparam dari Tuner bila ada; selain itu pakai default."""
    if hparams is None:
        return default
    if isinstance(hparams, dict):
        return hparams.get(name, default)
    try:
        return hparams.get(name)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        return default


def _build_model(tf_transform_output, hparams: Optional[Dict[str, Any]] = None):
    """Bangun model tabular dengan one-hot drop-first + regularisasi."""
    hp = {
        "units1": int(_get_hparam(hparams, "units1", 64)),
        "units2": int(_get_hparam(hparams, "units2", 32)),
        "dropout": float(_get_hparam(hparams, "dropout", 0.5)),
        "l2": float(_get_hparam(hparams, "l2", 1e-4)),
        "lr": float(_get_hparam(hparams, "lr", 5e-4)),
        "label_smoothing": float(_get_hparam(hparams, "label_smoothing", 0.03)),
    }

    inputs, x, _ = build_feature_block(tf_transform_output)

    x = keras.layers.Dense(
        hp["units1"],
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(hp["l2"]),
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(hp["dropout"])(x)

    x = keras.layers.Dense(
        hp["units2"],
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(hp["l2"]),
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(hp["dropout"])(x)

    outputs = keras.layers.Dense(1, activation="sigmoid", name="prob")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(hp["lr"]),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=hp["label_smoothing"]),
        metrics=make_metrics(),
    )
    return model


def _parse_hparams(fn_args) -> Optional[Dict[str, Any]]:
    """Parse hyperparameters dari fn_args; kembalikan dict bila ada."""
    cfg = getattr(fn_args, "hyperparameters", None)
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg
    try:
        return json.loads(cfg)
    except (json.JSONDecodeError, TypeError):
        return None


def _make_callbacks():
    """Buat daftar callback pelatihan."""
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc", mode="max", factor=0.5, patience=2, verbose=1
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )
    return [reduce_lr, early_stop]


# pylint: disable=abstract-method
class ServingModule(tf.Module):
    """Bungkus model & TFT layer agar resources ter-track saat export."""

    def __init__(self, model: tf.keras.Model, tf_transform_output, label_key: str):
        super().__init__()
        self.model = model
        self.tft_layer = tf_transform_output.transform_features_layer()
        spec = tf_transform_output.raw_feature_spec()
        spec.pop(label_key, None)
        self.raw_feature_spec = spec

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")]
    )
    def serve(self, serialized_tf_examples):
        """Signature serving_default: menerima tf.Example ter-serialize."""
        raw = tf.io.parse_example(serialized_tf_examples, self.raw_feature_spec)
        transformed = self.tft_layer(raw)
        probs = self.model(transformed)
        return {"outputs": probs}


def run_fn(fn_args):
    """Entry point untuk komponen TFX Trainer."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    hparams = _parse_hparams(fn_args)

    train_ds, eval_ds = make_datasets(fn_args, tf_transform_output, batch_size=256)

    model = _build_model(tf_transform_output, hparams=hparams)
    callbacks = _make_callbacks()

    model.fit(
        train_ds,
        validation_data=eval_ds,
        epochs=40,
        steps_per_epoch=int(fn_args.train_steps),
        validation_steps=int(fn_args.eval_steps),
        callbacks=callbacks,
        verbose=1,
    )

    module = ServingModule(model, tf_transform_output, LABEL_KEY)
    tf.saved_model.save(
        module,
        fn_args.serving_model_dir,
        signatures={"serving_default": module.serve},
    )
