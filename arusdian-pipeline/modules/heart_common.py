"""Common defs untuk modul HeartDisease TFX (dipakai trainer/tuner/transform)."""

from typing import List, Tuple, Dict
import tensorflow as tf
from tensorflow import keras
from tfx_bsl.public import tfxio

# ==== Konstanta fitur ====
LABEL_KEY: str = "HeartDisease"

NUMERIC_FEATURE_KEYS: List[str] = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak",
]

CATEGORICAL_FEATURE_KEYS: List[str] = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]


def transformed_name(key: str) -> str:
    """Tambahkan sufiks '_xf' pada nama fitur yang sudah di-transform."""
    return key + "_xf"


def make_metrics() -> List[keras.metrics.Metric]:
    """Daftar metrik standar untuk klasifikasi biner."""
    return [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ]


def input_fn(file_pattern, data_accessor, tf_transform_output, batch_size: int = 256):
    """Bangun tf.data.Dataset dari artefak Transform."""
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=transformed_name(LABEL_KEY)
        ),
        tf_transform_output.transformed_metadata.schema,
    )


def make_datasets(fn_args, tf_transform_output, batch_size: int = 256):
    """Buat train_ds dan eval_ds dari fn_args."""
    train_ds = input_fn(
        fn_args.train_files, fn_args.data_accessor, tf_transform_output, batch_size
    )
    eval_ds = input_fn(
        fn_args.eval_files, fn_args.data_accessor, tf_transform_output, batch_size
    )
    return train_ds, eval_ds


def vocab_size_from_tft(tf_transform_output, vocab_name: str) -> int:
    """Ambil ukuran vocabulary dari TFT (+1 untuk OOV)."""
    path = tf_transform_output.vocabulary_file_by_name(vocab_name)
    count = 0
    with tf.io.gfile.GFile(path, "r") as f:
        for _ in f:
            count += 1
    return count + 1  # OOV = 1


def build_feature_block(
    tf_transform_output,
) -> Tuple[Dict[str, keras.Input], tf.Tensor, Dict[str, int]]:
    """Bangun blok input + encoding fitur (numeric + categorical one-hot).

    Returns:
        inputs: dict nama -> keras.Input
        x: tensor gabungan fitur siap masuk MLP
        cat_dims: peta fitur kategori -> ukuran vocab (+OOV)
    """
    # Inputs numeric
    inputs: Dict[str, keras.Input] = {}
    for key in NUMERIC_FEATURE_KEYS:
        name = transformed_name(key)
        inputs[name] = keras.Input(shape=(1,), dtype=tf.float32, name=name)

    # Inputs categorical + vocab dims
    cat_dims: Dict[str, int] = {}
    for key in CATEGORICAL_FEATURE_KEYS:
        name = transformed_name(key)
        inputs[name] = keras.Input(shape=(1,), dtype=tf.int32, name=name)
        cat_dims[key] = vocab_size_from_tft(tf_transform_output, key)

    # Numeric pipeline
    x_num = keras.layers.BatchNormalization(name="bn_numeric")(
        keras.layers.Concatenate(name="numeric_concat")(
            [inputs[transformed_name(k)] for k in NUMERIC_FEATURE_KEYS]
        )
    )

    # Categorical pipeline (one-hot + drop-first)
    cat_onehots = []
    for key in CATEGORICAL_FEATURE_KEYS:
        enc = keras.layers.CategoryEncoding(
            num_tokens=cat_dims[key],
            output_mode="one_hot",
            name=f"onehot_{key}",
        )(inputs[transformed_name(key)])
        enc = keras.layers.Reshape((cat_dims[key],))(enc)
        if cat_dims[key] > 1:
            enc = enc[:, 1:]  # drop-first
        cat_onehots.append(enc)

    x = x_num
    if cat_onehots:
        x = keras.layers.Concatenate(name="features_concat")(
            [x_num, keras.layers.Concatenate(name="cat_concat")(cat_onehots)]
        )

    return inputs, x, cat_dims
