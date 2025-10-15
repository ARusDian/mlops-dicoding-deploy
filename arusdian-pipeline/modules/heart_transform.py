"""Transform TFX untuk dataset HeartDisease.

Mendefinisikan preprocessing_fn:
- Numeric: standardization (z-score)
- Categorical: integer index via compute_and_apply_vocabulary (OOV bucket = 1)
- Label: cast ke float32; nama output mengikuti konvensi *_xf
"""

from typing import Mapping, Dict
import tensorflow as tf
import tensorflow_transform as tft
from .heart_common import (
    LABEL_KEY,
    NUMERIC_FEATURE_KEYS,
    CATEGORICAL_FEATURE_KEYS,
    transformed_name,
)


def _transform_numeric(features: Mapping[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Transform untuk fitur numerik (z-score)."""
    outputs: Dict[str, tf.Tensor] = {}
    for key in NUMERIC_FEATURE_KEYS:
        x = tf.cast(features[key], tf.float32)
        outputs[transformed_name(key)] = tft.scale_to_z_score(x)
    return outputs


def _transform_categorical(
    features: Mapping[str, tf.Tensor],
) -> Dict[str, tf.Tensor]:
    """Transform untuk fitur kategori -> integer id (mulai dari 0, OOV=last)."""
    outputs: Dict[str, tf.Tensor] = {}
    for key in CATEGORICAL_FEATURE_KEYS:
        x = tf.cast(features[key], tf.string)
        idx = tft.compute_and_apply_vocabulary(
            x,
            num_oov_buckets=1,
            vocab_filename=key,
        )
        outputs[transformed_name(key)] = tf.cast(idx, tf.int32)
    return outputs


def _transform_label(features: Mapping[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Transform label ke float32 (0/1)."""
    y = tf.cast(features[LABEL_KEY], tf.float32)
    y = tf.reshape(y, [-1])
    return {transformed_name(LABEL_KEY): y}


def preprocessing_fn(inputs: Mapping[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Mendefinisikan transformasi fitur untuk komponen TFX Transform."""
    outputs: Dict[str, tf.Tensor] = {}
    outputs.update(_transform_numeric(inputs))
    outputs.update(_transform_categorical(inputs))
    outputs.update(_transform_label(inputs))
    return outputs
