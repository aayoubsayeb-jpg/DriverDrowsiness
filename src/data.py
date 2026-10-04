"""tf.data input pipelines reading data/processed/{train,val,test}/<class>/*.jpg"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import CFG, CLASSES, IMG_SIZE
from .models import get_preprocess

AUTOTUNE = tf.data.AUTOTUNE


def _load(split, shuffle, batch_size):
    return keras.utils.image_dataset_from_directory(
        str(CFG["paths"]["processed_dir"] / split),
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,              # fixes the label order explicitly
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=CFG["seed"] if shuffle else None,
    )


def get_datasets(model_name, batch_size=None):
    """Returns (train, val, test). Augmentation is applied to train only."""
    bs = batch_size or CFG["batch_size"]
    preprocess = get_preprocess(model_name)
    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(30 / 360),   # +-30 degrees
        layers.RandomZoom(0.2),
    ])

    train = (_load("train", True, bs)
             .map(lambda x, y: (preprocess(augment(x, training=True)), y), num_parallel_calls=AUTOTUNE)
             .prefetch(AUTOTUNE))
    val = _load("val", False, bs).map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test = _load("test", False, bs).map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train, val, test
