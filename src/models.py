"""The four transfer-learning backbones compared in this project."""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import (DenseNet121, MobileNetV2, ResNet50, VGG16,
                                           densenet, mobilenet_v2, resnet50, vgg16)

from .config import CFG, CLASSES, IMG_SIZE

# trainable=True  -> full fine-tuning;  trainable=False -> frozen backbone (feature extractor)
# Hyper-parameters are the ones used in the original Colab experiments.
SPECS = {
    "mobilenet_v2": dict(base=MobileNetV2, preprocess=mobilenet_v2.preprocess_input,
                         trainable=True, dense=None, dropout=0.5),
    "resnet50": dict(base=ResNet50, preprocess=resnet50.preprocess_input,
                     trainable=False, dense=256, dropout=0.1),
    "vgg16": dict(base=VGG16, preprocess=vgg16.preprocess_input,
                  trainable=True, dense=None, dropout=0.5),
    "densenet121": dict(base=DenseNet121, preprocess=densenet.preprocess_input,
                        trainable=False, dense=256, dropout=0.1),
}


def get_preprocess(name):
    """Each ImageNet backbone expects its OWN input scaling (not a plain /255)."""
    return SPECS[name]["preprocess"]


def build_model(name, learning_rate=None):
    spec = SPECS[name]
    base = spec["base"](weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = spec["trainable"]

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs) if spec["trainable"] else base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    if spec["dense"]:
        x = layers.Dense(spec["dense"], activation="relu")(x)
    x = layers.Dropout(spec["dropout"])(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)

    model = keras.Model(inputs, outputs, name=name)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate or CFG["train"]["learning_rate"]),
        metrics=["accuracy"],
    )
    return model
