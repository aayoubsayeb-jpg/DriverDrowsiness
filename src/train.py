"""Train one (or all) backbones.

    python -m src.train --model mobilenet_v2
    python -m src.train --model all --epochs 10
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

from .config import CFG, SEED  # noqa: E402
from .data import get_datasets  # noqa: E402
from .models import SPECS, build_model  # noqa: E402


def plot_history(history, name, out_dir):
    for metric in ("loss", "accuracy"):
        plt.figure(figsize=(7, 4))
        plt.plot(history[metric], marker="o", label="train")
        plt.plot(history["val_" + metric], marker="o", label="validation")
        plt.title(f"{name} - {metric}")
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_{metric}.png", dpi=150)
        plt.close()


def train(name, epochs=None):
    keras.utils.set_random_seed(SEED)
    tcfg = CFG["train"]
    train_ds, val_ds, _ = get_datasets(name)
    model = build_model(name)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=tcfg["early_stopping_patience"],
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=tcfg["lr_factor"],
                                          patience=tcfg["lr_patience"], min_lr=tcfg["min_lr"], verbose=1),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs or tcfg["epochs"],
                        callbacks=callbacks).history

    models_dir = CFG["paths"]["models_dir"]
    fig_dir = CFG["paths"]["reports_dir"] / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / f"{name}.keras")
    (CFG["paths"]["reports_dir"] / f"{name}_history.json").write_text(
        json.dumps({k: [float(v) for v in vals] for k, vals in history.items()}, indent=2))
    plot_history(history, name, fig_dir)
    print(f"Saved models/{name}.keras")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SPECS) + ["all"])
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    for n in (SPECS if args.model == "all" else [args.model]):
        print(f"\n===== Training {n} =====")
        train(n, args.epochs)
