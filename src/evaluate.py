"""Evaluate trained model(s) on the held-out TEST split.

    python -m src.evaluate --model vgg16
    python -m src.evaluate --model all      # also writes reports/results.csv
"""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix  # noqa: E402
from tensorflow import keras  # noqa: E402

from .config import CFG, CLASSES  # noqa: E402
from .data import get_datasets  # noqa: E402
from .models import SPECS  # noqa: E402


def evaluate(name):
    _, _, test_ds = get_datasets(name)
    model = keras.models.load_model(CFG["paths"]["models_dir"] / f"{name}.keras")

    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    # target_names comes from CLASSES -> same order as training labels.
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

    reports = CFG["paths"]["reports_dir"]
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    (reports / f"{name}_report.json").write_text(json.dumps(report, indent=2))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap=plt.cm.Blues, values_format="d")
    plt.title(f"Confusion matrix - {name} (test set)")
    plt.tight_layout()
    plt.savefig(reports / "figures" / f"{name}_confusion_matrix.png", dpi=150)
    plt.close()
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SPECS) + ["all"])
    args = ap.parse_args()
    names = list(SPECS) if args.model == "all" else [args.model]
    rows = []
    for n in names:
        print(f"\n===== {n} =====")
        r = evaluate(n)
        rows.append([n, round(r["accuracy"], 4), round(r["macro avg"]["f1-score"], 4),
                     round(r["weighted avg"]["f1-score"], 4)])
    if len(rows) > 1:
        with open(CFG["paths"]["reports_dir"] / "results.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "accuracy", "macro_f1", "weighted_f1"])
            w.writerows(rows)
        print("Saved reports/results.csv")
