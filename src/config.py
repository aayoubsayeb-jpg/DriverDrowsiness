"""Loads configs/config.yaml. Importing this module does NOT require TensorFlow."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: ROOT / v for k, v in cfg["paths"].items()}
    return cfg


CFG = load_config()
CLASSES = CFG["classes"]          # ["Closed", "Open", "no_yawn", "yawn"]
IMG_SIZE = CFG["img_size"]
SEED = CFG["seed"]

# Which predicted classes count as a drowsiness sign in the demo.
DROWSY_CLASSES = {"Closed", "yawn"}

# Which raw folder each class lives in (see scripts/organize_raw.py).
RAW_GROUP = {"Closed": "eyes", "Open": "eyes", "no_yawn": "yawn", "yawn": "yawn"}
# Yawn classes are full-face images -> face crop (Haar cascade) during preparation.
FACE_CROP_CLASSES = {"no_yawn", "yawn"}
