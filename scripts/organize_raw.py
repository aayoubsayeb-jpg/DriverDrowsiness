"""Copy the ORIGINAL dataset layout into data/raw/ (eyes/ and yawn/ separated).

Original layout:   <src>/Closed, <src>/Open, <src>/yawn, <src>/no_yawn
New layout:        data/raw/eyes/{Closed,Open}   data/raw/yawn/{no_yawn,yawn}

Usage:
    python scripts/organize_raw.py --src "/path/to/drowsiness-dataset/train"
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CFG, CLASSES, RAW_GROUP  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing Closed/Open/yawn/no_yawn")
    args = ap.parse_args()
    src = Path(args.src)
    raw = CFG["paths"]["raw_dir"]
    for cls in CLASSES:
        s = src / cls
        if not s.is_dir():
            sys.exit(f"Missing class folder: {s}")
        d = raw / RAW_GROUP[cls] / cls
        d.mkdir(parents=True, exist_ok=True)
        files = [p for p in s.iterdir() if p.suffix.lower() in IMG_EXT]
        for p in files:
            shutil.copy2(p, d / p.name)
        print(f"{cls:8s}: {len(files):5d} images -> {d}")


if __name__ == "__main__":
    main()
