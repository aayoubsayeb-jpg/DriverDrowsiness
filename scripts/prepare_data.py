"""Build data/processed/{train,val,test}/<class>/ from data/raw/.

Steps
 1. Remove duplicate images (same file content) -> prevents train/test leakage.
 2. Stratified split per class (80/10/10 by default, fixed seed).
 3. For yawn / no_yawn: detect the face with a Haar cascade and crop it. If no face is
    found the whole image is kept (--no-fallback drops it instead). The raw yawn images
    are already mostly tight face crops, so the detector rarely fires. Their swapped
    R/B channels are fixed (config: fix_yawn_color).
 4. Resize to IMG_SIZE x IMG_SIZE and save as JPEG.

Usage:
    python scripts/prepare_data.py
"""
import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CFG, CLASSES, FACE_CROP_CLASSES, IMG_SIZE, RAW_GROUP, SEED  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png"}


def list_unique_images(cls_dir):
    """Sorted list of images in cls_dir with duplicate contents removed."""
    seen, unique, dups = set(), [], 0
    for p in sorted(cls_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen:
            dups += 1
            continue
        seen.add(h)
        unique.append((p, h))
    return unique, dups


def crop_largest_face(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return img[y:y + h, x:x + w]


def split_indices(n_items, ratios, seed):
    idx = list(range(n_items))
    train_r, val_r, test_r = ratios
    tr, rest = train_test_split(idx, test_size=val_r + test_r, random_state=seed)
    val, te = train_test_split(rest, test_size=test_r / (val_r + test_r), random_state=seed)
    return {"train": tr, "val": val, "test": te}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fallback", action="store_true",
                    help="drop yawn images where no face is detected (instead of keeping the full image)")
    args = ap.parse_args()

    raw, out = CFG["paths"]["raw_dir"], CFG["paths"]["processed_dir"]
    ratios = (CFG["split"]["train"], CFG["split"]["val"], CFG["split"]["test"])
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for split in ("train", "val", "test"):
        shutil.rmtree(out / split, ignore_errors=True)

    summary = {"duplicates_removed": {}, "face_detected": {}, "face_missing": {}, "counts": {}}
    for cls in CLASSES:
        cls_dir = raw / RAW_GROUP[cls] / cls
        if not cls_dir.is_dir():
            sys.exit(f"Missing {cls_dir}. Run scripts/organize_raw.py first.")
        items, dups = list_unique_images(cls_dir)
        summary["duplicates_removed"][cls] = dups
        parts = split_indices(len(items), ratios, SEED)  # split per class == stratified
        found = missing = 0
        counts = Counter()
        for split, ids in parts.items():
            dst = out / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for i in ids:
                path, h = items[i]
                img = cv2.imread(str(path))
                if img is None:
                    continue
                if cls in FACE_CROP_CLASSES:
                    face = crop_largest_face(img, cascade)
                    if face is None:
                        missing += 1
                        if args.no_fallback:
                            continue
                    else:
                        found += 1
                        img = face
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                if cls in FACE_CROP_CLASSES and CFG.get("fix_yawn_color", False):
                    img = img[:, :, ::-1]   # swap B<->R back (raw yawn images are blue-tinted)
                cv2.imwrite(str(dst / f"{h[:12]}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                counts[split] += 1
        summary["face_detected"][cls] = found
        summary["face_missing"][cls] = missing
        summary["counts"][cls] = dict(counts)
        print(f"{cls:8s} unique={len(items):4d} dups_removed={dups:3d} "
              f"splits={dict(counts)} faces_found={found} faces_missing={missing}")

    (out / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved", out / "split_summary.json")


if __name__ == "__main__":
    main()
