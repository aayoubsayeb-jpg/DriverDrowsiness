"""Inference: Haar face detection -> backbone classifier -> 4-class probabilities."""
import cv2
import numpy as np
from tensorflow import keras

from .config import CFG, CLASSES, DROWSY_CLASSES, IMG_SIZE
from .models import get_preprocess


class DrowsinessPredictor:
    def __init__(self, model_name="mobilenet_v2", model_path=None):
        path = model_path or CFG["paths"]["models_dir"] / f"{model_name}.keras"
        self.model = keras.models.load_model(path)
        self.preprocess = get_preprocess(model_name)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect_face(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])   # largest face

    def predict(self, rgb):
        """rgb: HxWx3 uint8 (RGB). Returns (annotated_image, {class: prob}, status) or (image, None, msg)."""
        box = self.detect_face(rgb)
        if box is None:
            return rgb, None, "No face detected"
        x, y, w, h = [int(v) for v in box]
        face = cv2.resize(rgb[y:y + h, x:x + w], (IMG_SIZE, IMG_SIZE)).astype("float32")
        probs = self.model.predict(self.preprocess(face[None]), verbose=0)[0]
        label = CLASSES[int(np.argmax(probs))]
        drowsy = label in DROWSY_CLASSES

        out = rgb.copy()
        color = (255, 0, 0) if drowsy else (0, 200, 0)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, f"{label} {probs.max() * 100:.0f}%", (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        status = f"Drowsiness sign: {label}" if drowsy else f"Alert ({label})"
        return out, {c: float(p) for c, p in zip(CLASSES, probs)}, status
