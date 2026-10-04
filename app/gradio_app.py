"""Gradio demo.   python app/gradio_app.py      (run from the repository root)

Environment variables:
    MODEL_NAME   mobilenet_v2 (default) | vgg16 | resnet50 | densenet121
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402

from src.predict import DrowsinessPredictor  # noqa: E402

predictor = DrowsinessPredictor(os.environ.get("MODEL_NAME", "mobilenet_v2"))


def run(image):
    if image is None:
        return None, None, "Upload an image"
    annotated, probs, status = predictor.predict(image)
    return annotated, probs, status


demo = gr.Interface(
    fn=run,
    inputs=gr.Image(type="numpy", label="Driver image"),
    outputs=[gr.Image(label="Detection"), gr.Label(num_top_classes=4, label="Class probabilities"),
             gr.Textbox(label="Status")],
    title="Driver Drowsiness Detection",
    description="Detects drowsiness signs (closed eyes, yawning) with a fine-tuned CNN. "
                "Single-image demo - not a certified safety system.",
)

if __name__ == "__main__":
    demo.launch()
