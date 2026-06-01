import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from predictor import EmotionPredictor


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.pt"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"

_PREDICTOR = None


def get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
            raise FileNotFoundError(
                "Trained model files were not found. Run: "
                "python train.py --data-dir \"../emotion_dataset\""
            )
        _PREDICTOR = EmotionPredictor(str(MODEL_PATH), str(CLASS_NAMES_PATH))
    return _PREDICTOR


def webcam_infer(frame: np.ndarray):
    if frame is None:
        return None, "No webcam frame detected."

    predictor = get_predictor()
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    rgb_annotated, results = predictor.annotate_frame(bgr)
    if not results:
        return rgb_annotated, "No face detected."

    top_result = sorted(results, key=lambda item: item["confidence"], reverse=True)[0]
    return rgb_annotated, f"Primary emotion: {top_result['emotion']} (confidence {top_result['confidence']:.2%})"


def image_infer(image: Image.Image):
    if image is None:
        return None, "Please upload an image."

    predictor = get_predictor()
    bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    rgb_annotated, results = predictor.annotate_frame(bgr)
    if not results:
        return rgb_annotated, "No face detected."

    lines = [
        f"{index + 1}. {result['emotion']} ({result['confidence']:.2%})"
        for index, result in enumerate(results)
    ]
    return rgb_annotated, "Detection results:\n" + "\n".join(lines)


def video_infer(video_path: str):
    if not video_path:
        return None, "Please upload a video."

    predictor = get_predictor()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temporary_file:
        output_path = temporary_file.name

    output_video = predictor.predict_video(video_path, output_path)
    return output_video, "Video analysis completed."


with gr.Blocks(title="Autism Emotion Recognition") as demo:
    gr.Markdown("# Autism Emotion Recognition Demo")
    gr.Markdown(
        "Supports real-time webcam, uploaded image, and uploaded video analysis. "
        "Predictions use the trained model saved in artifacts/best_model.pt."
    )

    with gr.Tab("Real-time Webcam"):
        webcam_input = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam")
        webcam_output_image = gr.Image(type="numpy", label="Annotated Output")
        webcam_output_text = gr.Textbox(label="Recognition Info")
        webcam_input.stream(
            webcam_infer,
            inputs=webcam_input,
            outputs=[webcam_output_image, webcam_output_text],
        )

    with gr.Tab("Upload Image"):
        image_input = gr.Image(type="pil", label="Upload Image")
        image_button = gr.Button("Analyze Image")
        image_output = gr.Image(type="numpy", label="Annotated Output")
        image_text = gr.Textbox(label="Recognition Info")
        image_button.click(image_infer, inputs=image_input, outputs=[image_output, image_text])

    with gr.Tab("Upload Video"):
        video_input = gr.Video(label="Upload Video")
        video_button = gr.Button("Analyze Video")
        video_output = gr.Video(label="Analyzed Video")
        video_text = gr.Textbox(label="Processing Status")
        video_button.click(video_infer, inputs=video_input, outputs=[video_output, video_text])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

