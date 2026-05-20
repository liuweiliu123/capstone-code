import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


class EmotionPredictor:
    def __init__(self, model_path: str, class_names_path: str, img_size: int = 224):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names = self._load_class_names(class_names_path)
        self.num_classes = len(self.class_names)
        self.img_size = img_size

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    @staticmethod
    def _load_class_names(path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def predict_pil(self, image: Image.Image) -> Tuple[str, float, np.ndarray]:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

        class_index = int(np.argmax(probabilities))
        return self.class_names[class_index], float(probabilities[class_index]), probabilities

    def detect_faces(self, bgr_frame: np.ndarray):
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        return self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60),
        )

    def annotate_frame(self, bgr_frame: np.ndarray):
        faces = self.detect_faces(bgr_frame)
        overlay = bgr_frame.copy()
        frame_height, frame_width = bgr_frame.shape[:2]

        if len(faces) == 0:
            rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            return rgb, []

        results = []
        for (x, y, width, height) in faces:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame_width, x + width)
            y2 = min(frame_height, y + height)
            face = overlay[y1:y2, x1:x2]
            if face.size == 0:
                continue

            pil_face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            label, confidence, _ = self.predict_pil(pil_face)
            text = f"{label} ({confidence:.2f})"

            results.append(
                {
                    "emotion": label,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                }
            )

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                overlay,
                text,
                (x1, max(18, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        return rgb, results

    def predict_video(self, input_path: str, output_path: str, sample_every_n_frames: int = 3):
        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            raise RuntimeError("Unable to read the uploaded video.")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % sample_every_n_frames == 0:
                faces = self.detect_faces(frame)
                for (x, y, face_width, face_height) in faces:
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(width, x + face_width)
                    y2 = min(height, y + face_height)
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue

                    pil_face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    label, confidence, _ = self.predict_pil(pil_face)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label} ({confidence:.2f})",
                        (x1, max(18, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            writer.write(frame)
            frame_index += 1

        capture.release()
        writer.release()
        return str(Path(output_path).resolve())

