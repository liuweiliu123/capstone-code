# Autism Emotion Recognition Web Application

This folder contains the source code for the facial emotion recognition web application.

The implementation uses:

- PyTorch and Torchvision for model training.
- ResNet-18 as the image classification backbone.
- OpenCV Haar cascade detection for locating faces.
- Gradio for the interactive web interface.

## Expected Dataset Structure

Place the image dataset outside this folder, for example:

```text
emotion_dataset/
|-- train/
|   |-- anger/
|   |-- fear/
|   |-- joy/
|   |-- Natural/
|   |-- sadness/
|   `-- surprise/
`-- test/
    |-- anger/
    |-- fear/
    |-- joy/
    |-- Natural/
    |-- sadness/
    `-- surprise/
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train.py --data-dir "../emotion_dataset" --epochs 10
```

The training script saves:

```text
artifacts/best_model.pt
artifacts/class_names.json
artifacts/history.json
```

## Launch the Web App

```bash
python app.py
```

Open the local URL shown in the terminal. The default URL is:

```text
http://127.0.0.1:7860
```

## Supported Inputs

- Webcam frames
- Uploaded images
- Uploaded videos

