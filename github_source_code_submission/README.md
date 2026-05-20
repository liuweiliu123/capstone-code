# Autism Screening and Emotion Recognition Source Code

This repository provides the source code needed to verify the implementation. It contains two implementation components:

1. Toddler autism trait classification using traditional machine learning models.
2. Facial emotion recognition using a PyTorch ResNet-18 model and a Gradio web application.

The repository is intended for reviewer inspection and reproducibility.

## Repository Structure

```text
.
|-- README.md
|-- REVIEWER_RESPONSE.md
|-- requirements.txt
|-- data/
|   |-- README.md
|   `-- Toddler Autism dataset July 2018.csv
|-- toddler_autism_classification/
|   |-- common.py
|   |-- logistic_regression_toddler.py
|   |-- random_forest_toddler.py
|   |-- xgboost_toddler.py
|   |-- catboost_toddler.py
|   `-- run_all_models.py
`-- emotion_webapp/
    |-- README.md
    |-- requirements.txt
    |-- app.py
    |-- predictor.py
    |-- train.py
    `-- artifacts/
        `-- .gitkeep
```

## Component 1: Toddler Autism Trait Classification

The folder `toddler_autism_classification/` contains four supervised machine learning implementations:

- Logistic Regression
- Random Forest
- XGBoost
- CatBoost

Each implementation loads the toddler autism dataset, cleans the target label, removes direct leakage columns, applies preprocessing, performs an 80/20 stratified train-test split, trains the model, and prints accuracy plus a classification report.

The removed leakage columns are:

```text
Case_No, Qchat-10-Score, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10
```

These columns are excluded because they can directly encode or strongly leak the target label.

### Install Dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

### Run All Classification Models

```bash
python toddler_autism_classification/run_all_models.py
```

The runner writes individual model logs and a summary CSV to:

```text
model_outputs/
```

### Run Individual Classification Models

```bash
python toddler_autism_classification/logistic_regression_toddler.py
python toddler_autism_classification/random_forest_toddler.py
python toddler_autism_classification/xgboost_toddler.py
python toddler_autism_classification/catboost_toddler.py
```

If the CSV file is stored in a different location, pass it explicitly:

```bash
python toddler_autism_classification/run_all_models.py --csv-path "path/to/Toddler Autism dataset July 2018.csv"
```

## Component 2: Facial Emotion Recognition Web Application

The folder `emotion_webapp/` contains a PyTorch and Gradio implementation for facial emotion recognition. It supports:

- Real-time webcam inference.
- Uploaded image inference.
- Uploaded video inference.

The model uses a ResNet-18 backbone. OpenCV Haar cascade face detection is used to locate faces before emotion classification.

### Expected Emotion Dataset Structure

The emotion dataset should be organized as:

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

### Train the Emotion Recognition Model

From the `emotion_webapp/` directory:

```bash
pip install -r requirements.txt
python train.py --data-dir "../emotion_dataset" --epochs 10
```

Training produces:

```text
emotion_webapp/artifacts/best_model.pt
emotion_webapp/artifacts/class_names.json
emotion_webapp/artifacts/history.json
```

### Launch the Web Application

From the `emotion_webapp/` directory:

```bash
python app.py
```

The application runs locally at:

```text
http://127.0.0.1:7860
```

## Reproducibility Notes

- The classification scripts use `random_state=42` where applicable.
- The classification experiments use an 80/20 stratified train-test split.
- The emotion recognition training script saves the best checkpoint based on test accuracy.
- If pretrained ResNet-18 weights cannot be downloaded, the training script falls back to random initialization.
- Generated outputs, Python caches, and trained model weights are excluded from version control by `.gitignore`.

## Source Code Availability Statement

All source code required to inspect and verify the implementation is included in this repository. The repository contains model training scripts, preprocessing logic, the inference module, the web application, dependency files, dataset placement instructions, and reproducibility commands.

