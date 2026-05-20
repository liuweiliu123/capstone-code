from pathlib import Path
from typing import List, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "Class/ASD Traits"
LEAKAGE_COLUMNS = ["Case_No", "Qchat-10-Score"] + [f"A{i}" for i in range(1, 11)]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def default_csv_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "Toddler Autism dataset July 2018.csv"


def read_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place the CSV file in the data folder or pass --csv-path."
        )

    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    y = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )
    if y.isna().any():
        invalid_values = sorted(df.loc[y.isna(), TARGET_COLUMN].astype(str).unique())
        raise ValueError(f"Unrecognized target labels found: {invalid_values}")

    x = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    x = x.drop(columns=LEAKAGE_COLUMNS, errors="ignore")
    return x, y.astype(int)


def prepare_one_hot_data(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    x, y = read_dataset(csv_path)

    for column in x.columns:
        if x[column].dtype == "object":
            x[column] = x[column].astype(str).str.strip()

    x = pd.get_dummies(x, drop_first=False)
    x = x.fillna(0)
    return x.astype(float), y


def prepare_catboost_data(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, List[int]]:
    x, y = read_dataset(csv_path)

    for column in x.columns:
        series = x[column]
        if is_string_dtype(series) or series.dtype == "object":
            stripped = series.astype(str).str.strip()
            numeric = pd.to_numeric(stripped, errors="coerce")
            if numeric.notna().any():
                median = numeric.median()
                x[column] = numeric.fillna(0 if pd.isna(median) else median)
            else:
                x[column] = stripped.fillna("missing")
        elif is_numeric_dtype(series):
            median = series.median()
            x[column] = series.fillna(0 if pd.isna(median) else median)
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                median = numeric.median()
                x[column] = numeric.fillna(0 if pd.isna(median) else median)
            else:
                x[column] = series.astype(str).fillna("missing")

    cat_features = [
        index
        for index, column in enumerate(x.columns)
        if is_string_dtype(x[column]) or x[column].dtype == "object"
    ]
    return x, y, cat_features


def split_data(x: pd.DataFrame, y: pd.Series):
    return train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def print_metrics(model_name: str, y_test, y_pred, train_size: int, test_size: int) -> None:
    print(f"=== {model_name} ===")
    print(f"Train/Test split: {train_size} / {test_size} (80/20)")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, digits=4))

