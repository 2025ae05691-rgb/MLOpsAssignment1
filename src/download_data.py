from io import StringIO
from pathlib import Path

import pandas as pd
import requests

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "heart_disease.csv"

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]


def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(
        StringIO(response.text),
        header=None,
        names=COLUMN_NAMES,
        na_values="?",
    )

    if df.shape[0] != 303 or df.shape[1] != 14:
        raise ValueError(
            f"Unexpected dataset shape: {df.shape}. "
            "Verify the source and dataset version."
        )

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Dataset saved to: {OUTPUT_FILE}")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst five rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isna().sum())


if __name__ == "__main__":
    download_dataset()