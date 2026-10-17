
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "heart_disease_model.joblib"
DATA_FILE = ROOT / "data" / "processed" / "heart_disease_cleaned.csv"

NUMERIC_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def main():
    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(DATA_FILE)

    sample = df[FEATURES].iloc[[0]]

    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]

    print("Model loaded successfully.")
    print("Input feature count:", len(FEATURES))
    print("Predicted class:", prediction)
    print("Probability for class 0:", round(float(probabilities[0]), 4))
    print("Probability for class 1:", round(float(probabilities[1]), 4))

    assert prediction in (0, 1)
    assert len(probabilities) == 2
    assert abs(float(probabilities.sum()) - 1.0) < 1e-6

    print("PASS: Model prediction and probability checks succeeded.")


if __name__ == "__main__":
    main()
