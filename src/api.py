
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.monitoring import configure_monitoring

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "heart_disease_model.joblib"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal",
]

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
)
configure_monitoring(app)

model = joblib.load(MODEL_PATH)


class PatientInput(BaseModel):
    age: int = Field(ge=1, le=120)
    sex: int
    cp: int
    trestbps: float = Field(gt=0)
    chol: float = Field(ge=0)
    fbs: int
    restecg: int
    thalach: float = Field(gt=0)
    exang: int
    oldpeak: float
    slope: int
    ca: float = Field(ge=0)
    thal: float


class PatientBatch(BaseModel):
    patients: list[PatientInput] = Field(min_length=1)


def predict_features(features: pd.DataFrame) -> list[dict[str, object]]:
    predictions = model.predict(features)
    results = [
        {"prediction": int(prediction)}
        for prediction in predictions
    ]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        for result, row in zip(results, probabilities):
            result["probabilities"] = {
                str(label): round(float(probability), 4)
                for label, probability in zip(model.classes_, row)
            }

    return results


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    return {
        "api_version": app.version,
        "model_type": type(model).__name__,
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "classes": [
            str(label) for label in getattr(model, "classes_", [])
        ],
    }


@app.post("/predict")
def predict(patient: PatientInput):
    try:
        features = pd.DataFrame(
            [patient.model_dump()],
            columns=FEATURES,
        )
        return predict_features(features)[0]

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid prediction input: {exc}",
        ) from exc


@app.post("/predict/batch")
def predict_batch(batch: PatientBatch):
    try:
        features = pd.DataFrame(
            [patient.model_dump() for patient in batch.patients],
            columns=FEATURES,
        )
        return {"predictions": predict_features(features)}

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid prediction input: {exc}",
        ) from exc
