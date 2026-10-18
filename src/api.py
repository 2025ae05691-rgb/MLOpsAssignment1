
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


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(patient: PatientInput):
    try:
        features = pd.DataFrame(
            [patient.model_dump()],
            columns=FEATURES,
        )

        prediction = int(model.predict(features)[0])
        result = {"prediction": prediction}

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            result["probabilities"] = {
                str(label): round(float(probability), 4)
                for label, probability in zip(
                    model.classes_, probabilities
                )
            }

        return result

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid prediction input: {exc}",
        ) from exc
