import importlib
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.monitoring import configure_monitoring


class StubModel:
    classes_ = (0, 1)

    def predict(self, features):
        return (features["age"] >= 50).astype(int).to_numpy()

    def predict_proba(self, features):
        predictions = self.predict(features)
        return [[1.0 - prediction, float(prediction)] for prediction in predictions]


with patch("joblib.load", return_value=StubModel()):
    api_app = importlib.import_module("src.api").app


def create_test_app():
    app = FastAPI()
    configure_monitoring(app)

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.get("/test-error")
    def test_error():
        return {"detail": "test error"}, 500

    return app


def test_health_endpoint():
    client = TestClient(create_test_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_metrics_endpoint():
    client = TestClient(create_test_app())

    # Generate a request so that request metrics exist.
    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "heart_disease_http_requests_total" in response.text
    assert "heart_disease_http_request_duration_seconds" in response.text


def test_api_registers_metrics_endpoint():
    response = TestClient(api_app).get("/metrics")

    assert response.status_code == 200
    assert "heart_disease_http_requests_total" in response.text


def test_model_info_endpoint():
    response = TestClient(api_app).get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["api_version"] == api_app.version
    assert body["feature_count"] == len(body["features"]) == 13
    assert body["model_type"]


def test_metrics_summary_endpoint():
    client = TestClient(api_app)
    client.get("/health")

    response = client.get("/metrics/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["requests_total"] >= 1
    assert body["errors_total"] >= 0
    assert body["average_latency_seconds"] is not None
    assert any(item["path"] == "/health" for item in body["requests"])


def test_batch_prediction_endpoint():
    client = TestClient(api_app)
    response = client.post(
        "/predict/batch",
        json={
            "patients": [
                {
                    "age": 40,
                    "sex": 1,
                    "cp": 2,
                    "trestbps": 120,
                    "chol": 200,
                    "fbs": 0,
                    "restecg": 1,
                    "thalach": 160,
                    "exang": 0,
                    "oldpeak": 0.5,
                    "slope": 2,
                    "ca": 0,
                    "thal": 3,
                },
                {
                    "age": 60,
                    "sex": 0,
                    "cp": 3,
                    "trestbps": 140,
                    "chol": 230,
                    "fbs": 1,
                    "restecg": 0,
                    "thalach": 130,
                    "exang": 1,
                    "oldpeak": 1.2,
                    "slope": 1,
                    "ca": 1,
                    "thal": 6,
                },
            ]
        },
    )

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert [item["prediction"] for item in predictions] == [0, 1]
    assert predictions[0]["probabilities"] == {"0": 1.0, "1": 0.0}
    assert predictions[1]["probabilities"] == {"0": 0.0, "1": 1.0}

    metrics = client.get("/metrics/summary").json()
    assert any(
        item["path"] == "/predict/batch"
        for item in metrics["requests"]
    )


def test_batch_prediction_rejects_empty_patients():
    response = TestClient(api_app).post(
        "/predict/batch",
        json={"patients": []},
    )

    assert response.status_code == 422