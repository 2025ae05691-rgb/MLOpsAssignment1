from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import app as api_app
from src.monitoring import configure_monitoring


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