"""
Application monitoring and logging for the Heart Disease API.

Exposes Prometheus metrics at /metrics and records operational
request information without logging patient input payloads.
"""

import logging
import time

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

# Configure application logging.
logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("heart_disease_api")

# Track requests by HTTP method, normalized path and status code.
REQUEST_COUNT = Counter(
    "heart_disease_http_requests_total",
    "Total HTTP requests handled by the Heart Disease API",
    ["method", "path", "status"],
)

# Measure request duration in seconds.
REQUEST_LATENCY = Histogram(
    "heart_disease_http_request_duration_seconds",
    "HTTP request processing duration in seconds",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

# Count HTTP responses with status codes >= 400.
HTTP_ERRORS = Counter(
    "heart_disease_http_errors_total",
    "Total HTTP responses with error status codes",
    ["method", "path", "status"],
)


def normalize_path(request: Request) -> str:
    """Limit metric labels to a small, predictable set of paths."""
    path = request.url.path

    if path in {"/health", "/metrics", "/docs", "/openapi.json"}:
        return path

    if path == "/predict":
        return path

    return "/other"


def configure_monitoring(app: FastAPI) -> None:
    """Register logging middleware and the Prometheus endpoint."""

    @app.middleware("http")
    async def monitor_requests(request: Request, call_next):
        method = request.method
        path = normalize_path(request)
        start_time = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response

        except Exception:
            logger.exception(
                "Request failed | method=%s | path=%s",
                method,
                path,
            )
            raise

        finally:
            elapsed = time.perf_counter() - start_time

            REQUEST_COUNT.labels(
                method=method,
                path=path,
                status=str(status_code),
            ).inc()

            REQUEST_LATENCY.labels(
                method=method,
                path=path,
            ).observe(elapsed)

            if status_code >= 400:
                HTTP_ERRORS.labels(
                    method=method,
                    path=path,
                    status=str(status_code),
                ).inc()

            logger.info(
                "HTTP request | method=%s | path=%s | "
                "status=%s | latency_seconds=%.4f",
                method,
                path,
                status_code,
                elapsed,
            )

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        """Return metrics in Prometheus exposition format."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )