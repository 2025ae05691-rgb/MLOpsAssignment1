
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-runtime.txt .
RUN python -m pip install \
    --no-cache-dir \
    --timeout=600 \
    --retries=10 \
    --progress-bar off \
    -r requirements-runtime.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
