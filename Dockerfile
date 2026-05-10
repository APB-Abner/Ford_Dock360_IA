FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

COPY requirements-api.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY scripts ./scripts

CMD ["sh", "-c", "python scripts/fetch_model_artifacts.py && uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
