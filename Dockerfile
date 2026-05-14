FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; \
    nltk.download('wordnet', quiet=True); \
    nltk.download('omw-1.4', quiet=True); \
    nltk.download('stopwords', quiet=True)"

COPY src/ ./src/
COPY scripts/ ./scripts/

RUN mkdir -p data saved_models plots reports

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=http://localhost:5000 \
    MLFLOW_EXPERIMENT_NAME=offensive-text-detection