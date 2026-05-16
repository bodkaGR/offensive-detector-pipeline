FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV NLTK_DATA=/usr/local/share/nltk_data
RUN python -m nltk.downloader -d $NLTK_DATA wordnet omw-1.4 stopwords

COPY src/ ./src/
COPY scripts/ ./scripts/

RUN mkdir -p data saved_models plots reports

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1