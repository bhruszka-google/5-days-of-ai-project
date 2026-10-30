# ==============================================================================
# AutoDeal Copilot — Production Cloud Run Container (Python 3.13 + Google ADK)
# ==============================================================================
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    AUTODEAL_FAST_MODEL="gemini-3-flash-preview" \
    AUTODEAL_PRO_MODEL="gemini-3.1-pro-preview" \
    OTEL_TRACES_EXPORTER="google_cloud_trace"

WORKDIR /app

# Create a non-root runtime user for Cloud Run security best practices
RUN groupadd --system autodeal && useradd --system --gid autodeal --create-home autodeal

COPY pyproject.toml README.md ./
COPY app/ ./app/
COPY evals/ ./evals/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . uvicorn

USER autodeal

EXPOSE 8080

CMD ["uvicorn", "app.fast_api_app:app", "--host", "0.0.0.0", "--port", "8080"]
