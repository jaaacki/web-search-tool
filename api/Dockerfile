FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && pip check \
    && addgroup --system app \
    && adduser --system --ingroup app app

COPY --chown=app:app app.py .
USER app

FROM base AS test
COPY --chown=app:app test_*.py ./
RUN set -eu; for test in test_*.py; do python "$test"; done

FROM base AS runtime
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
