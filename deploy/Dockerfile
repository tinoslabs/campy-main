# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DJANGO_SETTINGS_MODULE=campyai.settings.prod

# libGL and libglib are needed by OpenCV, which decodes RTSP streams.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libpq-dev libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-prod.txt ./
RUN pip install -r requirements-prod.txt

COPY . .

# Run as a non-root user, and give it the writable paths it needs.
RUN useradd --create-home --uid 10001 campy \
    && mkdir -p /var/lib/campyai/{models,datasets,media} /app/staticfiles \
    && chown -R campy:campy /app /var/lib/campyai
USER campy

RUN SECRET_KEY=build-only python manage.py collectstatic --noinput

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["gunicorn", "campyai.wsgi:application", \
     "--bind", "0.0.0.0:8000", "--workers", "4", \
     "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-"]
