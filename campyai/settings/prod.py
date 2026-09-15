"""Production settings — Postgres, Redis, hardened security headers."""
from .base import *  # noqa: F401,F403
from .base import env, env_bool, env_int, env_list

DEBUG = False
ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", "")
CSRF_TRUSTED_ORIGINS = env_list("DJANGO_CSRF_TRUSTED_ORIGINS", "")

if env("DATABASE_URL"):
    # postgres://user:pass@host:5432/dbname
    from urllib.parse import urlparse

    _url = urlparse(env("DATABASE_URL"))
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": _url.path.lstrip("/"),
            "USER": _url.username or "",
            "PASSWORD": _url.password or "",
            "HOST": _url.hostname or "",
            "PORT": str(_url.port or 5432),
            "CONN_MAX_AGE": env_int("DB_CONN_MAX_AGE", 60),
            "OPTIONS": {"connect_timeout": 10},
        }
    }

if env("REDIS_URL"):
    CACHES = {
        "default": {
            "BACKEND": "django_redis.cache.RedisCache",
            "LOCATION": env("REDIS_URL"),
            "OPTIONS": {"CLIENT_CLASS": "django_redis.client.DefaultClient"},
        }
    }
    SESSION_ENGINE = "django.contrib.sessions.backends.cache"
    SESSION_CACHE_ALIAS = "default"
    CELERY_BROKER_URL = env("CELERY_BROKER_URL", env("REDIS_URL"))
    CELERY_RESULT_BACKEND = env("CELERY_RESULT_BACKEND", env("REDIS_URL"))
    CELERY_TASK_ALWAYS_EAGER = False

SECURE_SSL_REDIRECT = env_bool("DJANGO_SECURE_SSL_REDIRECT", True)
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = env_int("DJANGO_SECURE_HSTS_SECONDS", 31536000)
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

EMAIL_BACKEND = env("DJANGO_EMAIL_BACKEND", "django.core.mail.backends.smtp.EmailBackend")

ADMINS = [
    tuple(part.strip() for part in entry.split(":", 1))
    for entry in env_list("DJANGO_ADMINS")
    if ":" in entry
]

if env("SENTRY_DSN"):  # pragma: no cover - production only
    try:
        import sentry_sdk
        from sentry_sdk.integrations.django import DjangoIntegration

        sentry_sdk.init(
            dsn=env("SENTRY_DSN"),
            integrations=[DjangoIntegration()],
            traces_sample_rate=float(env("SENTRY_TRACES_SAMPLE_RATE", "0.05")),
            send_default_pii=False,
            environment=env("SENTRY_ENVIRONMENT", "production"),
        )
    except ImportError:
        pass
