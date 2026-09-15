"""Shared Django settings for the Campy AI platform."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv(BASE_DIR / ".env")


def env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_list(key: str, default: str = "") -> list[str]:
    raw = os.environ.get(key, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------
SECRET_KEY = env("DJANGO_SECRET_KEY", "django-insecure-campyai-change-me-in-production")
DEBUG = env_bool("DJANGO_DEBUG", False)
ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]")
CSRF_TRUSTED_ORIGINS = env_list("DJANGO_CSRF_TRUSTED_ORIGINS")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.humanize",
    "django.contrib.sitemaps",
    # third party
    "rest_framework",
    "rest_framework_simplejwt",
    "django_filters",
    "drf_spectacular",
    "corsheaders",
    # Campy AI apps
    "apps.core",
    "apps.accounts",
    "apps.cameras",
    "apps.aiengine",
    "apps.training",
    "apps.events",
    "apps.analytics",
    "apps.billing",
    "apps.cms",
    "apps.api",
    "apps.dashboard",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "apps.accounts.middleware.CurrentOrganizationMiddleware",
    "apps.accounts.middleware.AuditContextMiddleware",
    "apps.billing.middleware.SubscriptionGuardMiddleware",
    "apps.cms.middleware.SiteSettingsMiddleware",
]

ROOT_URLCONF = "campyai.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "apps.core.context_processors.branding",
                "apps.accounts.context_processors.workspace",
                "apps.cms.context_processors.cms_navigation",
                "apps.billing.context_processors.subscription",
            ],
            # Registered as builtins so every template — including the small
            # included partials — can use them without a per-file {% load %}.
            "builtins": [
                "apps.core.templatetags.campy",
                "django.contrib.humanize.templatetags.humanize",
            ],
        },
    },
]

WSGI_APPLICATION = "campyai.wsgi.application"
ASGI_APPLICATION = "campyai.asgi.application"

# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "data" / "campyai.sqlite3",
        "OPTIONS": {"timeout": 30},
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
AUTH_USER_MODEL = "accounts.User"

AUTHENTICATION_BACKENDS = [
    "apps.accounts.backends.EmailOrUsernameBackend",
    "django.contrib.auth.backends.ModelBackend",
]

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
        "OPTIONS": {"min_length": 10},
    },
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LOGIN_URL = "accounts:login"
LOGIN_REDIRECT_URL = "dashboard:home"
LOGOUT_REDIRECT_URL = "cms:home"

# --------------------------------------------------------------------------
# I18N / TZ
# --------------------------------------------------------------------------
LANGUAGE_CODE = env("DJANGO_LANGUAGE_CODE", "en-us")
TIME_ZONE = env("DJANGO_TIME_ZONE", "Asia/Kolkata")
USE_I18N = True
USE_TZ = True

# --------------------------------------------------------------------------
# Static / media
# --------------------------------------------------------------------------
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"]
MEDIA_URL = "/media/"
MEDIA_ROOT = Path(env("DJANGO_MEDIA_ROOT", str(BASE_DIR / "media")))

STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage"},
}

FILE_UPLOAD_MAX_MEMORY_SIZE = 20 * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = 30 * 1024 * 1024
DATA_UPLOAD_MAX_NUMBER_FIELDS = 5000

# --------------------------------------------------------------------------
# REST framework
# --------------------------------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "apps.api.authentication.OrganizationAPIKeyAuthentication",
        "rest_framework_simplejwt.authentication.JWTAuthentication",
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
    "DEFAULT_FILTER_BACKENDS": [
        "django_filters.rest_framework.DjangoFilterBackend",
        "rest_framework.filters.SearchFilter",
        "rest_framework.filters.OrderingFilter",
    ],
    "DEFAULT_PAGINATION_CLASS": "apps.api.pagination.CampyPageNumberPagination",
    "PAGE_SIZE": 25,
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.ScopedRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "ingest": "6000/hour",
        "auth": "30/hour",
        "public": "120/hour",
    },
    "EXCEPTION_HANDLER": "apps.api.exceptions.campy_exception_handler",
}

SPECTACULAR_SETTINGS = {
    "TITLE": "Campy AI Platform API",
    "DESCRIPTION": (
        "Turn ordinary CCTV into an AI workforce-intelligence platform. "
        "Gesture tracking, face recognition, geofencing, crowd management, "
        "theft detection, dangerous-object and fire detection."
    ),
    "VERSION": "1.0.0",
    "SERVE_INCLUDE_SCHEMA": False,
    "COMPONENT_SPLIT_REQUEST": True,
    "SCHEMA_PATH_PREFIX": "/api/v1",
    # Severity and model slots are one vocabulary each, reused across events,
    # zones, incidents, alert rules, jobs and model versions. Without these the
    # generator names the same enum after whichever field it met first, so the
    # schema gains near-duplicate types that clients then generate twice.
    "ENUM_NAME_OVERRIDES": {
        "SeverityEnum": "apps.events.models.SEVERITY_CHOICES",
        "ModelSlotEnum": "apps.training.models.SLOT_CHOICES",
    },
}

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": __import__("datetime").timedelta(minutes=env_int("JWT_ACCESS_MINUTES", 60)),
    "REFRESH_TOKEN_LIFETIME": __import__("datetime").timedelta(days=env_int("JWT_REFRESH_DAYS", 14)),
    "ROTATE_REFRESH_TOKENS": True,
    "UPDATE_LAST_LOGIN": True,
    "AUTH_HEADER_TYPES": ("Bearer",),
}

CORS_ALLOWED_ORIGINS = env_list("DJANGO_CORS_ALLOWED_ORIGINS")
CORS_ALLOW_CREDENTIALS = True

# --------------------------------------------------------------------------
# Email
# --------------------------------------------------------------------------
EMAIL_BACKEND = env("DJANGO_EMAIL_BACKEND", "django.core.mail.backends.console.EmailBackend")
EMAIL_HOST = env("EMAIL_HOST", "localhost")
EMAIL_PORT = env_int("EMAIL_PORT", 587)
EMAIL_HOST_USER = env("EMAIL_HOST_USER", "")
EMAIL_HOST_PASSWORD = env("EMAIL_HOST_PASSWORD", "")
EMAIL_USE_TLS = env_bool("EMAIL_USE_TLS", True)
DEFAULT_FROM_EMAIL = env("DEFAULT_FROM_EMAIL", "Campy AI <no-reply@campy.ai>")
SERVER_EMAIL = DEFAULT_FROM_EMAIL

# --------------------------------------------------------------------------
# Caching / queue
# --------------------------------------------------------------------------
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "campyai-default",
    }
}

CELERY_BROKER_URL = env("CELERY_BROKER_URL", "memory://")
CELERY_RESULT_BACKEND = env("CELERY_RESULT_BACKEND", "cache+memory://")
CELERY_TASK_ALWAYS_EAGER = env_bool("CELERY_TASK_ALWAYS_EAGER", True)
CELERY_TASK_SERIALIZER = "json"
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TIMEZONE = TIME_ZONE

# --------------------------------------------------------------------------
# Campy AI — product configuration
# --------------------------------------------------------------------------
# Django 6.0 makes "https" the default scheme for URL form fields. Opting in
# now means a link typed as "example.com/hook" is stored the same way before
# and after that upgrade, instead of silently changing on it.
#
# Django warns that this setting is itself transitional: DELETE THIS LINE when
# moving to Django 6.0, where the behaviour it selects becomes the default.
FORMS_URLFIELD_ASSUME_HTTPS = True

CAMPY = {
    "BRAND_NAME": env("CAMPY_BRAND_NAME", "Campy AI"),
    "BRAND_TAGLINE": env(
        "CAMPY_BRAND_TAGLINE", "Turn ordinary CCTV into workplace intelligence"
    ),
    "SUPPORT_EMAIL": env("CAMPY_SUPPORT_EMAIL", "support@campy.ai"),
    "LOGO_PATH": env("CAMPY_LOGO_PATH", "brand/campy-ai-logo.svg"),
    "LOGO_MARK_PATH": env("CAMPY_LOGO_MARK_PATH", "brand/campy-ai-mark.svg"),
    # Directory that holds trained model weights (.npz) produced by our trainer.
    "MODEL_ROOT": Path(env("CAMPY_MODEL_ROOT", str(BASE_DIR / "var" / "models"))),
    "DATASET_ROOT": Path(env("CAMPY_DATASET_ROOT", str(BASE_DIR / "var" / "datasets"))),
    "SNAPSHOT_ROOT": Path(env("CAMPY_SNAPSHOT_ROOT", str(MEDIA_ROOT / "snapshots"))),
    # Inference worker tuning
    "WORKER_TARGET_FPS": float(env("CAMPY_WORKER_TARGET_FPS", "6")),
    "WORKER_FRAME_WIDTH": env_int("CAMPY_WORKER_FRAME_WIDTH", 480),
    "WORKER_FRAME_HEIGHT": env_int("CAMPY_WORKER_FRAME_HEIGHT", 270),
    "EVENT_RETENTION_DAYS": env_int("CAMPY_EVENT_RETENTION_DAYS", 90),
    "SNAPSHOT_RETENTION_DAYS": env_int("CAMPY_SNAPSHOT_RETENTION_DAYS", 30),
    # Live view: one reader per camera holds the stream open and every viewer
    # is served from its newest frame, so watching costs the camera one session.
    "LIVE_FIRST_FRAME_TIMEOUT": float(env("CAMPY_LIVE_FIRST_FRAME_TIMEOUT", "10")),
    "LIVE_STREAM_SECONDS": float(env("CAMPY_LIVE_STREAM_SECONDS", "300")),
    # Privacy: blur faces of non-enrolled people in stored snapshots
    "PRIVACY_BLUR_UNKNOWN_FACES": env_bool("CAMPY_PRIVACY_BLUR", False),
    "DEFAULT_CURRENCY": env("CAMPY_DEFAULT_CURRENCY", "INR"),
}

for _directory in (
    CAMPY["MODEL_ROOT"],
    CAMPY["DATASET_ROOT"],
    CAMPY["SNAPSHOT_ROOT"],
    BASE_DIR / "data",
):
    Path(_directory).mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Payments
# --------------------------------------------------------------------------
RAZORPAY = {
    "KEY_ID": env("RAZORPAY_KEY_ID", ""),
    "KEY_SECRET": env("RAZORPAY_KEY_SECRET", ""),
    "WEBHOOK_SECRET": env("RAZORPAY_WEBHOOK_SECRET", ""),
    "CURRENCY": env("RAZORPAY_CURRENCY", "INR"),
}

STRIPE = {
    "PUBLISHABLE_KEY": env("STRIPE_PUBLISHABLE_KEY", ""),
    "SECRET_KEY": env("STRIPE_SECRET_KEY", ""),
    "WEBHOOK_SECRET": env("STRIPE_WEBHOOK_SECRET", ""),
    "CURRENCY": env("STRIPE_CURRENCY", "usd"),
}

# --------------------------------------------------------------------------
# Security
# --------------------------------------------------------------------------
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_HTTPONLY = False
CSRF_COOKIE_SAMESITE = "Lax"
X_FRAME_OPTIONS = "DENY"
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "strict-origin-when-cross-origin"
SESSION_COOKIE_AGE = env_int("SESSION_COOKIE_AGE", 60 * 60 * 12)

MESSAGE_STORAGE = "django.contrib.messages.storage.session.SessionStorage"

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[{asctime}] {levelname} {name}:{lineno} — {message}",
            "style": "{",
        },
        "simple": {"format": "{levelname} {message}", "style": "{"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "verbose"},
    },
    "root": {"handlers": ["console"], "level": env("DJANGO_LOG_LEVEL", "INFO")},
    "loggers": {
        "django.db.backends": {"level": "WARNING", "handlers": ["console"], "propagate": False},
        "campy": {"level": env("CAMPY_LOG_LEVEL", "INFO"), "handlers": ["console"], "propagate": False},
    },
}
