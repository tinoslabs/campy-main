
import os
from decouple import config, Csv

DEBUG = config("DEBUG", default=False, cast=bool)

ALLOWED_HOSTS = config("ALLOWED_HOSTS", cast=Csv())
CSRF_TRUSTED_ORIGINS = config("CSRF_TRUSTED_ORIGINS", cast=Csv())

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.mysql",
        "NAME": config("DB_NAME"),
        "USER": config("DB_USERNAME"),
        "PASSWORD": config("DB_PASSWORD"),
        "HOST": config("DB_HOSTNAME"),
        "PORT": config("DB_PORT", cast=int),
    }
}

if config("MEDIA_ROOT"):
    MEDIA_ROOT = config("MEDIA_ROOT")

STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedStaticFilesStorage",
    },
}

if not DEBUG:
  LOGGING = {
      'version': 1,
      'disable_existing_loggers': False,
      'formatters': {
          'verbose': {
              'format' : "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
              'datefmt' : "%d/%b/%Y %H:%M:%S"
          },
          'simple': {
              'format': '%(levelname)s %(message)s'
          },
      },
      'handlers': {
          'file': {
              'level': 'DEBUG',
              'class': 'logging.FileHandler',
              'filename': 'error.log',
              'formatter': 'verbose'
          },
      },
      'loggers': {
          'django': {
              'handlers':['file'],
              'propagate': True,
              'level':'ERROR',
          },
          'app': {
              'handlers': ['file'],
              'level': 'DEBUG',
          },
      }
  }

# EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
# EMAIL_HOST = config("EMAIL_HOST")
# EMAIL_HOST_USER = config("EMAIL_HOST_USER")
# EMAIL_HOST_PASSWORD = config("EMAIL_HOST_PASSWORD")
# EMAIL_PORT = config("EMAIL_PORT", cast=int)
# EMAIL_USE_TLS = True
# EMAIL_USE_SSL = False