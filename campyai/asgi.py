"""ASGI entrypoint for Campy AI."""
import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "campyai.settings.prod")

application = get_asgi_application()
