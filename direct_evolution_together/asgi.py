"""ASGI entrypoint for running Django HTTP views under Uvicorn."""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "direct_evolution_together.settings")

application = get_asgi_application()
