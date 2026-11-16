import os
import sys

from django.apps import AppConfig
from django.conf import settings
from django.db.backends.signals import connection_created


def _enable_sqlite_wal(sender, connection, **kwargs) -> None:
    if connection.vendor == "sqlite":
        with connection.cursor() as cursor:
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")


class DirectEvolutionTogetherConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "direct_evolution_together"

    def ready(self) -> None:
        connection_created.connect(_enable_sqlite_wal, dispatch_uid="det_sqlite_wal")
        is_runserver = "runserver" in sys.argv or os.environ.get("DET_START_WS") == "1"
        if not is_runserver:
            return
        # Under StatReloader, only start Mummy in the serving child process (RUN_MAIN=true),
        # or immediately when --noreload is used.
        if os.environ.get("RUN_MAIN") == "true" or "--noreload" in sys.argv:
            from direct_evolution_together.ws_bridge import start_ws_server

            start_ws_server(
                host=settings.WS_HOST,
                port=settings.WS_PORT,
                pop_size=settings.POPULATION_SIZE,
                min_votes=settings.MIN_VOTES if settings.MIN_VOTES_EXPLICIT else None,
                min_votes_override=settings.MIN_VOTES_OVERRIDE,
                countdown_ms=settings.COUNTDOWN_MS,
                max_votes=settings.MAX_VOTES_PER_USER,
                mut_sigma=settings.MUTATION_SIGMA,
                elite_frac=settings.ELITE_FRAC,
                crossover_rate=settings.CROSSOVER_RATE,
                evolve_token=settings.EVOLVE_NOW_TOKEN,
            )
