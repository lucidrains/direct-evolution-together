"""Standalone Nim Mummy WebSocket + PyTorch evolution server process."""

import os
import signal
import threading


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "direct_evolution_together.settings")
    os.environ["DET_START_WS"] = "0"

    import django
    from django.conf import settings

    django.setup()

    from direct_evolution_together.ws_bridge import start_ws_server, stop_ws_server

    stop_event = threading.Event()

    def _handle_signal(signum: int, frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    start_ws_server(
        host=settings.WS_HOST,
        port=settings.WS_PORT,
        seed=True,
        pop_size=settings.POPULATION_SIZE,
        min_votes=settings.MIN_VOTES,
        max_votes=settings.MAX_VOTES_PER_USER,
        mut_sigma=settings.MUTATION_SIGMA,
        elite_frac=settings.ELITE_FRAC,
        crossover_rate=settings.CROSSOVER_RATE,
        admin_password=settings.ADMIN_PASSWORD,
    )
    try:
        stop_event.wait()
    finally:
        stop_ws_server()


if __name__ == "__main__":
    main()
