"""Standalone Nim Mummy WebSocket + PyTorch evolution server process."""

import os
import signal
import threading


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "direct_evolution_together.settings")
    os.environ["DET_START_WS"] = "0"

    import django
    from django.conf import settings
    import torch

    if not torch.cuda.is_available():
        torch.set_num_threads(1)

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
        min_votes=settings.MIN_VOTES if settings.MIN_VOTES_EXPLICIT else None,
        min_votes_override=settings.MIN_VOTES_OVERRIDE,
        countdown_ms=settings.COUNTDOWN_MS,
        max_votes=settings.MAX_VOTES_PER_USER,
        mut_sigma=settings.MUTATION_SIGMA,
        elite_frac=settings.ELITE_FRAC,
        crossover_rate=settings.CROSSOVER_RATE,
        brood_size=settings.BROOD_SIZE,
        admin_password=settings.ADMIN_PASSWORD,
        evolve_token=settings.EVOLVE_NOW_TOKEN,
    )
    try:
        stop_event.wait()
    finally:
        stop_ws_server()


if __name__ == "__main__":
    main()
