import argparse
import importlib
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser

NIM_INSTALL_GUIDE = """\
direct-evolution-together runs its WebSocket server on Nim + Mummy
(we intentionally ship pure Nim source with no precompiled binaries!).

To install Nim & Nimble packages:
  1. Install Nim:
     - macOS:         brew install nim
     - Linux / Unix:  curl https://nim-lang.org/choosenim/init.sh -sSf | sh
     - Any platform:  https://nim-lang.org/install.html
  2. Install required Nim packages:
     nimble install -y mummy nimpy
  3. Re-run:
     direct-evolution-together
"""


def check_nim_toolchain(*, auto_install_deps: bool = False) -> tuple[bool, str]:
    """Verify Nim, Nimble, mummy, and nimpy are available for compiling Mummy."""
    if shutil.which("nim") is None or shutil.which("nimble") is None:
        return False, "Missing `nim` or `nimble` in PATH.\n\n" + NIM_INSTALL_GUIDE

    missing = []
    for pkg in ("mummy", "nimpy"):
        res = subprocess.run(
            ["nimble", "path", pkg],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            missing.append(pkg)

    if missing and auto_install_deps:
        install_res = subprocess.run(
            ["nimble", "install", "-y", *missing],
            check=False,
        )
        if install_res.returncode == 0:
            missing = []

    if missing:
        pkg_list = " ".join(missing)
        return (
            False,
            f"Nim is installed, but required Nimble package(s) are missing: {pkg_list}\n"
            f"Run `nimble install -y {pkg_list}` (or pass `--install-nim-deps`) and try again.\n\n"
            + NIM_INSTALL_GUIDE,
        )

    return True, "Nim toolchain OK (nim, nimble, mummy, nimpy)."


def open_browser_when_ready(host: str, port: int, timeout: float = 10.0) -> threading.Thread:
    """Open the local Web UI URL in the default browser once the HTTP port is accepting connections."""
    browse_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    url = f"http://{browse_host}:{port}/"

    def _worker() -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((browse_host, port), timeout=0.25):
                    webbrowser.open(url, new=2)
                    return
            except OSError:
                time.sleep(0.1)

    thread = threading.Thread(target=_worker, name="det-browser-open", daemon=True)
    thread.start()
    return thread


def main() -> None:
    """Run auto-migrations, verify Nim toolchain, and launch Django + Mummy."""
    from direct_evolution_together.settings import load_dotenv

    load_dotenv()
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "direct_evolution_together.settings")

    parser = argparse.ArgumentParser(
        prog="direct-evolution-together",
        description="Launch direct-evolution-together (Django web UI + Nim Mummy WebSockets + PyTorch CPPN)",
    )
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--ws-host", default=os.getenv("MUMMY_WS_HOST", "127.0.0.1"))
    parser.add_argument("--ws-port", type=int, default=int(os.getenv("MUMMY_WS_PORT", "8001")))
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="Override population size across rooms/goals (3-1024, default: per-task policy / 30 for image)",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=None,
        help="Quick local override for required votes per round across all rooms/goals (1-256)",
    )
    parser.add_argument(
        "--countdown-ms",
        type=int,
        default=int(os.getenv("MUMMY_COUNTDOWN_MS", "8000")),
        help="Round countdown timer duration in milliseconds once quorum is reached (100-60000, default: 8000)",
    )
    parser.add_argument(
        "--max-votes-per-user",
        type=int,
        default=int(os.getenv("MUMMY_MAX_VOTES", "20")),
        help="Max active votes per user per round in public rooms and default for private rooms (1-64, default: 20)",
    )
    parser.add_argument(
        "--mutation-sigma",
        type=float,
        default=float(os.getenv("MUMMY_MUT_SIGMA", "0.30")),
        help="Constant GA mutation sigma for public rooms and default for private rooms (0.05-1.0, default: 0.30)",
    )
    parser.add_argument(
        "--elite-frac",
        type=float,
        default=float(os.getenv("MUMMY_ELITE_FRAC", "0.20")),
        help="Elite fraction carried over per generation for public rooms and default for private rooms (0.0-0.5, default: 0.20)",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=float(os.getenv("MUMMY_CROSSOVER_RATE", "0.75")),
        help="Crossover probability for breeding for public rooms and default for private rooms (0.0-1.0, default: 0.75)",
    )
    parser.add_argument(
        "--large-pop",
        action="store_true",
        default=os.getenv("DET_LARGE_POP", "0") == "1",
        help="Enable large-population preset (250 Microduck gait + 1,024 room cap)",
    )
    parser.add_argument(
        "--evolve-now",
        action="store_true",
        default=os.getenv("DET_EVOLVE_NOW", "0") == "1",
        help="Enable operator-only 'Evolve now' button on localhost sessions",
    )
    parser.add_argument(
        "--admin-password",
        default=os.getenv("DET_ADMIN_PASSWORD", ""),
        help="Admin/operator password for /reset/ and ?evolve=<password> unlock",
    )
    parser.add_argument("--reload", action="store_true", help="Enable Django dev autoreloader")
    parser.add_argument("--noreload", action="store_true", help="Disable Django autoreloader (default)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the Web UI in a browser")
    parser.add_argument("--check", action="store_true", help="Check Nim/Mummy readiness and exit")
    parser.add_argument(
        "--compile-scene",
        nargs=2,
        metavar=("SCENE_ID", "ENTRY_XML"),
        help="Compile an MJCF + STL/OBJ scene directory into physics.bin + visual.bin + manifest.json and exit",
    )
    parser.add_argument(
        "--stage-assets",
        metavar="DEST_DIR",
        help="Stage frontend & simulation assets (+ .gz sidecars) into DEST_DIR for Nginx and exit",
    )
    parser.add_argument(
        "--install-nim-deps",
        action="store_true",
        help="Automatically run `nimble install -y mummy nimpy` if missing",
    )
    args = parser.parse_args()

    if args.stage_assets:
        from direct_evolution_together.public_assets import stage_public_assets

        stats = stage_public_assets(args.stage_assets, precompress=True)
        print(
            f"Staged {stats['total']} public assets -> {args.stage_assets} "
            f"({stats['copied']} updated, {stats['gzipped']} precompressed, {stats['removed']} pruned)"
        )
        return

    if args.compile_scene:
        import json
        from direct_evolution_together.environments.scene_assets import compile_mjcf_scene

        sid, xml_p = args.compile_scene
        manifest = compile_mjcf_scene(sid, xml_p)
        print(json.dumps(manifest, indent=2))
        return

    is_reloader_child = os.environ.get("RUN_MAIN") == "true"
    if not is_reloader_child:
        ok, message = check_nim_toolchain(auto_install_deps=args.install_nim_deps)
        if not ok:
            print(message, file=sys.stderr)
            raise SystemExit(1)
        if args.check:
            print(message)
            return

    pop_size_override = args.population_size is not None
    raw_pop = args.population_size if pop_size_override else int(os.getenv("MUMMY_POP_SIZE", "30"))
    pop_size = max(3, min(1024, raw_pop))
    min_votes_override = args.min_votes is not None
    countdown_ms = max(100, min(60000, args.countdown_ms))
    max_votes = max(1, min(256, args.max_votes_per_user))
    mut_sigma = max(0.05, min(1.0, float(args.mutation_sigma)))
    elite_frac = max(0.0, min(0.5, float(args.elite_frac)))
    crossover_rate = max(0.0, min(1.0, float(args.crossover_rate)))
    os.environ["MUMMY_WS_HOST"] = args.ws_host
    os.environ["MUMMY_WS_PORT"] = str(args.ws_port)
    if pop_size_override:
        os.environ["MUMMY_POP_SIZE"] = str(pop_size)
        os.environ["MUMMY_POP_SIZE_OVERRIDE"] = "1"
    if min_votes_override:
        os.environ["MUMMY_MIN_VOTES"] = str(max(1, min(1024, args.min_votes)))
        os.environ["MUMMY_MIN_VOTES_OVERRIDE"] = "1"
    os.environ["MUMMY_COUNTDOWN_MS"] = str(countdown_ms)
    os.environ["MUMMY_MAX_VOTES"] = str(max_votes)
    os.environ["MUMMY_MUT_SIGMA"] = f"{mut_sigma:.2f}"
    os.environ["MUMMY_ELITE_FRAC"] = f"{elite_frac:.2f}"
    os.environ["MUMMY_CROSSOVER_RATE"] = f"{crossover_rate:.2f}"
    if args.large_pop:
        os.environ["DET_LARGE_POP"] = "1"
    if args.admin_password:
        os.environ["DET_ADMIN_PASSWORD"] = args.admin_password
    if args.evolve_now:
        os.environ["DET_EVOLVE_NOW"] = "1"
        os.environ.setdefault("DET_EVOLVE_SECRET", secrets.token_hex(16))

    from direct_evolution_together import settings as det_settings
    from direct_evolution_together.config import reload_config

    reload_config()
    importlib.reload(det_settings)

    eff_pop_override = pop_size_override or det_settings.POPULATION_SIZE_OVERRIDE
    eff_min_votes = (
        det_settings.MIN_VOTES
        if (min_votes_override or det_settings.MIN_VOTES_EXPLICIT)
        else None
    )

    import django
    from django.core.management import call_command, execute_from_command_line

    use_noreload = (not args.reload) or args.noreload

    if not is_reloader_child:
        os.environ["DET_START_WS"] = "0"
        django.setup()
        call_command("migrate", interactive=False, verbosity=0)
        print(
            f"direct-evolution-together · Web UI: http://{args.host}:{args.port}/ "
            f"| WebSocket (Nim + Mummy): ws://{args.ws_host}:{args.ws_port}",
            flush=True,
        )
        overrides: list[str] = []
        if min_votes_override:
            overrides.append(f"min_votes={eff_min_votes}")
        if countdown_ms != 8000:
            overrides.append(f"countdown_ms={countdown_ms}")
        if args.large_pop:
            overrides.append("large_pop=1")
        if args.evolve_now:
            overrides.append("evolve_now=localhost-only")
        if overrides:
            print(f"Local overrides active: {', '.join(overrides)}", flush=True)
        if use_noreload:
            from direct_evolution_together.ws_bridge import start_ws_server

            start_ws_server(
                host=args.ws_host,
                port=args.ws_port,
                pop_size=pop_size,
                pop_size_override=eff_pop_override,
                min_votes=eff_min_votes,
                min_votes_override=min_votes_override,
                countdown_ms=countdown_ms,
                max_votes=max_votes,
                mut_sigma=mut_sigma,
                elite_frac=elite_frac,
                crossover_rate=crossover_rate,
                admin_password=det_settings.ADMIN_PASSWORD,
                evolve_token=det_settings.EVOLVE_NOW_TOKEN,
            )
        if not args.no_browser:
            open_browser_when_ready(args.host, args.port)

    os.environ["DET_START_WS"] = "1"
    cmd = [__file__, "runserver", f"{args.host}:{args.port}"]
    if use_noreload:
        cmd.append("--noreload")
    sys.argv = cmd
    execute_from_command_line(cmd)


if __name__ == "__main__":
    main()
