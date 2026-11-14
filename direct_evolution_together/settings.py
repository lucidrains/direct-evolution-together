import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent


def load_dotenv() -> None:
    """Populate unset environment variables from .env in CWD or repo root."""
    seen: set[Path] = set()
    for candidate in (Path.cwd() / ".env", BASE_DIR / ".env"):
        resolved = candidate.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        for raw_line in resolved.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if not key:
                continue
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            os.environ.setdefault(key, val)


load_dotenv()


def _resolve_db_path() -> Path:
    """Use repo db.sqlite3 in development, or a user data dir when pip-installed."""
    env_path = os.getenv("DET_DB_PATH")
    if env_path:
        return Path(env_path)
    if (BASE_DIR / "manage.py").exists():
        return BASE_DIR / "db.sqlite3"
    data_dir = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "direct-evolution-together"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "db.sqlite3"

DEBUG = os.getenv("DJANGO_DEBUG", "1") == "1"
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "").strip()
if not SECRET_KEY:
    if not DEBUG:
        raise ImproperlyConfigured(
            "DJANGO_SECRET_KEY must be set when DJANGO_DEBUG=0"
        )
    SECRET_KEY = "dev-insecure-secret-key-change-in-prod"
ALLOWED_HOSTS = [
    h.strip()
    for h in os.getenv("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")
    if h.strip()
]
CSRF_TRUSTED_ORIGINS = [
    o.strip()
    for o in os.getenv("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",")
    if o.strip()
]
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
USE_X_FORWARDED_HOST = True

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "direct_evolution_together.apps.DirectEvolutionTogetherConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "direct_evolution_together.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": _resolve_db_path(),
        "OPTIONS": {"timeout": 20},
    }
}

_raw_subpath = os.getenv("DET_SUBPATH", "").strip().strip("/")
SUBPATH = f"/{_raw_subpath}" if _raw_subpath else ""
WS_VIA_PROXY = os.getenv("DET_WS_PROXY", "1" if SUBPATH else "0") == "1"
WS_HOST = os.getenv("MUMMY_WS_HOST", "127.0.0.1")
WS_PORT = int(os.getenv("MUMMY_WS_PORT", "8001"))
POPULATION_SIZE = max(3, min(64, int(os.getenv("MUMMY_POP_SIZE", "30"))))
MIN_VOTES = max(1, min(100, int(os.getenv("MUMMY_MIN_VOTES", "3"))))
MAX_VOTES_PER_USER = max(1, min(20, int(os.getenv("MUMMY_MAX_VOTES", "5"))))
MUTATION_SIGMA = max(0.05, min(1.0, float(os.getenv("MUMMY_MUT_SIGMA", "0.35"))))
ELITE_FRAC = max(0.0, min(0.5, float(os.getenv("MUMMY_ELITE_FRAC", "0.10"))))
CROSSOVER_RATE = max(0.0, min(1.0, float(os.getenv("MUMMY_CROSSOVER_RATE", "0.75"))))
ADMIN_PASSWORD = os.getenv("DET_ADMIN_PASSWORD", "")

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
