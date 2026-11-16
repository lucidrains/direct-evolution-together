import hashlib
import hmac
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

def _resolve_media_dir() -> Path:
    """Store cached media renders beside db.sqlite3 in dev or XDG data dir."""
    env_path = os.getenv("DET_MEDIA_DIR")
    if env_path:
        p = Path(env_path)
    elif (BASE_DIR / "manage.py").exists():
        p = BASE_DIR / "media"
    else:
        p = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "direct-evolution-together" / "media"
    p.mkdir(parents=True, exist_ok=True)
    return p


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
    "direct_evolution_together.middleware.MediaSafeGZipMiddleware",
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
MIN_VOTES_EXPLICIT = bool(os.getenv("MUMMY_MIN_VOTES", "").strip())
MIN_VOTES_OVERRIDE = os.getenv("MUMMY_MIN_VOTES_OVERRIDE", "0") == "1"
MIN_VOTES = max(1, min(256, int(os.getenv("MUMMY_MIN_VOTES", "20"))))
COUNTDOWN_MS = max(100, min(60000, int(os.getenv("MUMMY_COUNTDOWN_MS", "8000"))))
MAX_VOTES_PER_USER = max(1, min(64, int(os.getenv("MUMMY_MAX_VOTES", "20"))))
MUTATION_SIGMA = max(0.05, min(1.0, float(os.getenv("MUMMY_MUT_SIGMA", "0.30"))))
ELITE_FRAC = max(0.0, min(0.5, float(os.getenv("MUMMY_ELITE_FRAC", "0.20"))))
CROSSOVER_RATE = max(0.0, min(1.0, float(os.getenv("MUMMY_CROSSOVER_RATE", "0.75"))))
ADMIN_PASSWORD = os.getenv("DET_ADMIN_PASSWORD", "")
EVOLVE_NOW_LOCAL = os.getenv("DET_EVOLVE_NOW", "0") == "1"
EVOLVE_NOW_SECRET = os.getenv("DET_EVOLVE_SECRET", "").strip()
if EVOLVE_NOW_LOCAL and not EVOLVE_NOW_SECRET:
    EVOLVE_NOW_SECRET = "local-loopback"
_evolve_seed = EVOLVE_NOW_SECRET or ADMIN_PASSWORD
EVOLVE_NOW_TOKEN = (
    hmac.new(
        SECRET_KEY.encode("utf-8"),
        f"det-evolve:{_evolve_seed}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:32]
    if _evolve_seed
    else ""
)

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
MEDIA_URL = "media/"
MEDIA_ROOT = _resolve_media_dir()
MAX_MEDIA_CACHE_BYTES = int(os.getenv("DET_MAX_MEDIA_CACHE_BYTES", str(256 * 1024 * 1024)))
DATA_UPLOAD_MAX_MEMORY_SIZE = 4 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 256 * 1024
DATA_UPLOAD_MAX_NUMBER_FILES = 64
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
