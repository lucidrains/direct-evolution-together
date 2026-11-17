"""Env-driven Fabric deployment script for direct-evolution-together."""

from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from fabric import Connection, task

REPO_ROOT = Path(__file__).resolve().parent
NGINX_TEMPLATE = REPO_ROOT / "deploy" / "nginx.conf.tpl"


def _load_local_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
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
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        if key:
            os.environ.setdefault(key, val)


def _require_env(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise SystemExit(f"Missing required environment variable: {name}")
    return val


@dataclass(frozen=True)
class DeployConfig:
    host: str
    user: str | None
    app_name: str
    deploy_dir: str
    domain: str
    subpath: str
    http_port: int
    ws_port: int
    uvicorn_workers: int
    python_version: str
    cuda_visible_devices: str
    nginx_vhost_file: str
    secret_key: str
    admin_password: str
    allowed_hosts: str
    csrf_trusted_origins: str
    pop_size: int
    min_votes: str
    max_votes: int
    mut_sigma: str
    elite_frac: str
    crossover_rate: str
    process_manager: str = "pm2"

    @classmethod
    def from_env(cls) -> "DeployConfig":
        _load_local_dotenv()
        host = _require_env("DEPLOY_HOST")
        app_name = os.getenv("DEPLOY_APP_NAME", "direct-evolution-together").strip() or "direct-evolution-together"
        deploy_dir = os.getenv("DEPLOY_DIR", f"/var/www/{app_name}").rstrip("/")
        domain = _require_env("DEPLOY_DOMAIN")
        raw_subpath = (
            os.getenv("DEPLOY_SUBPATH") or os.getenv("DET_SUBPATH") or f"/{app_name}"
        ).strip().strip("/")
        subpath = f"/{raw_subpath}" if raw_subpath else ""
        http_port = int(os.getenv("DEPLOY_HTTP_PORT") or os.getenv("DET_HTTP_PORT") or "8000")
        ws_port = int(os.getenv("DEPLOY_WS_PORT") or os.getenv("MUMMY_WS_PORT") or "8001")
        workers = max(1, int(os.getenv("DEPLOY_UVICORN_WORKERS", "2")))
        python_version = os.getenv("DEPLOY_PYTHON_VERSION", "3.11").strip() or "3.11"
        process_manager = (os.getenv("DEPLOY_PROCESS_MANAGER", "pm2").strip().lower() or "pm2")
        if process_manager not in ("pm2", "systemd"):
            raise SystemExit(f"Invalid DEPLOY_PROCESS_MANAGER={process_manager!r}; expected 'pm2' or 'systemd'")
        cuda_visible_devices = os.getenv("DEPLOY_CUDA_VISIBLE_DEVICES", "").strip()
        vhost_file = os.getenv("NGINX_VHOST_FILE", "").strip()
        secret_key = (os.getenv("DEPLOY_SECRET_KEY") or os.getenv("DJANGO_SECRET_KEY") or "").strip()
        if not secret_key:
            raise SystemExit("Missing required environment variable: DEPLOY_SECRET_KEY (or DJANGO_SECRET_KEY)")
        admin_password = (os.getenv("DEPLOY_ADMIN_PASSWORD") or os.getenv("DET_ADMIN_PASSWORD") or "").strip()
        if not admin_password:
            raise SystemExit("Missing required environment variable: DEPLOY_ADMIN_PASSWORD (or DET_ADMIN_PASSWORD)")
        default_hosts = ",".join(dict.fromkeys(domain.split() + ["127.0.0.1", "localhost"]))
        allowed_hosts = os.getenv("DEPLOY_ALLOWED_HOSTS") or os.getenv("DJANGO_ALLOWED_HOSTS") or default_hosts
        default_origins = ",".join(f"https://{h}" for h in domain.split() if h)
        csrf_origins = os.getenv("DEPLOY_CSRF_TRUSTED_ORIGINS") or os.getenv("DJANGO_CSRF_TRUSTED_ORIGINS") or default_origins
        return cls(
            host=host,
            user=os.getenv("DEPLOY_USER", "").strip() or None,
            app_name=app_name,
            deploy_dir=deploy_dir,
            domain=domain,
            subpath=subpath,
            http_port=http_port,
            ws_port=ws_port,
            uvicorn_workers=workers,
            python_version=python_version,
            cuda_visible_devices=cuda_visible_devices,
            nginx_vhost_file=vhost_file,
            secret_key=secret_key,
            admin_password=admin_password,
            allowed_hosts=allowed_hosts,
            csrf_trusted_origins=csrf_origins,
            pop_size=int(os.getenv("MUMMY_POP_SIZE", "30")),
            min_votes=os.getenv("MUMMY_MIN_VOTES", "20").strip(),
            max_votes=int(os.getenv("MUMMY_MAX_VOTES", "20")),
            mut_sigma=os.getenv("MUMMY_MUT_SIGMA", "0.30"),
            elite_frac=os.getenv("MUMMY_ELITE_FRAC", "0.20"),
            crossover_rate=os.getenv("MUMMY_CROSSOVER_RATE", "0.75"),
            process_manager=process_manager,
        )

    @property
    def src_dir(self) -> str:
        return f"{self.deploy_dir}/src"

    @property
    def data_dir(self) -> str:
        return f"{self.deploy_dir}/data"

    @property
    def media_dir(self) -> str:
        return f"{self.data_dir}/media"

    @property
    def run_dir(self) -> str:
        return f"{self.deploy_dir}/run"

    @property
    def env_file(self) -> str:
        return f"{self.deploy_dir}/.env"

    @property
    def ecosystem_file(self) -> str:
        return f"{self.deploy_dir}/ecosystem.config.cjs"

    @property
    def nginx_conf(self) -> str:
        return f"{self.deploy_dir}/nginx.conf"

    @property
    def ws_service(self) -> str:
        return f"{self.app_name}-ws"

    @property
    def uvicorn_service(self) -> str:
        return f"{self.app_name}-uvicorn"

    def connect(self) -> Connection:
        return Connection(host=self.host, user=self.user)


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _render_runtime_env_dict(cfg: DeployConfig) -> dict[str, str]:
    env: dict[str, str] = {
        "DJANGO_SECRET_KEY": cfg.secret_key,
        "DJANGO_DEBUG": "0",
        "DJANGO_ALLOWED_HOSTS": cfg.allowed_hosts,
        "DJANGO_CSRF_TRUSTED_ORIGINS": cfg.csrf_trusted_origins,
        "DET_DB_PATH": f"{cfg.data_dir}/db.sqlite3",
        "DET_MEDIA_DIR": cfg.media_dir,
        "DET_SUBPATH": cfg.subpath,
        "DET_WS_PROXY": "1" if cfg.subpath else "0",
        "MUMMY_WS_HOST": "127.0.0.1",
        "MUMMY_WS_PORT": str(cfg.ws_port),
        "DET_ADMIN_PASSWORD": cfg.admin_password,
        "CUDA_VISIBLE_DEVICES": cfg.cuda_visible_devices,
        "MUMMY_POP_SIZE": str(cfg.pop_size),
        **({"MUMMY_MIN_VOTES": cfg.min_votes} if cfg.min_votes else {}),
        "MUMMY_MAX_VOTES": str(cfg.max_votes),
        "MUMMY_MUT_SIGMA": cfg.mut_sigma,
        "MUMMY_ELITE_FRAC": cfg.elite_frac,
        "MUMMY_CROSSOVER_RATE": cfg.crossover_rate,
    }
    _pass_prefixes = (
        "DET_CONFIG_FILE",
        "DET_MUJOCO_WASM_URL",
        "DET_CROWD_",
        "DET_EXECUTION_POLICY_",
        "DET_TASK_ENABLED_",
        "DET_POP_SIZE_",
        "DET_MAX_POP_SIZE_",
        "DET_MIN_VOTES_",
        "DET_ELITE_FRAC_",
        "DET_ALLOW_VIDEO_",
    )
    for key, val in sorted(os.environ.items()):
        if key.startswith(_pass_prefixes) and val.strip():
            env[key] = val.strip()
    return env


def _render_runtime_env(cfg: DeployConfig) -> str:
    return "\n".join(f"{k}={v}" for k, v in _render_runtime_env_dict(cfg).items()) + "\n"


def _remote_bin_path(cfg: DeployConfig) -> str:
    home_dir = f"/home/{cfg.user}" if cfg.user and cfg.user != "root" else "/root"
    parts = [
        f"{cfg.src_dir}/.venv/bin",
        f"{home_dir}/.local/bin",
        f"{home_dir}/.nimble/bin",
    ]
    if home_dir != "/root":
        parts.extend(["/root/.local/bin", "/root/.nimble/bin"])
    parts.extend(["/usr/local/bin", "/usr/bin", "/bin"])
    return ":".join(parts)


def _render_pm2_ecosystem(cfg: DeployConfig) -> str:
    """Render remote-only PM2 ecosystem config (never checked into git)."""
    app_env = {
        **_render_runtime_env_dict(cfg),
        "PATH": _remote_bin_path(cfg),
        "PYTHONUNBUFFERED": "1",
    }
    apps = [
        {
            "name": cfg.ws_service,
            "cwd": cfg.src_dir,
            "script": f"{cfg.src_dir}/.venv/bin/python",
            "args": "-m direct_evolution_together.ws_server",
            "interpreter": "none",
            "exec_mode": "fork",
            "instances": 1,
            "autorestart": True,
            "restart_delay": 2000,
            "kill_timeout": 5000,
            "env": app_env,
        },
        {
            "name": cfg.uvicorn_service,
            "cwd": cfg.src_dir,
            "script": f"{cfg.src_dir}/.venv/bin/uvicorn",
            "args": (
                f"direct_evolution_together.asgi:application "
                f"--host 127.0.0.1 --port {cfg.http_port} "
                f"--workers {cfg.uvicorn_workers} --proxy-headers --forwarded-allow-ips=127.0.0.1"
            ),
            "interpreter": "none",
            "exec_mode": "fork",
            "instances": 1,
            "autorestart": True,
            "restart_delay": 2000,
            "kill_timeout": 5000,
            "env": app_env,
        },
    ]
    return f"module.exports = {json.dumps({'apps': apps}, indent=2)};\n"


def _bash_remote_path_export() -> str:
    return "\n".join([
        'for _nd in "$HOME"/.nvm/versions/node/*/bin "$HOME"/.volta/bin "$HOME"/.npm-global/bin "$HOME"/.local/share/pnpm; do',
        '  [ -d "$_nd" ] && PATH="$_nd:$PATH"',
        "done",
        'export PATH="$HOME/.local/bin:$HOME/.nimble/bin:/usr/local/bin:$PATH"',
    ])


def _render_nginx_conf(cfg: DeployConfig) -> str:
    subpath = cfg.subpath
    redirect_block = (
        f"location = {subpath} {{\n"
        f"    return 301 $scheme://$http_host{subpath}/;\n"
        "}\n\n"
        if subpath
        else ""
    )
    http_loc_mod = "^~ " if subpath else ""
    forwarded_prefix_line = f"    proxy_set_header X-Forwarded-Prefix {subpath};\n" if subpath else ""

    locations = NGINX_TEMPLATE.read_text(encoding="utf-8")
    for k, v in (
        ("{{SUBPATH_REDIRECT_BLOCK}}", redirect_block),
        ("{{SUBPATH}}", subpath),
        ("{{HTTP_PORT}}", str(cfg.http_port)),
        ("{{WS_PORT}}", str(cfg.ws_port)),
        ("{{MEDIA_DIR}}", cfg.media_dir),
        ("{{HTTP_LOC_MOD}}", http_loc_mod),
        ("{{FORWARDED_PREFIX_LINE}}", forwarded_prefix_line),
    ):
        locations = locations.replace(k, v)

    if not cfg.nginx_vhost_file:
        indented = "\n".join(f"    {line}" if line else "" for line in locations.splitlines())
        return (
            "server {\n"
            "    listen 80;\n"
            "    listen [::]:80;\n"
            f"    server_name {cfg.domain};\n\n"
            f"{indented}\n"
            "}\n"
        )
    return locations


def _render_systemd_units(cfg: DeployConfig) -> tuple[str, str]:
    remote_path = f"{cfg.src_dir}/.venv/bin:/root/.local/bin:/root/.nimble/bin:/usr/local/bin:/usr/bin:/bin"
    ws_unit = (
        f"[Unit]\nDescription={cfg.app_name} Nim Mummy WebSocket + CPU bridge\nAfter=network.target\n\n"
        f"[Service]\nType=simple\nWorkingDirectory={cfg.src_dir}\nEnvironmentFile={cfg.env_file}\n"
        f"Environment=PATH={remote_path}\n"
        f"ExecStart={cfg.src_dir}/.venv/bin/python -m direct_evolution_together.ws_server\n"
        "Restart=always\nRestartSec=3\n\n[Install]\nWantedBy=multi-user.target\n"
    )
    uvicorn_unit = (
        f"[Unit]\nDescription={cfg.app_name} Uvicorn HTTP server\nAfter=network.target {cfg.ws_service}.service\n\n"
        f"[Service]\nType=simple\nWorkingDirectory={cfg.src_dir}\nEnvironmentFile={cfg.env_file}\n"
        f"Environment=PATH={remote_path}\n"
        f"ExecStart={cfg.src_dir}/.venv/bin/uvicorn direct_evolution_together.asgi:application "
        f"--host 127.0.0.1 --port {cfg.http_port} --workers {cfg.uvicorn_workers} --proxy-headers --forwarded-allow-ips=127.0.0.1\n"
        "Restart=always\nRestartSec=3\n\n[Install]\nWantedBy=multi-user.target\n"
    )
    return ws_unit, uvicorn_unit


def _bash_bootstrap_snippet(cfg: DeployConfig) -> str:
    return """
missing=""
for bin_pkg in sqlite3:sqlite3 gcc:gcc rsync:rsync curl:curl; do
  bin=${bin_pkg%%:*}; pkg=${bin_pkg##*:}
  command -v "$bin" >/dev/null 2>&1 || missing="$missing $pkg"
done
if [ -n "$missing" ]; then
  apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y $missing libsqlite3-dev
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if ! uv python find __PYTHON_VERSION__ >/dev/null 2>&1; then
  uv python install __PYTHON_VERSION__
fi
mummy_ok=0
for d in "$HOME"/.nimble/pkgs2/mummy-0.[45]*; do
  [ -d "$d" ] && mummy_ok=1 && break
done
nimpy_ok=0
for d in "$HOME"/.nimble/pkgs2/nimpy-*; do
  [ -d "$d" ] && nimpy_ok=1 && break
done
if [ "$mummy_ok" -eq 0 ] || [ "$nimpy_ok" -eq 0 ]; then
  nimble refresh -y && nimble install -y mummy nimpy
fi
""".replace("__PYTHON_VERSION__", shlex.quote(cfg.python_version)).strip()


def _bash_write_if_changed_fn() -> str:
    return """
write_if_changed() {
  local target="$1" mode="$2" b64="$3" tmp
  tmp=$(mktemp)
  printf '%s' "$b64" | base64 -d > "$tmp"
  if [ -f "$target" ] && cmp -s "$tmp" "$target"; then
    rm -f "$tmp"
    [ -n "$mode" ] && chmod "$mode" "$target"
    return 1
  fi
  mkdir -p "$(dirname "$target")"
  mv "$tmp" "$target"
  [ -n "$mode" ] && chmod "$mode" "$target"
  return 0
}
""".strip()


def _bash_nginx_snippet(cfg: DeployConfig) -> str:
    nginx_b64 = _b64(_render_nginx_conf(cfg))
    if cfg.nginx_vhost_file:
        vhost_name = Path(cfg.nginx_vhost_file).name
        backup_path = f"{cfg.run_dir}/{vhost_name}.bak"
        py_patch = """
import pathlib, re, sys
p = pathlib.Path(sys.argv[1])
snippet = sys.argv[2]
app = sys.argv[3]
text = p.read_text()
begin_mark = f"# BEGIN {app}"
end_mark = f"# END {app}"
block = f"\\t{begin_mark}\\n\\tinclude {snippet};\\n\\t{end_mark}"
pattern = re.compile(rf"[ \\t]*{re.escape(begin_mark)}.*?{re.escape(end_mark)}", re.DOTALL)
if pattern.search(text):
    new_text = pattern.sub(block, text)
else:
    loc_match = re.search(r"^([ \\t]*)location\\s+/\\s*\\{", text, re.MULTILINE)
    if not loc_match:
        raise SystemExit("Could not find location / { block in nginx vhost")
    idx = loc_match.start()
    new_text = text[:idx] + block + "\\n\\n" + text[idx:]
if new_text != text:
    p.write_text(new_text)
    sys.exit(10)
""".strip()
        return f"""
nginx_changed=0
if write_if_changed {shlex.quote(cfg.nginx_conf)} 644 {shlex.quote(nginx_b64)}; then
  nginx_changed=1
fi
mkdir -p /etc/nginx/snippets
snippet_link=/etc/nginx/snippets/{shlex.quote(cfg.app_name)}.conf
if [ "$(readlink -f "$snippet_link" 2>/dev/null || true)" != {shlex.quote(cfg.nginx_conf)} ]; then
  ln -sfn {shlex.quote(cfg.nginx_conf)} "$snippet_link"
  nginx_changed=1
fi
cp -p {shlex.quote(cfg.nginx_vhost_file)} {shlex.quote(backup_path)}
set +e
python3 -c {shlex.quote(py_patch)} {shlex.quote(cfg.nginx_vhost_file)} "$snippet_link" {shlex.quote(cfg.app_name)}
patch_rc=$?
set -e
if [ "$patch_rc" -eq 10 ]; then
  nginx_changed=1
elif [ "$patch_rc" -ne 0 ]; then
  exit "$patch_rc"
fi
if [ "$nginx_changed" -eq 1 ]; then
  if ! nginx -t; then
    cp -p {shlex.quote(backup_path)} {shlex.quote(cfg.nginx_vhost_file)}
    exit 1
  fi
  systemctl reload nginx
fi
""".strip()

    avail = f"/etc/nginx/sites-available/{cfg.app_name}"
    enabled = f"/etc/nginx/sites-enabled/{cfg.app_name}"
    return f"""
nginx_changed=0
if write_if_changed {shlex.quote(cfg.nginx_conf)} 644 {shlex.quote(nginx_b64)}; then
  nginx_changed=1
fi
mkdir -p /etc/nginx/sites-available /etc/nginx/sites-enabled
if [ "$(readlink -f {shlex.quote(avail)} 2>/dev/null || true)" != {shlex.quote(cfg.nginx_conf)} ]; then
  ln -sfn {shlex.quote(cfg.nginx_conf)} {shlex.quote(avail)}
  nginx_changed=1
fi
if [ "$(readlink -f {shlex.quote(enabled)} 2>/dev/null || true)" != {shlex.quote(cfg.nginx_conf)} ]; then
  ln -sfn {shlex.quote(avail)} {shlex.quote(enabled)}
  nginx_changed=1
fi
if [ "$nginx_changed" -eq 1 ]; then
  nginx -t && systemctl reload nginx
fi
""".strip()


def _bash_systemd_snippet(cfg: DeployConfig) -> str:
    ws_unit, uvicorn_unit = _render_systemd_units(cfg)
    ws_b64 = _b64(ws_unit)
    uv_b64 = _b64(uvicorn_unit)
    ws_path = f"/etc/systemd/system/{cfg.ws_service}.service"
    uv_path = f"/etc/systemd/system/{cfg.uvicorn_service}.service"
    return f"""
unit_changed=0
if write_if_changed {shlex.quote(ws_path)} 644 {shlex.quote(ws_b64)}; then
  unit_changed=1
fi
if write_if_changed {shlex.quote(uv_path)} 644 {shlex.quote(uv_b64)}; then
  unit_changed=1
fi
if [ "$unit_changed" -eq 1 ]; then
  systemctl daemon-reload
  systemctl enable {shlex.quote(cfg.ws_service)} {shlex.quote(cfg.uvicorn_service)}
fi
""".strip()


def _bash_supervisor_setup_snippet(cfg: DeployConfig) -> str:
    if cfg.process_manager == "pm2":
        eco_b64 = _b64(_render_pm2_ecosystem(cfg))
        return "\n".join([
            "if ! command -v pm2 >/dev/null 2>&1; then",
            '  echo "ERROR: pm2 not found on remote PATH (set DEPLOY_PROCESS_MANAGER=systemd or install pm2)" >&2',
            "  exit 1",
            "fi",
            "if command -v systemctl >/dev/null 2>&1; then",
            f"  systemctl disable --now {shlex.quote(cfg.ws_service)} {shlex.quote(cfg.uvicorn_service)} 2>/dev/null || true",
            "fi",
            f"write_if_changed {shlex.quote(cfg.ecosystem_file)} 600 {shlex.quote(eco_b64)} || true",
        ])
    return _bash_systemd_snippet(cfg)


def _bash_supervisor_restart_snippet(cfg: DeployConfig, write_config: bool = True) -> str:
    if cfg.process_manager == "pm2":
        lines = [
            "if ! command -v pm2 >/dev/null 2>&1; then",
            '  echo "ERROR: pm2 not found on remote PATH (set DEPLOY_PROCESS_MANAGER=systemd or install pm2)" >&2',
            "  exit 1",
            "fi",
            "if command -v systemctl >/dev/null 2>&1; then",
            f"  systemctl disable --now {shlex.quote(cfg.ws_service)} {shlex.quote(cfg.uvicorn_service)} 2>/dev/null || true",
            "fi",
        ]
        if write_config:
            eco_b64 = _b64(_render_pm2_ecosystem(cfg))
            lines.append(f"write_if_changed {shlex.quote(cfg.ecosystem_file)} 600 {shlex.quote(eco_b64)} || true")
        if cfg.app_name not in (cfg.ws_service, cfg.uvicorn_service):
            lines.append(f"pm2 delete {shlex.quote(cfg.app_name)} >/dev/null 2>&1 || true")
        lines.extend([
            f"if [ -f {shlex.quote(cfg.ecosystem_file)} ]; then",
            f"  pm2 startOrRestart {shlex.quote(cfg.ecosystem_file)} --update-env",
            "else",
            f"  pm2 restart {shlex.quote(cfg.ws_service)} {shlex.quote(cfg.uvicorn_service)} --update-env",
            "fi",
            "pm2 save >/dev/null 2>&1 || true",
        ])
        return "\n".join(lines)

    return "\n".join([
        *(
            [
                "if command -v pm2 >/dev/null 2>&1; then",
                f"  pm2 delete {shlex.quote(cfg.ws_service)} {shlex.quote(cfg.uvicorn_service)} >/dev/null 2>&1 || true",
                "fi",
            ]
            if write_config
            else []
        ),
        f"systemctl restart {shlex.quote(cfg.ws_service)} {shlex.quote(cfg.uvicorn_service)}",
    ])


def _run_remote_bash(conn: Connection, script: str) -> None:
    conn.run(f"printf '%s' {shlex.quote(_b64(script))} | base64 -d | bash -se")


def _bash_health_check_snippet(cfg: DeployConfig) -> str:
    ws_url = shlex.quote(f"http://127.0.0.1:{cfg.ws_port}/health")
    http_url = shlex.quote(f"http://127.0.0.1:{cfg.http_port}/health/")
    return "\n".join([
        f"for _i in $(seq 1 50); do "
        f"if ws_out=$(curl -fsS {ws_url} 2>/dev/null); then echo \"ws: $ws_out\"; break; fi; "
        f"[ \"$_i\" -eq 50 ] && curl -fsS {ws_url}; sleep 0.2; "
        f"done",
        f"for _i in $(seq 1 50); do "
        f"if out=$(curl -fsS {http_url} 2>/dev/null); then echo \"http: $out\"; break; fi; "
        f"[ \"$_i\" -eq 50 ] && curl -fsS {http_url}; sleep 0.2; "
        f"done",
    ])


def _sync_source(cfg: DeployConfig) -> None:
    ssh_target = f"{cfg.user}@{cfg.host}" if cfg.user else cfg.host
    mkdir_cmd = f"mkdir -p {shlex.quote(cfg.src_dir)} {shlex.quote(cfg.data_dir)} {shlex.quote(cfg.media_dir)} {shlex.quote(cfg.run_dir)} && rsync"
    ssh_cmd = "ssh -o ControlMaster=auto -o ControlPath=/tmp/det-ssh-%C -o ControlPersist=60"
    cmd = [
        "rsync",
        "-az",
        "--delete",
        "--no-owner",
        "--no-group",
        "--omit-dir-times",
        f"--rsync-path={mkdir_cmd}",
        "-e",
        ssh_cmd,
        "--exclude=.git/",
        "--exclude=.venv/",
        "--exclude=.env",
        "--exclude=.env.*",
        "--exclude=ecosystem.config.*",
        "--exclude=pm2*.config.*",
        "--exclude=pm2*.json",
        "--exclude=justfile",
        "--exclude=uv.lock",
        "--exclude=dist/",
        "--exclude=build/",
        "--exclude=__pycache__/",
        "--exclude=*.pyc",
        "--exclude=*.so",
        "--exclude=*.sqlite3",
        "--exclude=*.sqlite3-*",
        "--exclude=media/",
        "--exclude=nimcache/",
        "--exclude=direct_evolution_together/nimble.paths",
        f"{REPO_ROOT}/",
        f"{ssh_target}:{cfg.src_dir}/",
    ]
    subprocess.run(cmd, check=True)


def _ensure_media_dir_cmd(cfg: DeployConfig) -> str:
    return (
        f"mkdir -p {shlex.quote(cfg.media_dir)} && "
        f"chmod 755 {shlex.quote(cfg.deploy_dir)} {shlex.quote(cfg.data_dir)} {shlex.quote(cfg.media_dir)}"
    )


@task
def setup(c):
    """Bootstrap remote OS packages (sqlite3, uv, nim deps), directories, nginx, and process supervisor."""
    cfg = DeployConfig.from_env()
    script = "\n".join([
        "set -euo pipefail",
        _bash_remote_path_export(),
        f"mkdir -p {shlex.quote(cfg.src_dir)} {shlex.quote(cfg.data_dir)} {shlex.quote(cfg.run_dir)}",
        _ensure_media_dir_cmd(cfg),
        _bash_write_if_changed_fn(),
        _bash_bootstrap_snippet(cfg),
        _bash_supervisor_setup_snippet(cfg),
        _bash_nginx_snippet(cfg),
    ]) + "\n"
    with cfg.connect() as conn:
        _run_remote_bash(conn, script)


@task
def deploy(c, force_deps: bool = False):
    """Sync code, install CPU-only deps if changed, compile Nim if changed, migrate, update configs, and restart."""
    cfg = DeployConfig.from_env()
    _sync_source(cfg)

    env_b64 = _b64(_render_runtime_env(cfg))
    force_flag = "1" if force_deps else "0"
    py_tag = cfg.python_version.replace(".", "")
    deps_hash_file = f"{cfg.deploy_dir}/.deps.hash"
    nim_hash_file = f"{cfg.deploy_dir}/.nim.hash"
    nim_so_file = f"{cfg.deploy_dir}/.nim.so"

    script = "\n".join([
        "set -euo pipefail",
        _bash_remote_path_export(),
        _ensure_media_dir_cmd(cfg),
        _bash_write_if_changed_fn(),
        _bash_bootstrap_snippet(cfg),
        f"write_if_changed {shlex.quote(cfg.env_file)} 600 {shlex.quote(env_b64)} || true",
        f"cd {shlex.quote(cfg.src_dir)}",
        f"want_deps=$(sha256sum pyproject.toml | awk '{{print $1}}')-cpu-py{py_tag}",
        f'have_deps=$(cat {shlex.quote(deps_hash_file)} 2>/dev/null || true)',
        f'if [ {force_flag} -eq 1 ] || [ "$want_deps" != "$have_deps" ] || [ ! -x .venv/bin/uvicorn ] || ! .venv/bin/python -c "import torch; assert torch.version.cuda is None" 2>/dev/null; then',
        f"  uv venv --clear --python {shlex.quote(cfg.python_version)} .venv",
        '  uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu "torch>=2.5"',
        "  uv pip install --python .venv/bin/python -e .",
        '  printf "%s" "$want_deps" > ' + shlex.quote(deps_hash_file),
        "else",
        '  echo "deps: cached (cpu)"',
        "fi",
        f"want_nim=$(sha256sum direct_evolution_together/mummy_ws.nim | awk '{{print $1}}')-py{py_tag}",
        f'have_nim=$(cat {shlex.quote(nim_hash_file)} 2>/dev/null || true)',
        f'built_so=$(cat {shlex.quote(nim_so_file)} 2>/dev/null || true)',
        'if [ "$want_nim" != "$have_nim" ] || [ -z "$built_so" ] || [ ! -f "$built_so" ]; then',
        "  rm -f direct_evolution_together/__pycache__/mummy_ws*.so 2>/dev/null || true",
        '  built_so=$(.venv/bin/python -c "from direct_evolution_together import mummy_ws; assert hasattr(mummy_ws, \'start_server\'); print(mummy_ws.__file__)")',
        '  printf "%s" "$want_nim" > ' + shlex.quote(nim_hash_file),
        '  printf "%s" "$built_so" > ' + shlex.quote(nim_so_file),
        "else",
        '  echo "nim: cached"',
        "fi",
        f"mig_hash_file={shlex.quote(cfg.deploy_dir + '/.mig.hash')}",
        "want_mig=$(cat direct_evolution_together/migrations/*.py direct_evolution_together/models.py | sha256sum | awk '{print $1}')",
        'have_mig=$(cat "$mig_hash_file" 2>/dev/null || true)',
        f'if [ "$want_mig" != "$have_mig" ] || [ ! -f {shlex.quote(cfg.data_dir + "/db.sqlite3")} ]; then',
        f"  set -a && . {shlex.quote(cfg.env_file)} && set +a",
        "  .venv/bin/python manage.py migrate --noinput",
        '  printf "%s" "$want_mig" > "$mig_hash_file"',
        "else",
        '  echo "migrations: cached"',
        "fi",
        _bash_supervisor_setup_snippet(cfg),
        _bash_nginx_snippet(cfg),
        _bash_supervisor_restart_snippet(cfg, write_config=False),
        _bash_health_check_snippet(cfg),
    ]) + "\n"

    with cfg.connect() as conn:
        _run_remote_bash(conn, script)


@task
def reload(c):
    """Push updated .env to the server, restart services cleanly, and verify health."""
    cfg = DeployConfig.from_env()
    env_b64 = _b64(_render_runtime_env(cfg))
    script = "\n".join([
        "set -euo pipefail",
        _bash_remote_path_export(),
        _ensure_media_dir_cmd(cfg),
        _bash_write_if_changed_fn(),
        f"write_if_changed {shlex.quote(cfg.env_file)} 600 {shlex.quote(env_b64)} || true",
        _bash_supervisor_restart_snippet(cfg, write_config=True),
        _bash_health_check_snippet(cfg),
    ]) + "\n"
    with cfg.connect() as conn:
        _run_remote_bash(conn, script)


@task
def nginx(c):
    """Render, link, validate, and reload the Nginx config if changed."""
    cfg = DeployConfig.from_env()
    script = "\n".join([
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(cfg.run_dir)}",
        _ensure_media_dir_cmd(cfg),
        _bash_write_if_changed_fn(),
        _bash_nginx_snippet(cfg),
    ]) + "\n"
    with cfg.connect() as conn:
        _run_remote_bash(conn, script)


@task
def restart(c):
    """Restart the socket server and Uvicorn servers cleanly."""
    cfg = DeployConfig.from_env()
    script = "\n".join([
        "set -euo pipefail",
        _bash_remote_path_export(),
        _bash_supervisor_restart_snippet(cfg, write_config=False),
        _bash_health_check_snippet(cfg),
    ]) + "\n"
    with cfg.connect() as conn:
        _run_remote_bash(conn, script)


@task
def status(c):
    """Show PM2/systemd and local HTTP + WebSocket health status on the production server."""
    cfg = DeployConfig.from_env()
    if cfg.process_manager == "pm2":
        sup_cmd = (
            f"pm2 describe {shlex.quote(cfg.ws_service)} >/dev/null 2>&1 && pm2 status || pm2 list || true"
        )
    else:
        sup_cmd = f"systemctl --no-pager status {shlex.quote(cfg.ws_service)} {shlex.quote(cfg.uvicorn_service)} || true"
    script = "\n".join([
        "set -euo pipefail",
        _bash_remote_path_export(),
        sup_cmd,
        f"curl -fsS http://127.0.0.1:{cfg.ws_port}/health && echo ''",
        f"curl -fsS http://127.0.0.1:{cfg.http_port}/health/ && echo ''",
    ]) + "\n"
    with cfg.connect() as conn:
        _run_remote_bash(conn, script)


@task
def logs(c, lines: int = 80):
    """Show recent logs for Uvicorn and WebSocket services."""
    cfg = DeployConfig.from_env()
    n = int(lines)
    if cfg.process_manager == "pm2":
        log_cmd = f"pm2 logs {shlex.quote(cfg.ws_service + '|' + cfg.uvicorn_service)} --lines {n} --nostream"
    else:
        log_cmd = f"journalctl -u {shlex.quote(cfg.ws_service)} -u {shlex.quote(cfg.uvicorn_service)} -n {n} --no-pager"
    script = "\n".join([
        "set -euo pipefail",
        _bash_remote_path_export(),
        log_cmd,
    ]) + "\n"
    with cfg.connect() as conn:
        _run_remote_bash(conn, script)


@task
def clear(c, room: str = "lobby", goal: str = ""):
    """Clear evolution state in production for a lobby goal, the whole lobby, or a private room."""
    cfg = DeployConfig.from_env()
    payload = json.dumps({
        "password": cfg.admin_password,
        "room": (room or "lobby").strip() or "lobby",
        "goal": (goal or "").strip(),
        "action": "clear",
    })
    url = shlex.quote(f"http://127.0.0.1:{cfg.http_port}/api/clear/")
    with cfg.connect() as conn:
        conn.run(
            f"curl -fsS -X POST -H 'Content-Type: application/json' -d {shlex.quote(payload)} {url}"
        )
