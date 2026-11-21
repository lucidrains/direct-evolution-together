"""Standalone export scripts for trained policies and external researchers."""

from __future__ import annotations

from direct_evolution_together.cppn import build_uv_script_header

from .adapters import TASK_ADAPTERS
from .registry import get_task_spec, sanitize_task_env_params


def build_control_runner_script(
    task_id: str,
    weights_filename: str = "weights.pt",
    env_params: dict[str, object] | None = None,
) -> str:
    """Return a standalone uv script that evaluates an exported policy on a control task."""
    spec = get_task_spec(task_id)
    if spec.genome_kind == "populora":
        clean_env = sanitize_task_env_params(spec.task_id, env_params) if env_params else {}
        user_overrides = {
            str(h["key"]): clean_env[str(h["key"])]
            for h in spec.hyperparam_specs
            if h.get("key") and str(h["key"]) in clean_env
        }
        env_line = f"ENV_PARAMS = {user_overrides!r}" if user_overrides else "ENV_PARAMS = None"
        return "\n".join(
            [
                *build_uv_script_header(extra_deps=("populora",)),
                "import sys",
                "import torch",
                "from direct_evolution_together.environments import evaluate_populora_genome, get_task_spec",
                "from direct_evolution_together.populora_engine import build_policy_module, load_populora_checkpoint_bytes",
                "",
                f'spec = get_task_spec("{spec.task_id}")',
                env_line,
                "model = build_policy_module(spec.policy_spec)",
                f'ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "{weights_filename}"',
                'raw_bytes = open(ckpt_path, "rb").read()',
                'state = torch.load(ckpt_path, map_location="cpu", weights_only=True)',
                "model.load_state_dict(state, strict=False)",
                f'genome = load_populora_checkpoint_bytes(raw_bytes, spec.policy_spec, task_id="{spec.task_id}", assign_id=False)',
                f'ev = evaluate_populora_genome("{spec.task_id}", genome, env_params=ENV_PARAMS, include_replay=False)',
                f'print("Loaded PopuLoRA policy for {spec.task_id}:", sum(p.numel() for p in model.parameters()), "params ->", ev["summary"]["primary_display"])',
                "",
            ]
        )
    if spec.task_id not in TASK_ADAPTERS:
        return "\n".join(
            [
                *build_uv_script_header(),
                "import sys",
                "from direct_evolution_together.cppn import CPPN",
                "",
                f'net = CPPN.init_and_load(sys.argv[1] if len(sys.argv) > 1 else "{weights_filename}")',
                'print("Loaded CPPN genome:", net.genome_id, "layers:", [tuple(l.weight.shape) for l in net.layers])',
                "",
            ]
        )

    return "\n".join(
        [
            *build_uv_script_header(extra_deps=("imageio-ffmpeg",)),
            "import sys",
            "from pathlib import Path",
            "from direct_evolution_together.cppn import CPPN",
            "from direct_evolution_together.environments import evaluate_control_genome, render_task_media",
            "",
            f'net = CPPN.init_and_load(sys.argv[1] if len(sys.argv) > 1 else "{weights_filename}")',
            f'summary = evaluate_control_genome("{spec.task_id}", net, seeds=tuple(range(10)))',
            'print(f"Evaluated 10 held-out episodes: primary={summary[\'primary_display\']}, solved={summary[\'solved\']} ({summary[\'secondary_text\']})")',
            f'media = render_task_media("{spec.task_id}", net.to_dict(), seed=0, media_root=Path("."))',
            'Path("out.mp4").write_bytes((Path(".") / media["mp4_rel"]).read_bytes())',
            'print("Voila! Saved out.mp4")',
            "",
        ]
    )


def build_worker_script(
    room_slug: str,
    server_url: str,
    worker_token: str = "<YOUR_WORKER_TOKEN>",
) -> str:
    """Return a standalone PEP 723 uv worker script so researchers can plug in any custom simulator."""
    clean_base = server_url.rstrip("/")
    return "\n".join(
        [
            "# /// script",
            '# dependencies = ["numpy", "torch", "pillow", "direct-evolution-together"]',
            "# ///",
            '"""External simulation worker for direct-evolution-together.',
            "",
            "Run your own MuJoCo / Gymnasium / Microduck / custom simulator locally and stream",
            "per-genome telemetry, 2D/3D pose frames, or rendered PNG/MP4 previews to the room.",
            '"""',
            "import base64",
            "import io",
            "import json",
            "import time",
            "import urllib.request",
            "import numpy as np",
            "from PIL import Image, ImageDraw",
            "from direct_evolution_together.environments import HyperNEATSubstrate",
            "",
            f'BASE_URL = "{clean_base}"',
            f'ROOM_SLUG = "{room_slug}"',
            f'WORKER_TOKEN = "{worker_token}"',
            "SUBSTRATE = HyperNEATSubstrate((8, 12, 4))  # Customize obs_dim -> hidden -> act_dim",
            "",
            "def _req(method, path, payload=None):",
            "    data = json.dumps(payload).encode('utf-8') if payload is not None else None",
            "    headers = {'X-Worker-Token': WORKER_TOKEN}",
            "    if data is not None:",
            "        headers['Content-Type'] = 'application/json'",
            "    req = urllib.request.Request(f'{BASE_URL}{path}', data=data, headers=headers, method=method)",
            "    with urllib.request.urlopen(req, timeout=15) as resp:",
            "        return json.loads(resp.read().decode('utf-8'))",
            "",
            "def evaluate_genome(genome: dict) -> dict:",
            "    # Replace this body with your own simulator (e.g. MicroduckGaitEnv, Gymnasium, etc.)",
            "    weights, biases = SUBSTRATE.decode(genome)",
            "    obs = np.zeros(SUBSTRATE.dims[0], dtype=np.float64)",
            "    act = SUBSTRATE.forward(weights, biases, obs)",
            "    score = float(np.sum(np.abs(act)))",
            "    img = Image.new('RGB', (160, 160), (9, 9, 11))",
            "    draw = ImageDraw.Draw(img)",
            "    draw.text((14, 70), f'score {score:.2f}', fill=(52, 211, 153))",
            "    buf = io.BytesIO()",
            "    img.save(buf, format='PNG')",
            "    return {",
            "        'summary': {",
            "            'primary_value': round(score, 2),",
            "            'badge_text': f'{score:.2f}',",
            "            'tier': 'solved' if score > 2.5 else ('mid' if score > 1.2 else ''),",
            "            'secondary_text': 'evaluated by local researcher script',",
            "        },",
            "        'poster_png_b64': base64.b64encode(buf.getvalue()).decode('ascii'),",
            "    }",
            "",
            "def main():",
            "    print(f'Worker connected to {BASE_URL}/p/{ROOM_SLUG}/')",
            "    while True:",
            "        job = _req('GET', f'/api/rooms/{ROOM_SLUG}/worker/jobs/')",
            "        pending = job.get('pending', [])",
            "        if pending:",
            "            results = []",
            "            for item in pending:",
            "                res = evaluate_genome(item['genome'])",
            "                results.append({'hash': item['hash'], **res})",
            "            _req('POST', f'/api/rooms/{ROOM_SLUG}/worker/results/', {",
            "                'goal': job['goal'],",
            "                'generation': job['generation'],",
            "                'results': results,",
            "            })",
            "            print(f\"Submitted {len(results)} evals for {job['goal']} gen {job['generation']}\")",
            "        time.sleep(1.0)",
            "",
            "if __name__ == '__main__':",
            "    main()",
            "",
        ]
    )


def build_crowd_worker_script(
    room_slug: str,
    server_url: str,
    goal_name: str = "",
    default_batch_size: int = 4,
    *,
    max_batch_size: int = 256,
    req_timeout_sec: int = 45,
    max_default_workers: int = 8,
    idle_sleep_sec: float = 0.5,
) -> str:
    """Return a standalone PEP 723 ``uv run`` script that crowdsources MuJoCo simulation shards locally."""
    clean_base = server_url.rstrip("/")
    eff_batch = max(1, min(max_batch_size, int(default_batch_size)))
    return "\n".join(
        [
            *build_uv_script_header(extra_deps=("populora",)),
            '"""Local crowdsourced MuJoCo shard simulator for direct-evolution-together.',
            "",
            "Claims disjoint batches of unevaluated individuals from a large-population room,",
            "runs native MuJoCo rollouts locally across CPU threads, ranks the shard, and posts",
            "verified fitnesses + trajectories back to advance evolution.",
            '"""',
            "import argparse",
            "import json",
            "import os",
            "import secrets",
            "import time",
            "import urllib.request",
            "from direct_evolution_together.environments import evaluate_populora_shard",
            "",
            f'DEFAULT_URL = "{clean_base}"',
            f'DEFAULT_ROOM = "{room_slug}"',
            f'DEFAULT_GOAL = "{goal_name}"',
            "BASE_MODEL_CACHE: dict[tuple[str, str], dict] = {}",
            "",
            f"def _json_req(method: str, url: str, payload: dict | None = None, peer_id: str = '', timeout: int = {int(req_timeout_sec)}) -> dict:",
            "    data = json.dumps(payload).encode('utf-8') if payload is not None else None",
            "    headers = {'Accept': 'application/json'}",
            "    if peer_id:",
            "        headers['X-Crowd-Peer'] = peer_id",
            "    if data is not None:",
            "        headers['Content-Type'] = 'application/json'",
            "    req = urllib.request.Request(url, data=data, headers=headers, method=method)",
            "    with urllib.request.urlopen(req, timeout=timeout) as resp:",
            "        return json.loads(resp.read().decode('utf-8'))",
            "",
            "def run_once(base_url: str, room: str, goal: str, batch_size: int, peer_id: str, workers: int = 1) -> int:",
            "    claim = _json_req(",
            "        'POST',",
            "        f'{base_url}/api/rooms/{room}/crowd/claim/',",
            "        {'goal': goal, 'batch_size': batch_size, 'peer_id': peer_id},",
            "        peer_id=peer_id,",
            "    )",
            "    shard = claim.get('shard') or []",
            "    if not shard:",
            "        return 0",
            "    task_id = claim['task']",
            "    env_params = claim.get('env_params') or {}",
            "    seeds = claim.get('eval_seeds') or [0, 1]",
            "    base_hash = claim.get('base_hash') or ''",
            "    base_model = BASE_MODEL_CACHE.get((task_id, base_hash)) if base_hash else None",
            "    if base_hash and base_model is None:",
            "        bm_resp = _json_req('GET', f'{base_url}/api/sim/base-model/?task={task_id}&hash={base_hash}', peer_id=peer_id)",
            "        if bm_resp.get('tensors'):",
            "            base_model = bm_resp",
            "            BASE_MODEL_CACHE[(task_id, base_hash)] = base_model",
            "    evaluated = evaluate_populora_shard(",
            "        task_id,",
            "        shard,",
            "        seeds=seeds,",
            "        env_params=env_params,",
            "        base_model_payload=base_model,",
            "        include_replay=True,",
            "        max_workers=workers,",
            "    )",
            "    res = _json_req(",
            "        'POST',",
            "        f'{base_url}/api/rooms/{room}/crowd/results/',",
            "        {",
            "            'goal': claim['goal'],",
            "            'generation': claim['generation'],",
            "            'peer_id': peer_id,",
            "            'sharded': True,",
            "            'include_rankings': False,",
            "            'results': evaluated['results'],",
            "            'shard_ranking': evaluated['shard_ranking'],",
            "        },",
            "        peer_id=peer_id,",
            "    )",
            "    top = evaluated['shard_ranking'][0] if evaluated['shard_ranking'] else {}",
            "    print(",
            "        f\"[{claim['goal']} gen {claim['generation']}] simulated {len(shard)} individuals \"",
            "        f\"(best #{top.get('uuid', '?')} = {top.get('primary_value', 0.0):.1f}) -> \"",
            "        f\"{res.get('completed_count', 0)}/{res.get('total_count', 0)} ready\"",
            "    )",
            "    return len(shard)",
            "",
            f"def main(max_default_workers: int = {int(max_default_workers)}, idle_sleep_sec: float = {float(idle_sleep_sec)}) -> None:",
            "    default_workers = max(1, min(max_default_workers, os.cpu_count() or 1))",
            "    ap = argparse.ArgumentParser(description='Crowdsourced local MuJoCo shard runner')",
            "    ap.add_argument('--url', default=DEFAULT_URL)",
            "    ap.add_argument('--room', default=DEFAULT_ROOM)",
            "    ap.add_argument('--goal', default=DEFAULT_GOAL)",
            f"    ap.add_argument('--batch-size', type=int, default={eff_batch})",
            "    ap.add_argument('--workers', type=int, default=default_workers, help='Parallel MuJoCo worker threads')",
            "    ap.add_argument('--once', action='store_true', help='Drain current generation shards once and exit')",
            "    args = ap.parse_args()",
            "    peer_id = f'pyworker-{secrets.token_hex(6)}'",
            "    print(f'Crowd runner {peer_id} ({args.workers} workers) connected to {args.url}/p/{args.room}/')",
            "    while True:",
            "        n = run_once(args.url.rstrip('/'), args.room, args.goal, args.batch_size, peer_id, workers=args.workers)",
            "        if args.once and n == 0:",
            "            break",
            "        if n == 0:",
            "            time.sleep(idle_sleep_sec)",
            "",
            "if __name__ == '__main__':",
            "    main()",
            "",
        ]
    )
