"""Standalone export scripts for trained policies and external researchers."""

from __future__ import annotations

from direct_evolution_together.cppn import build_uv_script_header

from .adapters import TASK_ADAPTERS
from .registry import get_task_spec


def build_control_runner_script(task_id: str, weights_filename: str = "weights.pt") -> str:
    """Return a standalone uv script that evaluates an exported policy on a control task."""
    spec = get_task_spec(task_id)
    if spec.genome_kind == "populora":
        return "\n".join(
            [
                *build_uv_script_header(extra_deps=("populora",)),
                "import sys",
                "import torch",
                "from direct_evolution_together.environments import get_task_spec",
                "from direct_evolution_together.populora_engine import build_policy_module",
                "",
                f'spec = get_task_spec("{spec.task_id}")',
                "model = build_policy_module(spec.policy_spec)",
                f'state = torch.load(sys.argv[1] if len(sys.argv) > 1 else "{weights_filename}", map_location="cpu", weights_only=True)',
                "model.load_state_dict(state, strict=False)",
                f'print("Loaded PopuLoRA policy for {spec.task_id}:", sum(p.numel() for p in model.parameters()), "params")',
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
            'print(f"Evaluated 10 held-out episodes: primary={summary[\'primaryDisplay\']}, solved={summary[\'solved\']} ({summary[\'secondaryText\']})")',
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
            "            'primaryValue': round(score, 2),",
            "            'badgeText': f'{score:.2f}',",
            "            'tier': 'solved' if score > 2.5 else ('mid' if score > 1.2 else ''),",
            "            'secondaryText': 'evaluated by local researcher script',",
            "        },",
            "        'posterPngB64': base64.b64encode(buf.getvalue()).decode('ascii'),",
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
