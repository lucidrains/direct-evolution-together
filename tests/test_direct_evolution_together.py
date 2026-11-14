from contextlib import contextmanager
import io
import json
import os
import socket
import subprocess
import zipfile

import pytest
import torch
from websockets.sync.client import connect

from direct_evolution_together import mummy_ws, ws_bridge
from direct_evolution_together.cppn import CPPN, create_random_population, evolve_population
from direct_evolution_together.models import ChatMessage
from direct_evolution_together.ws_bridge import (
    flush_pending_messages,
    seed_from_db,
    start_ws_server,
    stop_ws_server,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _recv_until(ws, target_type: str, timeout: float = 5.0) -> dict:
    while True:
        msg = json.loads(ws.recv(timeout=timeout))
        if msg.get("type") == target_type:
            return msg


@contextmanager
def _ws_server(**kwargs):
    mummy_ws.clear_state()
    port = _free_port()
    start_ws_server(host="127.0.0.1", port=port, **kwargs)
    try:
        yield port
    finally:
        stop_ws_server()


@pytest.mark.django_db
def test_http_pages_health_and_subpath(client, settings):
    assert client.get("/health/").json() == {"status": "ok"}

    lobby_html = client.get("/").content.decode()
    assert "direct-evolution-together" in lobby_html
    assert 'id="reset-btn"' not in lobby_html
    assert "10.1145/1357054.1357328" in lobby_html

    priv = client.get("/p/p-testroom1/?pop=24&max_votes=7&sigma=0.45&elite_frac=0.20&crossover=0.60")
    assert priv.status_code == 200
    priv_html = priv.content.decode()
    assert "Start a goal" in priv_html
    assert 'id="det-initial-pop" type="application/json">24<' in priv_html
    assert 'id="det-initial-max-votes" type="application/json">7<' in priv_html
    assert 'id="det-initial-sigma" type="application/json">"0.45"<' in priv_html
    assert 'id="det-initial-elite" type="application/json">"0.20"<' in priv_html
    assert 'id="det-initial-crossover" type="application/json">"0.60"<' in priv_html

    settings.SUBPATH = "/evolver"
    settings.WS_VIA_PROXY = True
    sub_html = client.get("/p/p-demo123/").content.decode()
    assert '"/evolver"' in sub_html
    assert '<script id="det-ws-via-proxy" type="application/json">true</script>' in sub_html
    assert 'href="/evolver/"' in sub_html
    assert 'href="/evolver/"' in client.get("/reset/").content.decode()


@pytest.mark.django_db
def test_create_room_api_names_and_renders_header(client):
    resp = client.post(
        "/api/rooms/",
        data=json.dumps({"name": "Neon Orchid", "description": "Glowing flowers"}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    room = resp.json()
    assert room["slug"] == "neon-orchid"
    assert room["url"] == "/p/neon-orchid/"

    html = client.get(room["url"]).content.decode()
    assert "Neon Orchid" in html
    assert "Glowing flowers" in html

    duplicate = client.post(
        "/api/rooms/",
        data=json.dumps({"name": "Neon Orchid"}),
        content_type="application/json",
    ).json()
    assert duplicate["slug"] == "neon-orchid-2"

    reserved = client.post(
        "/api/rooms/",
        data=json.dumps({"name": "Lobby"}),
        content_type="application/json",
    ).json()
    assert reserved["slug"] != "lobby"

    assert (
        client.post(
            "/api/rooms/",
            data=json.dumps({"name": "  "}),
            content_type="application/json",
        ).status_code
        == 400
    )


def test_cppn_evolution_and_export_bundle(client, tmp_path):
    net = CPPN.random(42)
    restored = CPPN.from_dict(net.to_dict())
    coords = torch.tensor([[0.25, -0.5, 0.559, 1.0]])
    assert net.render(32).shape == (32, 32, 3)
    assert torch.allclose(net(coords), restored(coords), atol=5e-3)

    pop = create_random_population()
    assert len(pop) == 30
    scores = [-2, 0, 3, 1] + [0] * 26
    ups = [0, 0, 3, 1] + [0] * 26
    next_pop, top_score = evolve_population(pop, scores, ups, mut_sigma=0.3)
    assert top_score == 3 and len(next_pop) == 30
    assert next_pop[0]["weights"] == pop[2]["weights"] and next_pop[0]["op"] == "elite"
    assert next_pop[1]["weights"] == pop[3]["weights"] and next_pop[1]["op"] == "elite"

    resp = client.post(
        "/export/bundle/",
        data=json.dumps({"goal": "Butterfly", "generation": 2, "index": 3, "genome": net.to_dict()}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    assert 'filename="butterfly_g2_3.zip"' in resp["Content-Disposition"]
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        assert set(zf.namelist()) == {"butterfly_g2_3/weights.pt", "butterfly_g2_3/run.py"}
        zf.extractall(tmp_path)

    extracted_dir = tmp_path / "butterfly_g2_3"
    proc = subprocess.run(
        ["uv", "run", "--no-project", "run.py"],
        cwd=extracted_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Voila!" in proc.stdout
    assert (extracted_dir / "out.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.django_db(transaction=True)
def test_e2e_chat_persistence_cursor_and_evolution():
    ChatMessage.objects.bulk_create(
        [ChatMessage(room="histroom", username="seed", message=f"m-{i}") for i in range(105)]
    )
    with _ws_server(seed=True, pop_size=12, max_votes=2) as port:
        seed_from_db()
        with connect(f"ws://127.0.0.1:{port}/ws/chat/histroom/?username=viewer") as v:
            hist = _recv_until(v, "history")
            assert len(hist["messages"]) == 100
            assert hist["messages"][0]["message"] == "m-5"

        mummy_ws.set_round_config(2, 150)
        with connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=alice") as a1:
            assert _recv_until(a1, "identity")["username"] == "alice"
            evo0 = _recv_until(a1, "evolution_state")
            assert [g["name"] for g in evo0["goals"]] == ["Butterfly", "Galaxy", "Creature"]
            assert len(evo0["goals"][0]["genomes"]) == 12

            with connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=alice") as a2:
                assert _recv_until(a2, "identity")["username"] == "alice-2"
                _recv_until(a2, "evolution_state")

                # Chat persists to SQLite across connections
                a1.send(json.dumps({"type": "chat", "message": "hello e2e"}))
                assert _recv_until(a2, "chat")["message"] == "hello e2e"
                flush_pending_messages()
                assert ChatMessage.objects.filter(room="lobby", message="hello e2e").exists()

                # Cursor broadcast + rename + close cleanup
                a1.send(json.dumps({"type": "cursor", "x": 0.4, "y": 0.7, "visible": True, "goal": "Butterfly"}))
                cur = _recv_until(a2, "cursor")
                assert cur["username"] == "alice" and cur["goal"] == "Butterfly"
                a1.send(json.dumps({"type": "rename", "username": "Alice"}))
                assert _recv_until(a2, "cursor_leave")["username"] == "alice"
                a1.send(json.dumps({"type": "rename", "username": "alice"}))
                _recv_until(a1, "identity")

                # Per-user vote cap (2) + throttle + flip + unvote reclaim + countdown -> gen 1
                a1.send(json.dumps({"type": "cast_vote", "goal": "Butterfly", "index": 0, "vote": 1}))
                assert _recv_until(a1, "votes_updated")["totalVotes"] == 1
                a1.send(json.dumps({"type": "cast_vote", "goal": "Butterfly", "index": 1, "vote": 1}))
                _recv_until(a1, "votes_updated")
                a1.send(json.dumps({"type": "cast_vote", "goal": "Butterfly", "index": 2, "vote": 1}))
                assert _recv_until(a1, "vote_throttled")["maxVotesPerUser"] == 2
                a1.send(json.dumps({"type": "cast_vote", "goal": "Butterfly", "index": 0, "vote": 1}))
                _recv_until(a1, "votes_updated")

                a2.send(json.dumps({"type": "cast_vote", "goal": "Butterfly", "index": 2, "vote": 1}))
                upd = _recv_until(a1, "goal_updated", timeout=5.0)
                assert upd["goal"]["generation"] == 1
                assert len(upd["goal"]["lineage"]) == 24


@pytest.mark.django_db(transaction=True)
def test_e2e_private_room_ga_and_admin_reset(client, monkeypatch, settings):
    with _ws_server(seed=False, pop_size=12, max_votes=3, admin_password="top-secret-pw") as port:
        mummy_ws.set_round_config(1, 150)
        settings.WS_HOST = "127.0.0.1"
        settings.WS_PORT = port

        # Private room accepts custom pop & GA params while lobby ignores create_goal & reset_goal
        with (
            connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=pub") as pub,
            connect(
                f"ws://127.0.0.1:{port}/ws/chat/p-priv1/?username=bob&goal=Lotus&pop=8&max_votes=4&sigma=0.50&elite_frac=0.25&crossover=0.40"
            ) as priv,
        ):
            _recv_until(pub, "evolution_state")
            p_evo = _recv_until(priv, "evolution_state")
            lotus = p_evo["goals"][0]
            assert lotus["name"] == "Lotus" and lotus["populationSize"] == 8
            assert lotus["maxVotesPerUser"] == 4
            assert abs(lotus["mutationSigma"] - 0.50) < 1e-6
            assert abs(lotus["eliteFrac"] - 0.25) < 1e-6
            assert abs(lotus["crossoverRate"] - 0.40) < 1e-6

            priv.send(json.dumps({
                "type": "create_goal", "goal": "Beetle", "population_size": 6,
                "max_votes_per_user": 5, "mutation_sigma": 0.25, "elite_frac": 0.20, "crossover_rate": 0.90,
            }))
            beetle = {g["name"]: g for g in _recv_until(priv, "evolution_state")["goals"]}["Beetle"]
            assert beetle["populationSize"] == 6 and beetle["maxVotesPerUser"] == 5

            # Named private rooms (no p- prefix) start empty and accept goals
            with connect(f"ws://127.0.0.1:{port}/ws/chat/named-room/?username=carol") as named:
                assert _recv_until(named, "evolution_state")["goals"] == []
                named.send(json.dumps({"type": "create_goal", "goal": "Fern", "population_size": 4}))
                assert [g["name"] for g in _recv_until(named, "evolution_state")["goals"]] == ["Fern"]

            # Evolve lobby Butterfly -> gen 1
            pub.send(json.dumps({"type": "cast_vote", "goal": "Butterfly", "index": 0, "vote": 1}))
            assert _recv_until(pub, "goal_updated", timeout=5.0)["goal"]["generation"] == 1

            # Unauthenticated WS reset_goal is ignored
            pub.send(json.dumps({"type": "reset_goal", "goal": "Butterfly"}))
            with pytest.raises(TimeoutError):
                _recv_until(pub, "goal_updated", timeout=0.15)

            # Admin reset requires DET_ADMIN_PASSWORD (503 unset, 403 bad, 200 valid + IPC)
            monkeypatch.delenv("DET_ADMIN_PASSWORD", raising=False)
            assert client.post("/api/reset/", data=json.dumps({"password": "x"}), content_type="application/json").status_code == 503

            monkeypatch.setenv("DET_ADMIN_PASSWORD", "top-secret-pw")
            assert client.post("/api/reset/", data=json.dumps({"password": "bad"}), content_type="application/json").status_code == 403
            assert client.post("/reset/", {"room": "lobby", "goal": "Butterfly", "password": "bad"}).status_code == 403

            assert client.post(
                "/api/reset/",
                data=json.dumps({"password": "top-secret-pw", "room": "lobby", "goal": "Butterfly"}),
                content_type="application/json",
            ).status_code == 200
            assert _recv_until(pub, "goal_updated", timeout=3.0)["goal"]["generation"] == 0

            # Also verify separate uvicorn -> Mummy loopback HTTP bridge path
            monkeypatch.setattr(ws_bridge, "is_running", lambda: False)
            assert client.post(
                "/api/reset/",
                data=json.dumps({"password": "top-secret-pw", "room": "p-priv1", "goal": "Lotus"}),
                content_type="application/json",
            ).status_code == 200
            assert _recv_until(priv, "goal_updated", timeout=3.0)["goal"]["generation"] == 0


def test_cli_and_dotenv(tmp_path, monkeypatch):
    from direct_evolution_together.cli import check_nim_toolchain, open_browser_when_ready
    from direct_evolution_together.settings import load_dotenv

    assert check_nim_toolchain()[0] is True
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url, new=0: opened.append((url, new)) or True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        t = open_browser_when_ready("0.0.0.0", srv.getsockname()[1], timeout=2.0)
        t.join(timeout=2.0)
    assert len(opened) == 1

    (tmp_path / ".env").write_text('# c\nDET_TEST_K="from_file"\nDET_TEST_E="no"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DET_TEST_K", raising=False)
    monkeypatch.setenv("DET_TEST_E", "yes")
    load_dotenv()
    assert os.environ.pop("DET_TEST_K", None) == "from_file"
    assert os.environ["DET_TEST_E"] == "yes"
