from contextlib import contextmanager
import gzip
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


def _decode_ws_message(raw):
    if isinstance(raw, bytes):
        raw = gzip.decompress(raw).decode("utf-8")
    return json.loads(raw)


def _recv_until(ws, target_type: str, timeout: float = 5.0) -> dict:
    while True:
        msg = _decode_ws_message(ws.recv(timeout=timeout))
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
                assert upd["goal"]["lineageTotal"] == 24
                assert len(upd["goal"]["lineage"]) < 24

                a1.send(json.dumps({"type": "lineage_all", "goal": "Butterfly"}))
                full = _recv_until(a1, "lineage_all")
                assert len(full["lineage"]) == 24


def test_ws_gzip_state_and_lazy_lineage():
    with _ws_server(seed=False, pop_size=6) as port:
        with connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=gz&gzip=1") as ws:
            saw_binary = False
            while True:
                raw = ws.recv(timeout=10)
                if isinstance(raw, bytes):
                    saw_binary = True
                msg = _decode_ws_message(raw)
                if msg.get("type") == "evolution_state":
                    break
            assert saw_binary

            goal = msg["goals"][0]
            assert goal["lineageTotal"] == len(goal["lineage"]) == 6
            assert all("weights" not in node for node in goal["lineage"])

            ws.send(json.dumps({"type": "lineage_all", "goal": goal["name"]}))
            full = _recv_until(ws, "lineage_all")
            assert full["goal"] == goal["name"]
            assert full["lineageTotal"] == len(full["lineage"]) == 6
            assert all("weights" in node for node in full["lineage"])


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


def test_cppn_no_regression_on_ties_and_empty_likes():
    torch.manual_seed(7)
    pop = create_random_population(pop_size=12)
    parent_ids = {g["id"] for g in pop}

    # 4 candidates tied at top score (+1, 1 up) even though elite_frac=0.20 -> target_elites=2
    scores = [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    ups = [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    next_pop, top_score = evolve_population(pop, scores, ups, pop_size=12, elite_frac=0.20)
    assert top_score == 1 and len(next_pop) == 12
    for slot, src_idx in enumerate((0, 1, 5, 8)):
        assert next_pop[slot]["op"] == "elite"
        assert next_pop[slot]["weights"] == pop[src_idx]["weights"]
        assert next_pop[slot]["id"] != pop[src_idx]["id"]
        assert next_pop[slot]["id"] not in parent_ids

    # Empty-liked round (only downvotes + neutral zeros) must still preserve neutral elites
    neg_scores = [-2, -1, 0, 0, -1, -3, -1, -2, -1, -1, -2, -1]
    neg_ups = [0] * 12
    after_neg, neg_top = evolve_population(pop, neg_scores, neg_ups, pop_size=12, elite_frac=0.20)
    assert neg_top == 0 and len(after_neg) == 12
    assert after_neg[0]["op"] == "elite" and after_neg[0]["weights"] == pop[2]["weights"]
    assert after_neg[1]["op"] == "elite" and after_neg[1]["weights"] == pop[3]["weights"]

    # Multi-generation deterministic vision-style scorer must have 0 top-score regressions
    def vision_score(genome_dict):
        rgb = CPPN.from_dict(genome_dict).render(16).float() / 255.0
        sym = 1.0 - float(torch.abs(rgb - torch.flip(rgb, dims=[1])).mean())
        contrast = float(rgb.std())
        return sym + 0.5 * contrast

    torch.manual_seed(11)
    cur = create_random_population(pop_size=16)
    best_history = []
    for _ in range(15):
        vscores = [vision_score(g) for g in cur]
        best_now = max(vscores)
        if best_history:
            assert best_now >= best_history[-1] - 1e-6
        best_history.append(best_now)
        ranked = sorted(range(len(cur)), key=lambda i: -vscores[i])
        disc_scores = [0] * len(cur)
        disc_ups = [0] * len(cur)
        for i in ranked[:4]:
            disc_scores[i] = 1
            disc_ups[i] = 1
        for i in ranked[-6:]:
            disc_scores[i] = -1
        cur, _ = evolve_population(cur, disc_scores, disc_ups, pop_size=16)


def test_ws_voting_threshold_and_generation_guard():
    with _ws_server(seed=False, pop_size=10, max_votes=20) as port:
        with connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=alice") as ws:
            evo = _recv_until(ws, "evolution_state")
            butterfly = next(g for g in evo["goals"] if g["name"] == "Butterfly")
            assert butterfly["minVotes"] == 5
            assert butterfly["maxVotesPerUser"] == 20

            # 5 downvotes alone meet candidate count (5/10) but NOT likedCandidates >= 1
            for idx in range(5):
                ws.send(json.dumps({
                    "type": "cast_vote",
                    "goal": "Butterfly",
                    "generation": 0,
                    "index": idx,
                    "vote": -1,
                }))
                upd = _recv_until(ws, "votes_updated")
            assert upd["votedCandidates"] == 5
            assert upd["likedCandidates"] == 0
            assert upd["countdownMs"] == 0

            # Flipping 1 downvote to an upvote triggers countdown
            ws.send(json.dumps({
                "type": "cast_vote",
                "goal": "Butterfly",
                "generation": 0,
                "index": 0,
                "vote": 1,
            }))
            started = _recv_until(ws, "votes_updated")
            assert started["votedCandidates"] == 5
            assert started["likedCandidates"] == 1
            assert started["countdownMs"] > 0

        # Fast countdown run to verify generation evolution + stale-generation vote guard
        mummy_ws.set_round_config(5, 80)
        with connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=bob") as bob:
            _recv_until(bob, "evolution_state")
            for idx in range(5):
                bob.send(json.dumps({
                    "type": "cast_vote",
                    "goal": "Butterfly",
                    "generation": 0,
                    "index": idx,
                    "vote": 1 if idx == 0 else -1,
                }))
            ev1 = _recv_until(bob, "goal_updated", timeout=3.0)
            assert ev1["goal"]["generation"] == 1

            # Stale generation=0 vote must be dropped cleanly
            bob.send(json.dumps({
                "type": "cast_vote",
                "goal": "Butterfly",
                "generation": 0,
                "index": 1,
                "vote": 1,
            }))
            with pytest.raises(TimeoutError):
                _recv_until(bob, "votes_updated", timeout=0.2)

            # Matching generation=1 vote succeeds
            bob.send(json.dumps({
                "type": "cast_vote",
                "goal": "Butterfly",
                "generation": 1,
                "index": 1,
                "vote": 1,
            }))
            v_ok = _recv_until(bob, "votes_updated")
            assert v_ok["totalVotes"] == 1 and v_ok["likedCandidates"] == 1


@pytest.mark.django_db(transaction=True)
def test_deploy_restart_restores_and_rebroadcasts_state():
    mummy_ws.clear_state()
    port1 = _free_port()
    ws_bridge.start_ws_server(host="127.0.0.1", port=port1, seed=True, pop_size=8, max_votes=4)
    try:
        mummy_ws.set_round_config(1, 60)
        with (
            connect(f"ws://127.0.0.1:{port1}/ws/chat/lobby/?username=alice") as lob1,
            connect(f"ws://127.0.0.1:{port1}/ws/chat/p-deploy1/?username=bob&goal=Orchid&pop=6&sigma=0.45") as priv1,
        ):
            _recv_until(lob1, "evolution_state")
            _recv_until(priv1, "evolution_state")
            lob1.send(json.dumps({"type": "cast_vote", "goal": "Butterfly", "generation": 0, "index": 0, "vote": 1}))
            lob_ev1 = _recv_until(lob1, "goal_updated", timeout=3.0)["goal"]
            assert lob_ev1["generation"] == 1

            priv1.send(json.dumps({"type": "cast_vote", "goal": "Orchid", "generation": 0, "index": 1, "vote": 1}))
            priv_ev1 = _recv_until(priv1, "goal_updated", timeout=3.0)["goal"]
            assert priv_ev1["generation"] == 1
            saved_bfly_id = lob_ev1["genomes"][0]["id"]
            saved_bfly_weights = lob_ev1["genomes"][0]["weights"]
            saved_orchid_id = priv_ev1["genomes"][0]["id"]
            saved_orchid_weights = priv_ev1["genomes"][0]["weights"]
    finally:
        ws_bridge.stop_ws_server()
        mummy_ws.clear_state()

    # Boot fresh server instance (simulating post-deploy restart)
    port2 = _free_port()
    ws_bridge.start_ws_server(host="127.0.0.1", port=port2, seed=True, pop_size=8, max_votes=4)
    try:
        with (
            connect(f"ws://127.0.0.1:{port2}/ws/chat/lobby/?username=alice") as lob2,
            connect(f"ws://127.0.0.1:{port2}/ws/chat/p-deploy1/?username=bob&goal=Orchid&pop=6&sigma=0.45") as priv2,
        ):
            lob_evo = _recv_until(lob2, "evolution_state")
            bfly2 = next(g for g in lob_evo["goals"] if g["name"] == "Butterfly")
            assert bfly2["generation"] == 1
            assert bfly2["lineageTotal"] == 16
            assert bfly2["genomes"][0]["id"] == saved_bfly_id
            assert bfly2["genomes"][0]["weights"] == saved_bfly_weights

            priv_evo = _recv_until(priv2, "evolution_state")
            orchid2 = next(g for g in priv_evo["goals"] if g["name"] == "Orchid")
            assert orchid2["generation"] == 1
            assert orchid2["lineageTotal"] == 12
            assert abs(orchid2["mutationSigma"] - 0.45) < 1e-6
            assert orchid2["genomes"][0]["id"] == saved_orchid_id
            assert orchid2["genomes"][0]["weights"] == saved_orchid_weights

            # Evolve gen 1 -> gen 2 after restart and verify genome IDs never collide
            mummy_ws.set_round_config(1, 60)
            lob2.send(json.dumps({"type": "cast_vote", "goal": "Butterfly", "generation": 1, "index": 0, "vote": 1}))
            lob_ev2 = _recv_until(lob2, "goal_updated", timeout=3.0)["goal"]
            assert lob_ev2["generation"] == 2
            assert lob_ev2["genomes"][0]["id"] > max(saved_bfly_id, saved_orchid_id)
    finally:
        ws_bridge.stop_ws_server()
        mummy_ws.clear_state()


@pytest.mark.django_db(transaction=True)
def test_admin_clear_lobby_goal_and_private_room(client, monkeypatch, settings):
    from direct_evolution_together.models import RoomEvolutionState

    mummy_ws.clear_state()
    port = _free_port()
    ws_bridge.start_ws_server(
        host="127.0.0.1",
        port=port,
        seed=True,
        pop_size=6,
        max_votes=4,
        admin_password="pw-clear",
    )
    try:
        settings.WS_HOST = "127.0.0.1"
        settings.WS_PORT = port
        monkeypatch.setenv("DET_ADMIN_PASSWORD", "pw-clear")
        mummy_ws.set_round_config(1, 50)

        with (
            connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=alice") as lob,
            connect(f"ws://127.0.0.1:{port}/ws/chat/p-clr1/?username=bob&goal=Lotus&pop=5") as priv,
        ):
            _recv_until(lob, "evolution_state")
            _recv_until(priv, "evolution_state")

            # Evolve Butterfly and Galaxy on lobby to gen 1, plus Lotus in private room to gen 1
            for gname in ("Butterfly", "Galaxy"):
                lob.send(json.dumps({"type": "cast_vote", "goal": gname, "generation": 0, "index": 0, "vote": 1}))
                assert _recv_until(lob, "goal_updated", timeout=3.0)["goal"]["generation"] == 1
            priv.send(json.dumps({"type": "cast_vote", "goal": "Lotus", "generation": 0, "index": 0, "vote": 1}))
            assert _recv_until(priv, "goal_updated", timeout=3.0)["goal"]["generation"] == 1

            # Clear ONLY Butterfly on lobby; Galaxy remains at gen 1
            resp = client.post(
                "/api/clear/",
                data=json.dumps({"password": "pw-clear", "room": "lobby", "goal": "Butterfly"}),
                content_type="application/json",
            )
            assert resp.status_code == 200
            upd = _recv_until(lob, "goal_updated", timeout=3.0)["goal"]
            assert upd["name"] == "Butterfly" and upd["generation"] == 0

            # Clear entire lobby via /api/clear/ (room="lobby", goal="")
            resp_all = client.post(
                "/api/clear/",
                data=json.dumps({"password": "pw-clear", "room": "lobby"}),
                content_type="application/json",
            )
            assert resp_all.status_code == 200
            lob_cleared = _recv_until(lob, "evolution_state", timeout=3.0)
            assert all(g["generation"] == 0 for g in lob_cleared["goals"])

            # Clear private room state via separate-process loopback HTTP path
            monkeypatch.setattr(ws_bridge, "is_running", lambda: False)
            resp_priv = client.post(
                "/api/clear/",
                data=json.dumps({"password": "pw-clear", "room": "p-clr1"}),
                content_type="application/json",
            )
            assert resp_priv.status_code == 200
            priv_cleared = _recv_until(priv, "evolution_state", timeout=3.0)
            assert priv_cleared["goals"] == []
            assert not RoomEvolutionState.objects.filter(room="p-clr1").exists()

            # Reset page renders quick-select room target
            reset_html = client.get("/reset/").content.decode()
            assert "Front Lobby" in reset_html and "Clear Room State" in reset_html
    finally:
        ws_bridge.stop_ws_server()
        mummy_ws.clear_state()
