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
        seed_from_db(default_pop_size=12, default_max_votes=2)
        with connect(f"ws://127.0.0.1:{port}/ws/chat/histroom/?username=viewer") as v:
            hist = _recv_until(v, "history")
            assert len(hist["messages"]) == 100
            assert hist["messages"][0]["message"] == "m-5"

        mummy_ws.set_round_config(2, 150)
        with connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=alice") as a1:
            assert _recv_until(a1, "identity")["username"] == "alice"
            evo0 = _recv_until(a1, "evolution_state")
            assert [g["name"] for g in evo0["goals"]] == [
                "Microduck",
                "Unitree Go2",
                "SO-102 Arm",
                "HalfCheetah",
                "CartPole",
                "Butterfly",
                "Galaxy",
                "Creature",
            ]
            assert evo0["goals"][0]["task"] == "microduck"
            assert evo0["goals"][0]["phase"] == "simulating"
            assert evo0["goals"][0]["simPhaseEnabled"] is True
            assert evo0["goals"][0]["eliteFrac"] == 0.25
            assert len(evo0["goals"][0]["genomes"]) == 8
            assert evo0["goals"][1]["task"] == "unitree_go2"
            assert evo0["goals"][1]["phase"] == "simulating"
            assert evo0["goals"][1]["simPhaseEnabled"] is True
            assert evo0["goals"][1]["eliteFrac"] == 0.25
            assert len(evo0["goals"][1]["genomes"]) == 8
            assert evo0["goals"][2]["task"] == "so101"
            assert evo0["goals"][2]["phase"] == "simulating"
            assert evo0["goals"][2]["simPhaseEnabled"] is True
            assert evo0["goals"][2]["eliteFrac"] == 0.25
            assert len(evo0["goals"][2]["genomes"]) == 8
            assert evo0["goals"][3]["task"] == "halfcheetah"
            assert evo0["goals"][3]["minVotes"] == 2
            assert evo0["goals"][3]["eliteFrac"] == 0.05
            assert len(evo0["goals"][3]["genomes"]) == 12
            assert evo0["goals"][4]["task"] == "cartpole"
            assert evo0["goals"][4]["minVotes"] == 2
            assert evo0["goals"][4]["eliteFrac"] == 0.0
            assert len(evo0["goals"][4]["genomes"]) == 12
            assert evo0["goals"][5]["task"] == "image"
            assert evo0["goals"][5]["minVotes"] == 2
            assert len(evo0["goals"][5]["genomes"]) == 12

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

            md_goal = msg["goals"][0]
            assert md_goal["task"] == "microduck"
            assert md_goal["lineageTotal"] == len(md_goal["lineage"]) == 6
            assert all(node.get("kind") == "populora" and "adapters" not in node for node in md_goal["lineage"])

            goal = next(g for g in msg["goals"] if g["task"] == "image")
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

    # Public CPPN image room always lets one researcher vote up to required quorum (>= minVotes)
    with _ws_server(seed=False, pop_size=24, min_votes=20, max_votes=5) as port:
        with connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=solo") as ws_solo:
            evo = _recv_until(ws_solo, "evolution_state")
            galaxy = next(g for g in evo["goals"] if g["name"] == "Galaxy")
            assert galaxy["minVotes"] == 20
            assert galaxy["maxVotesPerUser"] == 20
            mummy_ws.set_round_config(20, 60)
            for idx in range(20):
                ws_solo.send(json.dumps({
                    "type": "cast_vote",
                    "goal": "Galaxy",
                    "generation": 0,
                    "index": idx,
                    "vote": 1 if idx == 0 else -1,
                }))
                _recv_until(ws_solo, "votes_updated")
            gen1 = _recv_until(ws_solo, "goal_updated", timeout=3.0)
            assert gen1["goal"]["name"] == "Galaxy" and gen1["goal"]["generation"] == 1


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


@pytest.mark.django_db
def test_weight_upload_validation_and_population_cap(client):
    from django.core.files.uploadedfile import SimpleUploadedFile

    from direct_evolution_together.cppn import save_cppn_bytes
    from direct_evolution_together.models import Room, RoomEvolutionState

    valid_net = CPPN.random(101)
    valid_pt = save_cppn_bytes(valid_net.to_dict())

    # 1. Corrupt .pt, unsupported extension, and zip without .pt must fail compatibility check
    for bad_file in (
        SimpleUploadedFile("corrupt.pt", b"not-a-pytorch-checkpoint"),
        SimpleUploadedFile("notes.txt", valid_pt),
        SimpleUploadedFile("bad.zip", b"not-a-zip"),
    ):
        r = client.post("/api/weights/validate/", {"weights": [bad_file]})
        assert r.status_code == 400 and r.json()["ok"] is False

    empty_zip_buf = io.BytesIO()
    with zipfile.ZipFile(empty_zip_buf, "w") as zf:
        zf.writestr("readme.txt", "hello")
    r_no_pt = client.post(
        "/api/weights/validate/",
        {"weights": [SimpleUploadedFile("empty.zip", empty_zip_buf.getvalue())]},
    )
    assert r_no_pt.status_code == 400 and "no .pt" in r_no_pt.json()["error"]

    # 2. Wrong architecture / wrong shape / NaN weights / invalid activation codes must fail
    wrong_arch_buf = io.BytesIO()
    torch.save({"model": torch.nn.Linear(4, 3).state_dict()}, wrong_arch_buf)
    r_arch = client.post(
        "/api/weights/validate/",
        {"weights": [SimpleUploadedFile("linear.pt", wrong_arch_buf.getvalue())]},
    )
    assert r_arch.status_code == 400 and "Incompatible network architecture" in r_arch.json()["error"]

    wrong_shape_sd = {k: v.clone() for k, v in valid_net.state_dict().items()}
    wrong_shape_sd["layers.0.weight"] = torch.zeros(8, 4)
    wrong_shape_buf = io.BytesIO()
    torch.save({"model": wrong_shape_sd}, wrong_shape_buf)
    r_shape = client.post(
        "/api/weights/validate/",
        {"weights": [SimpleUploadedFile("wrong_shape.pt", wrong_shape_buf.getvalue())]},
    )
    assert r_shape.status_code == 400 and "Incompatible shape" in r_shape.json()["error"]

    nan_sd = {k: v.clone() for k, v in valid_net.state_dict().items()}
    nan_sd["layers.1.bias"][0] = float("nan")
    nan_buf = io.BytesIO()
    torch.save({"model": nan_sd}, nan_buf)
    r_nan = client.post(
        "/api/weights/validate/",
        {"weights": [SimpleUploadedFile("nan.pt", nan_buf.getvalue())]},
    )
    assert r_nan.status_code == 400 and "NaN or Inf" in r_nan.json()["error"]

    bad_act_sd = {k: v.clone() for k, v in valid_net.state_dict().items()}
    bad_act_sd["activations"][0] = 99
    bad_act_buf = io.BytesIO()
    torch.save({"model": bad_act_sd}, bad_act_buf)
    r_act = client.post(
        "/api/weights/validate/",
        {"weights": [SimpleUploadedFile("bad_act.pt", bad_act_buf.getvalue())]},
    )
    assert r_act.status_code == 400 and "Activation codes out of range" in r_act.json()["error"]

    # 3. Zip containing macOS resource forks (__MACOSX / ._weights.pt) ignores junk and loads real .pt
    mac_zip_buf = io.BytesIO()
    with zipfile.ZipFile(mac_zip_buf, "w") as zf:
        zf.writestr("__MACOSX/strain_a/._weights.pt", b"appledouble-junk")
        zf.writestr("strain_a/._weights.pt", b"appledouble-junk")
        zf.writestr("strain_a/weights.pt", valid_pt)
    r_mac = client.post(
        "/api/weights/validate/",
        {"weights": [SimpleUploadedFile("strain_a.zip", mac_zip_buf.getvalue())]},
    )
    assert r_mac.status_code == 200
    mac_json = r_mac.json()
    assert mac_json["ok"] is True and len(mac_json["strains"]) == 1
    assert mac_json["strains"][0]["label"] == "strain_a"
    assert mac_json["strains"][0]["genome"]["weights"] == valid_net.to_dict()["weights"]

    # 4. Population size boundary checks: uploaded count allowed up to population_size, rejected above
    r_full = client.post(
        "/api/rooms/",
        {
            "name": "Full Seeded Dish",
            "population_size": "3",
            "weights": [SimpleUploadedFile(f"s{i}.pt", valid_pt) for i in range(3)],
        },
    )
    assert r_full.status_code == 200
    full_json = r_full.json()
    assert full_json["seeded"] == 3
    assert full_json["population_size"] == 3
    saved_full = json.loads(RoomEvolutionState.objects.get(room="full-seeded-dish").state_json)
    assert len(saved_full["goals"][0]["genomes"]) == 3
    assert all(g["op"] == "seed" for g in saved_full["goals"][0]["genomes"])

    r_cap = client.post(
        "/api/rooms/",
        {
            "name": "Overfull Dish",
            "population_size": "3",
            "weights": [SimpleUploadedFile(f"s{i}.pt", valid_pt) for i in range(4)],
        },
    )
    assert r_cap.status_code == 400
    assert "cannot exceed total population size" in r_cap.json()["error"]
    assert not Room.objects.filter(name="Overfull Dish").exists()
    assert not RoomEvolutionState.objects.filter(room="overfull-dish").exists()

    # Out-of-range population sizes (2 and 65) are rejected explicitly
    for bad_pop in ("2", "65", "abc"):
        r_pop = client.post(
            "/api/rooms/",
            {
                "name": "Bad Pop Dish",
                "population_size": bad_pop,
                "weights": [SimpleUploadedFile("s0.pt", valid_pt)],
            },
        )
        assert r_pop.status_code == 400
        assert not Room.objects.filter(name="Bad Pop Dish").exists()

    # 5. Server-side large-file, zip-bomb, internal .pt bomb, and entry-flood protections
    huge_pt = b"x" * (160 * 1024)  # > 128 KB per-checkpoint limit
    r_huge_val = client.post(
        "/api/weights/validate/",
        {"weights": [SimpleUploadedFile("huge.pt", huge_pt)]},
    )
    assert r_huge_val.status_code == 400
    assert "exceeds the 128 KB size limit" in r_huge_val.json()["error"]

    r_huge_room = client.post(
        "/api/rooms/",
        {"name": "Huge Dish", "weights": [SimpleUploadedFile("huge.pt", huge_pt)]},
    )
    assert r_huge_room.status_code == 400
    assert "exceeds the 128 KB size limit" in r_huge_room.json()["error"]
    assert not Room.objects.filter(name="Huge Dish").exists()

    # Oversized HTTP request body (> 4 MB Content-Length) rejected with 413 pre-multipart
    r_oversized_req = client.post(
        "/api/weights/validate/",
        data=b"0" * 32,
        content_type="multipart/form-data; boundary=xyz",
        CONTENT_LENGTH=str(5 * 1024 * 1024),
    )
    assert r_oversized_req.status_code == 413
    assert "exceeds the 4 MB maximum request size" in r_oversized_req.json()["error"]

    # Outer .zip decompression bomb: tiny compressed zip (< 2 KB) expanding to 512 KB .pt
    zip_bomb_buf = io.BytesIO()
    with zipfile.ZipFile(zip_bomb_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bomb/weights.pt", b"\x00" * (512 * 1024))
    assert len(zip_bomb_buf.getvalue()) < 4096
    r_zip_bomb = client.post(
        "/api/weights/validate/",
        {"weights": [SimpleUploadedFile("bomb.zip", zip_bomb_buf.getvalue())]},
    )
    assert r_zip_bomb.status_code == 400
    assert "exceeds the 128 KB size limit" in r_zip_bomb.json()["error"]

    # Inner .pt zip decompression bomb: tiny .pt zip (< 2 KB) expanding to 512 KB internally
    pt_bomb_buf = io.BytesIO()
    with zipfile.ZipFile(pt_bomb_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("archive/data/0", b"\x00" * (512 * 1024))
    assert len(pt_bomb_buf.getvalue()) < 4096
    r_pt_bomb = client.post(
        "/api/weights/validate/",
        {"weights": [SimpleUploadedFile("inner_bomb.pt", pt_bomb_buf.getvalue())]},
    )
    assert r_pt_bomb.status_code == 400
    assert "decompressed size exceeds the 256 KB safety limit" in r_pt_bomb.json()["error"]

    # Zip archive entry-count flood (> 256 entries) and nested .zip rejected
    flood_buf = io.BytesIO()
    with zipfile.ZipFile(flood_buf, "w") as zf:
        for idx in range(270):
            zf.writestr(f"junk/f{idx}.txt", b"a")
    r_flood = client.post(
        "/api/weights/validate/",
        {"weights": [SimpleUploadedFile("flood.zip", flood_buf.getvalue())]},
    )
    assert r_flood.status_code == 400
    assert "too many entries" in r_flood.json()["error"]


@pytest.mark.django_db(transaction=True)
def test_e2e_download_weights_reseed_new_room_and_evolve(client, monkeypatch):
    from django.core.files.uploadedfile import SimpleUploadedFile

    with _ws_server(seed=True, pop_size=8, max_votes=4) as port:
        mummy_ws.set_round_config(2, 60)

        # Step 1: Run a full 5-generation evolution in /lobby ("Butterfly"), archiving 2 strains along the way
        saved_gen2_genome = None
        saved_gen5_genome = None
        with (
            connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=archivist") as a_lob,
            connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=breeder") as b_lob,
        ):
            lob_evo = _recv_until(a_lob, "evolution_state")
            _recv_until(b_lob, "evolution_state")
            cur_goal = {g["name"]: g for g in lob_evo["goals"]}["Butterfly"]
            assert cur_goal["generation"] == 0

            for expected_next_gen in range(1, 6):
                gen = cur_goal["generation"]
                a_lob.send(json.dumps({
                    "type": "cast_vote", "goal": "Butterfly", "generation": gen, "index": 0, "vote": 1,
                }))
                _recv_until(a_lob, "votes_updated")
                b_lob.send(json.dumps({
                    "type": "cast_vote", "goal": "Butterfly", "generation": gen, "index": 1, "vote": 1,
                }))
                cur_goal = _recv_until(a_lob, "goal_updated", timeout=4.0)["goal"]
                _recv_until(b_lob, "goal_updated", timeout=4.0)
                assert cur_goal["generation"] == expected_next_gen

                if expected_next_gen == 2:
                    # Save individual #1 midway through the run at Generation 2
                    saved_gen2_genome = cur_goal["genomes"][0]
                elif expected_next_gen == 5:
                    # Save individual #2 at the end of the 5-generation run (Generation 5)
                    saved_gen5_genome = cur_goal["genomes"][2]

            assert saved_gen2_genome is not None
            assert saved_gen5_genome is not None

            # Step 2: Export one as a .zip bundle (/export/bundle/) and one as raw .pt (/export/weights/)
            bundle_resp = client.post(
                "/export/bundle/",
                data=json.dumps({
                    "goal": "Butterfly",
                    "generation": 2,
                    "index": 0,
                    "genome": saved_gen2_genome,
                }),
                content_type="application/json",
            )
            assert bundle_resp.status_code == 200

            weights_resp = client.post(
                "/export/weights/",
                data=json.dumps({
                    "goal": "Butterfly",
                    "generation": 5,
                    "index": 2,
                    "genome": saved_gen5_genome,
                }),
                content_type="application/json",
            )
            assert weights_resp.status_code == 200

            # Step 3: Validate both archived downloads via /api/weights/validate/ (live UI preview path)
            val_resp = client.post(
                "/api/weights/validate/",
                {
                    "weights": [
                        SimpleUploadedFile("butterfly_g2_0.zip", bundle_resp.content),
                        SimpleUploadedFile("butterfly_g5_2.pt", weights_resp.content),
                    ]
                },
            )
            assert val_resp.status_code == 200
            preview_strains = val_resp.json()["strains"]
            assert [s["label"] for s in preview_strains] == ["butterfly_g2_0", "butterfly_g5_2"]
            assert preview_strains[0]["genome"]["weights"] == saved_gen2_genome["weights"]
            assert preview_strains[1]["genome"]["weights"] == saved_gen5_genome["weights"]

            # Step 4: Create a new room reseeded with the 2 archived strains (< pop_size=6)
            create_resp = client.post(
                "/api/rooms/",
                {
                    "name": "Petri Reseed",
                    "description": "Reseeded from Gen 2 & Gen 5 Butterfly strains",
                    "population_size": "6",
                    "max_votes_per_user": "4",
                    "mutation_sigma": "0.35",
                    "elite_frac": "0.35",
                    "crossover_rate": "0.80",
                    "weights": [
                        SimpleUploadedFile("butterfly_g2_0.zip", bundle_resp.content),
                        SimpleUploadedFile("butterfly_g5_2.pt", weights_resp.content),
                    ],
                },
            )
            assert create_resp.status_code == 200
            room_data = create_resp.json()
            assert room_data["ok"] is True
            assert room_data["slug"] == "petri-reseed"
            assert room_data["seeded"] == 2
            assert room_data["population_size"] == 6
            assert room_data["url"] == "/p/petri-reseed/"

            # Share the newly created room URL in chat and verify broadcast preserves full URL & room path
            invite_msg = "Reseeded Gen 2 + Gen 5 strains at http://127.0.0.1:8000/p/petri-reseed/ (/p/petri-reseed/)!"
            a_lob.send(json.dumps({"type": "chat", "message": invite_msg}))
            chat_evt = _recv_until(b_lob, "chat")
            assert chat_evt["username"] == "archivist"
            assert chat_evt["message"] == invite_msg

        page_html = client.get(room_data["url"]).content.decode()
        assert "Petri Reseed" in page_html
        assert "seed-dropzone" in page_html
        assert "Seed (S)" in page_html
        assert "appendLinkifiedText" in page_html
        assert "chat-link" in page_html

        # Step 5: Join the newly created room over WebSocket; verify Gen 0 has 2 seeded + 4 random = 6 total
        with (
            connect(f"ws://127.0.0.1:{port}/ws/chat/petri-reseed/?username=alice") as a_ws,
            connect(f"ws://127.0.0.1:{port}/ws/chat/petri-reseed/?username=bob") as b_ws,
        ):
            reseed_evo = _recv_until(a_ws, "evolution_state")
            _recv_until(b_ws, "evolution_state")
            assert len(reseed_evo["goals"]) == 1
            dish = reseed_evo["goals"][0]
            assert dish["name"] == "Petri Reseed"
            assert dish["generation"] == 0
            assert dish["populationSize"] == 6
            assert len(dish["genomes"]) == 6

            g0, g1 = dish["genomes"][0], dish["genomes"][1]
            assert g0["op"] == "seed" and g0["parents"] == []
            assert g1["op"] == "seed" and g1["parents"] == []
            assert g0["weights"] == saved_gen2_genome["weights"]
            assert g0["biases"] == saved_gen2_genome["biases"]
            assert g0["activations"] == saved_gen2_genome["activations"]
            assert g1["weights"] == saved_gen5_genome["weights"]
            assert g1["biases"] == saved_gen5_genome["biases"]
            assert g1["activations"] == saved_gen5_genome["activations"]
            assert all(g["op"] == "random" for g in dish["genomes"][2:])

            # Step 6: Collaboratively vote on the 2 seeded strains to evolve Gen 0 -> Gen 1
            a_ws.send(json.dumps({
                "type": "cast_vote", "goal": "Petri Reseed", "generation": 0, "index": 0, "vote": 1,
            }))
            _recv_until(a_ws, "votes_updated")
            b_ws.send(json.dumps({
                "type": "cast_vote", "goal": "Petri Reseed", "generation": 0, "index": 1, "vote": 1,
            }))
            gen1 = _recv_until(a_ws, "goal_updated", timeout=4.0)["goal"]
            assert gen1["generation"] == 1
            assert len(gen1["genomes"]) == 6
            seed_ids = {g0["id"], g1["id"]}
            elites = [g for g in gen1["genomes"] if g["op"] == "elite"]
            assert len(elites) >= 2
            assert {tuple(elites[0]["weights"]), tuple(elites[1]["weights"])} == {
                tuple(saved_gen2_genome["weights"]),
                tuple(saved_gen5_genome["weights"]),
            }
            assert all(set(e["parents"]) <= seed_ids for e in elites[:2])

            a_ws.send(json.dumps({"type": "lineage_all", "goal": "Petri Reseed"}))
            full_lin = _recv_until(a_ws, "lineage_all")
            gen0_ops = [n["op"] for n in full_lin["lineage"] if n["gen"] == 0]
            assert gen0_ops.count("seed") == 2 and gen0_ops.count("random") == 4

        # Step 7: Also verify separate-process (uvicorn + ws_server) DB handoff on init_room
        monkeypatch.setattr(ws_bridge, "is_running", lambda: False)
        async_resp = client.post(
            "/api/rooms/",
            {
                "name": "Async Dish",
                "population_size": "4",
                "weights": [SimpleUploadedFile("butterfly_g2_0.zip", bundle_resp.content)],
            },
        )
        assert async_resp.status_code == 200
        with connect(f"ws://127.0.0.1:{port}/ws/chat/async-dish/?username=carol") as c_ws:
            async_evo = _recv_until(c_ws, "evolution_state", timeout=4.0)
            async_goal = async_evo["goals"][0]
            assert async_goal["name"] == "Async Dish"
            assert len(async_goal["genomes"]) == 4
            assert async_goal["genomes"][0]["op"] == "seed"
            assert async_goal["genomes"][0]["weights"] == saved_gen2_genome["weights"]
            assert all(g["op"] == "random" for g in async_goal["genomes"][1:])


def test_cartpole_substrate_js_python_parity_and_convergence():
    from pathlib import Path
    import random as py_random
    from direct_evolution_together.environments import (
        cartpole_rollout,
        evaluate_control_genome,
        get_task_spec,
        mulberry32,
    )

    # 1. Exact Python <-> JS physics, PRNG, and HyperNEAT substrate rollout parity
    html_text = (
        Path(__file__).resolve().parent.parent
        / "direct_evolution_together"
        / "templates"
        / "direct_evolution_together"
        / "index.html"
    ).read_text(encoding="utf-8")
    p_start = html_text.index("function applyActivation(")
    p_end = html_text.index("const controlEvalCache =")
    js_core = "\n".join(
        [
            "const genomeFingerprints = new WeakMap();",
            'function genomeFingerprint(g) { return String((g && g.id) || "g0"); }',
            html_text[p_start:p_end],
            """
            const input = JSON.parse(require("fs").readFileSync(0, "utf8"));
            const prng = mulberry32(input.seed);
            const prngSamples = [prng(), prng(), prng(), prng()];
            const ro = cartpoleRollout(input.genome, input.seed, input.maxSteps);
            process.stdout.write(JSON.stringify({ prngSamples, rollout: ro }));
            """,
        ]
    )

    pop = create_random_population(pop_size=4)
    for seed in (0, 7, 42):
        py_rng = mulberry32(seed)
        py_samples = [py_rng() for _ in range(4)]
        py_ro = cartpole_rollout(pop[0], seed=seed, max_steps=500)
        proc = subprocess.run(
            ["node", "-e", js_core],
            input=json.dumps({"genome": pop[0], "seed": seed, "maxSteps": 500}),
            text=True,
            capture_output=True,
            check=True,
        )
        js_res = json.loads(proc.stdout)
        assert py_samples == pytest.approx(js_res["prngSamples"], abs=1e-12)
        assert py_ro["steps"] == js_res["rollout"]["steps"]
        for s_py, s_js in zip(py_ro["states"][:10], js_res["rollout"]["states"][:10]):
            assert list(s_py) == pytest.approx(s_js, abs=1e-5)

    # 2. Fast convergence with Stanley's CPPN + HyperNEAT (4->8->1) and zero elitism on pop=16 within 20 gens
    spec = get_task_spec("cartpole")
    assert spec.substrate_dims == (4, 8, 1)
    assert spec.default_pop_size == 30
    assert spec.max_pop_size == 32
    assert spec.default_min_votes == 20
    assert spec.default_elite_frac == 0.0

    saved_state = py_random.getstate()
    py_random.seed(2026)
    torch_state = torch.random.get_rng_state()
    torch.manual_seed(2026)
    try:
        evo_pop = create_random_population(pop_size=16)
        solved_gen = None
        best_heldout = None
        saw_zero_elitism = False
        for gen in range(20):
            ranked = sorted(
                (
                    (
                        evaluate_control_genome("cartpole", g, seeds=spec.eval_seeds),
                        i,
                    )
                    for i, g in enumerate(evo_pop)
                ),
                key=lambda item: (item[0]["meanSteps"], -item[0]["meanAbsX"]),
                reverse=True,
            )
            top_candidate = evo_pop[ranked[0][1]]
            heldout = evaluate_control_genome("cartpole", top_candidate, seeds=tuple(range(100, 110)))
            best_heldout = heldout
            if heldout["minSteps"] >= 490 and gen > 0:
                solved_gen = gen
                break
            votes = [0] * len(evo_pop)
            up_counts = [0] * len(evo_pop)
            for rank_pos, (_, idx) in enumerate(ranked[:3]):
                votes[idx] = 3 - rank_pos
                up_counts[idx] = 1
            evo_pop, _ = evolve_population(
                evo_pop,
                votes,
                up_counts,
                mut_sigma=0.35,
                pop_size=16,
                elite_frac=0.0,
                crossover_rate=0.75,
            )
            assert all(g["op"] != "elite" for g in evo_pop)
            saw_zero_elitism = True
        assert saw_zero_elitism
        assert solved_gen is not None and solved_gen < 20, f"Did not solve in 20 gens: {best_heldout}"
        assert best_heldout["meanSteps"] >= 495.0
    finally:
        py_random.setstate(saved_state)
        torch.random.set_rng_state(torch_state)


@pytest.mark.django_db(transaction=True)
def test_control_room_api_ws_quorum_zero_elitism_and_resume(client):
    from direct_evolution_together.models import RoomEvolutionState

    # Legacy lobby DB row without CartPole automatically receives CartPole on seed_from_db
    legacy_state = json.dumps(
        [
            {"name": "Butterfly", "genomes": create_random_population(pop_size=8)},
            {"name": "Galaxy", "genomes": create_random_population(pop_size=8)},
            {"name": "Creature", "genomes": create_random_population(pop_size=8)},
        ]
    )
    RoomEvolutionState.objects.update_or_create(room="lobby", defaults={"state_json": legacy_state})

    # Control room pop cap enforcement (>32 rejected)
    bad_resp = client.post(
        "/api/rooms/",
        data=json.dumps({"name": "Too Big CartPole", "task": "cartpole", "population_size": 33}),
        content_type="application/json",
    )
    assert bad_resp.status_code == 400

    # Valid CartPole private room creation seeds Gen 0 immediately
    ok_resp = client.post(
        "/api/rooms/",
        data=json.dumps({"name": "CartPole Lab", "task": "cartpole", "population_size": 16}),
        content_type="application/json",
    )
    assert ok_resp.status_code == 200
    room_data = ok_resp.json()
    assert room_data["task"] == "cartpole"
    assert room_data["population_size"] == 16
    from direct_evolution_together.environments import evaluate_control_genome

    page_html = client.get(room_data["url"]).content.decode()
    assert "CartPole" in page_html
    assert "advance-round-btn" not in page_html
    assert "Evolve now" not in page_html
    assert '<footer class="site-footer">' in page_html
    assert "justify-content: flex-end;" in page_html.split(".site-footer {", 1)[1].split("}", 1)[0]
    footer_part = page_html.split('<footer class="site-footer">', 1)[1].split("</footer>", 1)[0]
    header_part = page_html.split("<header>", 1)[1].split("</header>", 1)[0]
    assert "Picbreeder" in footer_part
    assert "Picbreeder" not in header_part
    assert '<form id="name-form" class="name-form">' in page_html
    name_form_html = page_html.split('<form id="name-form"', 1)[1].split("</form>", 1)[0]
    assert 'class="composer-box"' in name_form_html and 'id="name-btn"' in name_form_html
    assert 'id="new-room-min-votes"' in page_html and 'id="new-room-max-votes"' in page_html
    assert 'id="empty-min-votes-input"' in page_html and 'id="empty-max-votes-input"' in page_html

    split_resp = client.post(
        "/api/rooms/",
        data=json.dumps({
            "name": "Quorum Split",
            "task": "cartpole",
            "population_size": 8,
            "min_votes": 12,
            "max_votes_per_user": 4,
        }),
        content_type="application/json",
    )
    assert split_resp.status_code == 200
    split_data = split_resp.json()

    with _ws_server(seed=True, pop_size=12, max_votes=6) as port:
        seed_from_db(default_pop_size=12, default_max_votes=6)
        # Check migrated lobby has CartPole prepended with zero elitism, high quorum, and no pre-solved Gen-0 founders
        with connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=mig") as ws_lob:
            lob_state = _recv_until(ws_lob, "evolution_state")
            assert [g["name"] for g in lob_state["goals"]] == [
                "Microduck",
                "Unitree Go2",
                "SO-102 Arm",
                "HalfCheetah",
                "CartPole",
                "Butterfly",
                "Galaxy",
                "Creature",
            ]
            assert lob_state["goals"][0]["task"] == "microduck"
            assert len(lob_state["goals"][0]["genomes"]) == 8
            assert lob_state["goals"][0]["minVotes"] == 4
            assert lob_state["goals"][1]["task"] == "unitree_go2"
            assert len(lob_state["goals"][1]["genomes"]) == 8
            assert lob_state["goals"][1]["minVotes"] == 4
            assert lob_state["goals"][2]["task"] == "so101"
            assert len(lob_state["goals"][2]["genomes"]) == 8
            assert lob_state["goals"][2]["minVotes"] == 4
            assert lob_state["goals"][3]["task"] == "halfcheetah"
            assert lob_state["goals"][3]["eliteFrac"] == 0.05
            assert lob_state["goals"][3]["minVotes"] == 6
            assert all(
                evaluate_control_genome("halfcheetah", cand, seeds=(0, 1))["meanDistance"] < 4.5
                for cand in lob_state["goals"][3]["genomes"]
            )
            assert lob_state["goals"][4]["task"] == "cartpole"
            assert lob_state["goals"][4]["eliteFrac"] == 0.0
            assert lob_state["goals"][4]["minVotes"] == 6
            assert all(
                evaluate_control_genome("cartpole", cand, seeds=(0, 1, 2))["meanSteps"] < 60.0
                for cand in lob_state["goals"][4]["genomes"]
            )

        # CartPole quorum is capped by the per-person vote budget; image quorum stays independent
        with connect(f"ws://127.0.0.1:{port}/ws/chat/{split_data['slug']}/?username=splituser") as ws_split:
            split_evo = _recv_until(ws_split, "evolution_state")
            assert split_evo["goals"][0]["minVotes"] == 4
            assert split_evo["goals"][0]["maxVotesPerUser"] == 4

        with connect(
            f"ws://127.0.0.1:{port}/ws/chat/p-split-img/?username=imguser&goal=Prism&pop=8&min_votes=6&max_votes=2"
        ) as ws_img:
            img_evo = _recv_until(ws_img, "evolution_state")
            assert img_evo["goals"][0]["minVotes"] == 6
            assert img_evo["goals"][0]["maxVotesPerUser"] == 2

        # Check pre-seeded CartPole room + create_goal + one-person quorum -> zero-elite Gen 1
        mummy_ws.set_round_config(10, 120)
        with connect(f"ws://127.0.0.1:{port}/ws/chat/{room_data['slug']}/?username=pilot") as ws:
            evo0 = _recv_until(ws, "evolution_state")
            goal = evo0["goals"][0]
            assert goal["task"] == "cartpole"
            assert goal["minVotes"] == 20
            assert goal["eliteFrac"] == 0.0
            assert len(goal["genomes"]) == 16
            assert all(
                evaluate_control_genome("cartpole", cand, seeds=(0, 1, 2))["meanSteps"] < 60.0
                for cand in goal["genomes"]
            )

            # Create a second CartPole goal in the same private room
            ws.send(json.dumps({
                "type": "create_goal",
                "goal": "Pole Run 2",
                "task": "cartpole",
                "population_size": 8,
                "max_votes_per_user": 6,
            }))
            evo_after_create = _recv_until(ws, "evolution_state")
            created_goals = {g["name"]: g for g in evo_after_create["goals"]}
            assert "Pole Run 2" in created_goals
            created = created_goals["Pole Run 2"]
            assert created["task"] == "cartpole"
            assert created["minVotes"] == 6
            assert created["eliteFrac"] == 0.0
            assert len(created["genomes"]) == 8

            # One person's full vote budget now satisfies the cartpole quorum
            for idx in range(6):
                ws.send(json.dumps({
                    "type": "cast_vote",
                    "goal": "Pole Run 2",
                    "generation": 0,
                    "index": idx,
                    "vote": 1,
                }))
                upd = _recv_until(ws, "votes_updated")
            assert upd["totalVotes"] == 6
            assert upd["countdownMs"] > 0
            gen1 = _recv_until(ws, "goal_updated", timeout=5.0)["goal"]

            assert gen1["name"] == "Pole Run 2"
            assert gen1["generation"] == 1
            assert gen1["task"] == "cartpole"
            assert gen1["eliteFrac"] == 0.0
            assert all(g["op"] != "elite" for g in gen1["genomes"])

        # Download Gen 1 CartPole .pt weights directly and resume in a new private CartPole room
        from django.core.files.uploadedfile import SimpleUploadedFile

        champ_genome = gen1["genomes"][0]
        pt_resp = client.post(
            "/export/weights/",
            data=json.dumps({
                "goal": "Pole Run 2",
                "task": "cartpole",
                "generation": 1,
                "index": 0,
                "genome": champ_genome,
            }),
            content_type="application/json",
        )
        assert pt_resp.status_code == 200
        assert 'filename="pole_run_2_g1_0.pt"' in pt_resp["Content-Disposition"]

        resumed_resp = client.post(
            "/api/rooms/",
            {
                "name": "CartPole Resumed",
                "task": "cartpole",
                "population_size": "6",
                "weights": [SimpleUploadedFile("pole_run_2_g1_0.pt", pt_resp.content)],
            },
        )
        assert resumed_resp.status_code == 200
        resumed_data = resumed_resp.json()
        assert resumed_data["task"] == "cartpole"
        assert resumed_data["seeded"] == 1

        with connect(f"ws://127.0.0.1:{port}/ws/chat/{resumed_data['slug']}/?username=resumer") as r_ws:
            res_evo = _recv_until(r_ws, "evolution_state")
            res_goal = res_evo["goals"][0]
            assert res_goal["task"] == "cartpole"
            assert res_goal["minVotes"] == 20
            assert res_goal["eliteFrac"] == 0.0
            assert len(res_goal["genomes"]) == 6
            assert res_goal["genomes"][0]["op"] == "seed"
            assert res_goal["genomes"][0]["weights"] == champ_genome["weights"]
            assert res_goal["genomes"][0]["biases"] == champ_genome["biases"]
            assert res_goal["genomes"][0]["activations"] == champ_genome["activations"]


@pytest.mark.django_db
def test_lazy_mp4_rendering_range_serving_gzip_exclusion_and_bundle(client, settings, tmp_path):
    settings.MEDIA_ROOT = tmp_path / "media"
    genome = create_random_population(pop_size=1)[0]

    # Image task correctly rejects video render endpoint
    bad_video = client.post(
        "/api/render/video/",
        data=json.dumps({"task": "image", "genome": genome}),
        content_type="application/json",
    )
    assert bad_video.status_code == 400

    # Lazy render for CartPole produces MP4 + poster PNG
    render_resp = client.post(
        "/api/render/video/",
        data=json.dumps({"task": "cartpole", "genome": genome, "seed": 0}),
        content_type="application/json",
    )
    assert render_resp.status_code == 200
    payload = render_resp.json()
    assert payload["ok"] is True
    assert payload["videoUrl"].endswith(".mp4")
    assert payload["posterUrl"].endswith(".png")
    assert payload["steps"] >= 1

    # Second identical request reuses cached content-addressed files
    render_resp2 = client.post(
        "/api/render/video/",
        data=json.dumps({"task": "cartpole", "genome": genome, "seed": 0}),
        content_type="application/json",
    )
    assert render_resp2.json()["videoUrl"] == payload["videoUrl"]

    # Full MP4 fetch must NOT be gzip-compressed and must advertise byte ranges & immutable cache
    mp4_resp = client.get(payload["videoUrl"], HTTP_ACCEPT_ENCODING="gzip, deflate")
    assert mp4_resp.status_code == 200
    assert mp4_resp["Content-Type"] == "video/mp4"
    assert mp4_resp["Accept-Ranges"] == "bytes"
    assert "immutable" in mp4_resp["Cache-Control"]
    assert not mp4_resp.has_header("Content-Encoding")
    raw_mp4 = b"".join(mp4_resp.streaming_content) if mp4_resp.streaming else mp4_resp.content
    assert b"ftyp" in raw_mp4[:64] and b"moov" in raw_mp4

    # Range request -> 206 Partial Content with exact Content-Range slice
    range_resp = client.get(
        payload["videoUrl"],
        HTTP_RANGE="bytes=0-127",
        HTTP_ACCEPT_ENCODING="gzip",
    )
    assert range_resp.status_code == 206
    assert range_resp["Content-Range"] == f"bytes 0-127/{len(raw_mp4)}"
    assert not range_resp.has_header("Content-Encoding")
    part_bytes = b"".join(range_resp.streaming_content) if range_resp.streaming else range_resp.content
    assert part_bytes == raw_mp4[:128]

    # Poster PNG fetch
    png_resp = client.get(payload["posterUrl"], HTTP_ACCEPT_ENCODING="gzip")
    assert png_resp.status_code == 200
    assert png_resp["Content-Type"] == "image/png"
    assert not png_resp.has_header("Content-Encoding")
    raw_png = b"".join(png_resp.streaming_content) if png_resp.streaming else png_resp.content
    assert raw_png.startswith(b"\x89PNG\r\n\x1a\n")

    # Control bundle export runs standalone via uv run --no-project run.py
    bundle_resp = client.post(
        "/export/bundle/",
        data=json.dumps({"goal": "CartPole", "task": "cartpole", "generation": 1, "index": 0, "genome": genome}),
        content_type="application/json",
    )
    assert bundle_resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(bundle_resp.content)) as zf:
        zf.extractall(tmp_path)
    run_dir = tmp_path / "cartpole_g1_0"
    proc = subprocess.run(
        ["uv", "run", "--no-project", "run.py"],
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Evaluated 10 held-out episodes" in proc.stdout
    assert (run_dir / "out.mp4").is_file()


def test_nginx_conf_handles_media_and_compression_for_subpath_and_root():
    import fabfile

    base_cfg = fabfile.DeployConfig(
        host="example.com",
        user="ubuntu",
        app_name="direct-evolution-together",
        deploy_dir="/srv/det",
        domain="example.com",
        subpath="/evolution",
        http_port=8000,
        ws_port=8001,
        uvicorn_workers=2,
        python_version="3.11",
        cuda_visible_devices="",
        nginx_vhost_file="/etc/nginx/sites-enabled/det",
        secret_key="secret",
        admin_password="pw",
        allowed_hosts="*",
        csrf_trusted_origins="https://example.com",
        pop_size=30,
        min_votes="",
        max_votes=10,
        mut_sigma="0.35",
        elite_frac="0.10",
        crossover_rate="0.75",
    )
    assert base_cfg.media_dir == "/srv/det/data/media"
    sub_conf = fabfile._render_nginx_conf(base_cfg)
    assert "location ^~ /evolution/media/ {" in sub_conf
    assert "alias /srv/det/data/media/;" in sub_conf
    assert "gzip off;" in sub_conf
    assert "sendfile on;" in sub_conf
    assert "expires 365d;" in sub_conf
    assert 'add_header Cache-Control "public, max-age=31536000, immutable" always;' in sub_conf
    assert "client_max_body_size 4m;" in sub_conf
    assert "proxy_buffering off;" in sub_conf
    assert "proxy_send_timeout 86400;" in sub_conf
    assert "gzip on;" in sub_conf
    assert "text/html" not in sub_conf

    root_cfg = fabfile.DeployConfig(**{**base_cfg.__dict__, "subpath": "", "nginx_vhost_file": ""})
    root_conf = fabfile._render_nginx_conf(root_cfg)
    assert "location ^~ /media/ {" in root_conf
    assert "alias /srv/det/data/media/;" in root_conf
    assert "gzip off;" in root_conf
    assert "server {" in root_conf

    # Verify remote-only PM2 ecosystem generation & dual WS + HTTP health checks (no repo PM2 file)
    assert base_cfg.process_manager == "pm2"
    assert base_cfg.ecosystem_file == "/srv/det/ecosystem.config.cjs"
    eco_src = fabfile._render_pm2_ecosystem(base_cfg)
    eco_parsed = json.loads(
        subprocess.check_output(
            ["node", "-e", f"const m = {{}}; (function(module){{\n{eco_src}\n}})(m); console.log(JSON.stringify(m.exports));"],
            text=True,
        )
    )
    apps_by_name = {a["name"]: a for a in eco_parsed["apps"]}
    assert set(apps_by_name) == {"direct-evolution-together-ws", "direct-evolution-together-uvicorn"}
    ws_app = apps_by_name["direct-evolution-together-ws"]
    assert ws_app["script"] == "/srv/det/src/.venv/bin/python"
    assert ws_app["args"] == "-m direct_evolution_together.ws_server"
    assert ws_app["interpreter"] == "none"
    assert ws_app["instances"] == 1
    assert ws_app["env"]["MUMMY_WS_PORT"] == "8001"

    pm2_restart = fabfile._bash_supervisor_restart_snippet(base_cfg, write_config=True)
    assert "systemctl disable --now direct-evolution-together-ws direct-evolution-together-uvicorn" in pm2_restart
    assert "pm2 startOrRestart /srv/det/ecosystem.config.cjs --update-env" in pm2_restart

    systemd_cfg = fabfile.DeployConfig(**{**base_cfg.__dict__, "process_manager": "systemd"})
    sys_restart = fabfile._bash_supervisor_restart_snippet(systemd_cfg, write_config=True)
    assert "systemctl restart direct-evolution-together-ws direct-evolution-together-uvicorn" in sys_restart

    health_snip = fabfile._bash_health_check_snippet(base_cfg)
    assert "http://127.0.0.1:8001/health" in health_snip
    assert "http://127.0.0.1:8000/health/" in health_snip
    assert not list(fabfile.REPO_ROOT.glob("ecosystem.config.*"))


@pytest.mark.django_db
def test_halfcheetah_fk_parity_convergence_server_eval_and_video(client, settings, tmp_path):
    import base64
    from pathlib import Path
    import random
    import re
    import mujoco
    from direct_evolution_together.cppn import genome_content_hash
    from direct_evolution_together.environments import (
        TASK_SPECS,
        _cheetah_fk_segments,
        attach_halfcheetah_geoms,
        create_task_population,
        evaluate_control_batch,
        evaluate_control_genome,
        get_halfcheetah_model,
        get_substrate,
        halfcheetah_reset,
        render_task_media,
    )

    settings.MEDIA_ROOT = tmp_path

    # 1. Verify HalfCheetah default pop cap & config
    hc_spec = TASK_SPECS["halfcheetah"]
    assert hc_spec.default_pop_size == 16
    assert hc_spec.max_pop_size == 24
    assert hc_spec.default_min_votes == 8
    too_big = client.post(
        "/api/rooms/",
        data=json.dumps({"name": "Huge Cheetah", "task": "halfcheetah", "population_size": 25}),
        content_type="application/json",
    )
    assert too_big.status_code == 400

    # 2. Verify 2D FK matches real MuJoCo geom_xpos and cheetah_renderer.js FK matches Python to 1e-9
    model = get_halfcheetah_model()
    data = mujoco.MjData(model)
    halfcheetah_reset(model, data, 7)
    for step in range(40):
        for a in range(6):
            data.ctrl[a] = 0.35 * ((step % 5) - 2) * 0.4
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    frame = [float(data.qpos[i]) for i in range(model.nq)] + [0.0, 0.0, 0.0]
    py_segs = dict((name, (p0, p1)) for name, p0, p1 in _cheetah_fk_segments(frame))

    def _geom_xz(gname: str):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        pos = data.geom_xpos[gid]
        return float(pos[0]), float(pos[2])

    for gname in ("bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot", "torso"):
        p0, p1 = py_segs[gname]
        mid_x = 0.5 * (p0[0] + p1[0])
        mid_z = 0.5 * (p0[1] + p1[1])
        gx, gz = _geom_xz(gname)
        assert abs(mid_x - gx) < 1e-4 and abs(mid_z - gz) < 1e-4

    renderer_js = (
        Path(__file__).resolve().parent.parent
        / "direct_evolution_together"
        / "frontend"
        / "cheetah_renderer.js"
    )
    fk_script = f"""
const fs = require('fs');
global.window = {{}};
eval(fs.readFileSync({json.dumps(str(renderer_js))}, 'utf8'));
const segs = window.DetCheetahRenderer.cheetahFkSegments({json.dumps(frame[:9])});
console.log(JSON.stringify(segs));
"""
    js_fk_out = json.loads(subprocess.check_output(["node", "-e", fk_script], text=True))
    js_segs = {item[0]: (tuple(item[1]), tuple(item[2])) for item in js_fk_out}
    assert set(js_segs) == set(py_segs)
    for k in py_segs:
        for pt_idx in range(2):
            for coord_idx in range(2):
                assert abs(py_segs[k][pt_idx][coord_idx] - js_segs[k][pt_idx][coord_idx]) < 1e-9

    # 3. Verify Python vs JS anatomical (21, 12, 6) HyperNEAT substrate weight decoding parity
    rng = random.Random(1234)
    torch.manual_seed(1234)
    torch_free_genome = create_task_population("halfcheetah", 1)[0]
    substrate = get_substrate(hc_spec.substrate_dims, hc_spec.layer_coords)
    py_weights, py_biases = substrate.decode(torch_free_genome)
    net_viz_js = (
        Path(__file__).resolve().parent.parent
        / "direct_evolution_together"
        / "frontend"
        / "network_viz.js"
    )
    decode_script = f"""
const nv = require({json.dumps(str(net_viz_js))});
const dec = nv.decodeHyperNeatWeights({json.dumps(torch_free_genome)}, {json.dumps(list(hc_spec.substrate_dims))}, {json.dumps([list(c) if c is not None else None for c in hc_spec.layer_coords])});
console.log(JSON.stringify({{
  w0: Array.from(dec.layers[0].w),
  b0: Array.from(dec.layers[0].b),
  w1: Array.from(dec.layers[1].w),
  b1: Array.from(dec.layers[1].b),
}}));
"""
    js_dec = json.loads(subprocess.check_output(["node", "-e", decode_script], text=True))
    py_w0 = py_weights[0].reshape(-1).tolist()
    py_b0 = py_biases[0].reshape(-1).tolist()
    py_w1 = py_weights[1].reshape(-1).tolist()
    py_b1 = py_biases[1].reshape(-1).tolist()
    assert max(abs(a - b) for a, b in zip(py_w0, js_dec["w0"])) < 1e-5
    assert max(abs(a - b) for a, b in zip(py_b0, js_dec["b0"])) < 1e-5
    assert max(abs(a - b) for a, b in zip(py_w1, js_dec["w1"])) < 1e-5
    assert max(abs(a - b) for a, b in zip(py_b1, js_dec["b1"])) < 1e-5

    # 4. Verify pop=16 HalfCheetah converges to upright sprint >= 8.0m reliably within 20 gens
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    pop = create_task_population("halfcheetah", 16)
    best_summary = None
    best_genome = pop[0]
    for _gen in range(20):
        evals = evaluate_control_batch("halfcheetah", pop, seeds=(0, 1), include_primary_rollout=False)
        fits = [e["summary"]["primaryValue"] * max(0.15, e["summary"]["meanUpright"]) for e in evals]
        top_idx = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
        cur_best_i = top_idx[0]
        if best_summary is None or fits[cur_best_i] > best_summary["primaryValue"] * max(0.15, best_summary["meanUpright"]):
            best_summary = evals[cur_best_i]["summary"]
            best_genome = pop[cur_best_i]
        if best_summary["meanDistance"] >= 8.0 and best_summary["meanUpright"] >= 0.40:
            break
        winners = set(top_idx[:4])
        scores = [max(1, 5 - top_idx[:4].index(i)) if i in winners else 0 for i in range(len(pop))]
        ups = [1 if i in winners else 0 for i in range(len(pop))]
        pop, _ = evolve_population(pop, scores, ups, mut_sigma=0.35, pop_size=16, elite_frac=0.05, crossover_rate=0.75)
    assert best_summary is not None
    assert best_summary["meanDistance"] >= 8.0
    assert best_summary["meanUpright"] >= 0.40

    # 5. Client-locked policy enforcement for HalfCheetah + direct attach_halfcheetah_geoms + CartPole server fallback
    from direct_evolution_together.environments import halfcheetah_rollout, attach_halfcheetah_geoms

    locked_batch = client.post(
        "/api/sim/eval-batch/",
        data=json.dumps({"task": "halfcheetah", "genomes": [best_genome], "includeRollout": True}),
        content_type="application/json",
    )
    assert locked_batch.status_code == 400
    assert "locked" in locked_batch.json()["error"].lower()

    locked_ro = client.post(
        "/api/sim/rollout/",
        data=json.dumps({"task": "halfcheetah", "genome": best_genome, "seed": 0}),
        content_type="application/json",
    )
    assert locked_ro.status_code == 400
    assert "locked" in locked_ro.json()["error"].lower()

    direct_evals = evaluate_control_batch("halfcheetah", [best_genome], seeds=(0,), include_primary_rollout=True)
    assert direct_evals[0]["summary"]["meanDistance"] >= 8.0
    assert len(direct_evals[0]["primaryRollout"]["frames"]) == 201

    ro_direct = attach_halfcheetah_geoms(halfcheetah_rollout(best_genome, seed=0, max_steps=200, nudge="fwd"))
    assert len(ro_direct["geomSpecs"]) == 8  # 8 non-plane capsules of half_cheetah.xml
    assert len(ro_direct["geomPoses"]) == 201 and len(ro_direct["geomPoses"][0]) == 8 * 12

    cp_genome = create_task_population("cartpole", 1)[0]
    cp_batch = client.post(
        "/api/sim/eval-batch/",
        data=json.dumps({"task": "cartpole", "genomes": [cp_genome], "includeRollout": True}),
        content_type="application/json",
    )
    assert cp_batch.status_code == 200 and cp_batch.json()["engine"] == "server"
    cp_ro = client.post(
        "/api/sim/rollout/",
        data=json.dumps({"task": "cartpole", "genome": cp_genome, "seed": 0, "nudge": "left"}),
        content_type="application/json",
    )
    assert cp_ro.status_code == 200 and cp_ro.json()["ok"] is True

    # 6. Lazy MP4 render + zip bundle export
    vid_resp = client.post(
        "/api/render/video/",
        data=json.dumps({"task": "halfcheetah", "genome": best_genome, "seed": 0}),
        content_type="application/json",
    )
    assert vid_resp.status_code == 200
    v_data = vid_resp.json()
    assert v_data["ok"] is True and v_data["videoUrl"].endswith(".mp4")
    assert v_data["posterUrl"].endswith(".png")
    media_mp4 = client.get(v_data["videoUrl"])
    assert media_mp4.status_code == 200
    mp4_bytes = b"".join(media_mp4.streaming_content)
    assert b"ftyp" in mp4_bytes[:64]

    bun_resp = client.post(
        "/export/bundle/",
        data=json.dumps({
            "genome": best_genome,
            "goal": "HalfCheetah",
            "generation": 15,
            "index": 0,
            "task": "halfcheetah",
        }),
        content_type="application/json",
    )
    assert bun_resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(bun_resp.content)) as zf:
        names = set(zf.namelist())
        run_names = [n for n in names if n.endswith("/run.py")]
        assert len(run_names) == 1
        assert any(n.endswith("/weights.pt") for n in names)
        assert any(n.endswith("/half_cheetah.xml") for n in names)
        compile(zf.read(run_names[0]).decode("utf-8"), "run.py", "exec")


@pytest.mark.django_db(transaction=True)
def test_external_researcher_worker_api_and_ws_broadcast(client):
    import base64
    # Minimal valid 1x1 PNG bytes
    tiny_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc`\x00\x00"
        b"\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    room_resp = client.post(
        "/api/rooms/",
        data=json.dumps({"name": "Duck Gait Lab", "task": "external", "population_size": 6}),
        content_type="application/json",
    )
    assert room_resp.status_code == 200
    rdata = room_resp.json()
    slug = rdata["slug"]
    token = rdata.get("worker_token") or ""
    assert len(token) >= 16

    # Reject missing or invalid worker token
    assert client.get(f"/api/rooms/{slug}/worker/jobs/").status_code == 403
    assert client.get(f"/api/rooms/{slug}/worker/jobs/", HTTP_AUTHORIZATION="Bearer wrong").status_code == 403

    # Downloadable worker script compiles cleanly
    ws_script_resp = client.get(f"/export/worker-script/?room={slug}")
    assert ws_script_resp.status_code == 200
    script_src = ws_script_resp.content.decode("utf-8")
    assert slug in script_src and "/worker/results/" in script_src
    compile(script_src, "worker.py", "exec")

    with _ws_server(seed=True, pop_size=6, max_votes=4) as port:
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2.0) as h_resp:
            assert json.loads(h_resp.read().decode("utf-8")) == {"status": "ok", "service": "ws"}
        seed_from_db(default_pop_size=6, default_max_votes=4)
        with connect(f"ws://127.0.0.1:{port}/ws/chat/{slug}/?username=researcher") as ws:
            evo = _recv_until(ws, "evolution_state")
            assert evo["goals"][0]["task"] == "external"

            jobs_resp = client.get(
                f"/api/rooms/{slug}/worker/jobs/",
                HTTP_AUTHORIZATION=f"Bearer {token}",
            )
            assert jobs_resp.status_code == 200
            jdata = jobs_resp.json()
            assert jdata["goal"] == evo["goals"][0]["name"]
            first_item = jdata["pending"][0]

            post_resp = client.post(
                f"/api/rooms/{slug}/worker/results/",
                data=json.dumps({
                    "goal": jdata["goal"],
                    "generation": 0,
                    "results": [{
                        "hash": first_item["hash"],
                        "summary": {
                            "primaryValue": 12.34,
                            "primaryDisplay": "12.3m",
                            "badgeText": "12.3m",
                            "tier": "solved",
                            "secondaryText": "custom microduck",
                        },
                        "posterPngB64": base64.b64encode(tiny_png).decode("ascii"),
                        "topology": {
                            "nodes": [
                                {"id": "s0", "kind": "sensor", "label": "imu_pitch"},
                                {"id": "h0", "kind": "hidden", "activation": "sin"},
                                {"id": "o0", "kind": "output", "activation": "tanh"},
                            ],
                            "edges": [
                                {"from": "s0", "to": "h0", "weight": 1.25},
                                {"from": "h0", "to": "o0", "weight": -0.8},
                            ],
                        },
                    }],
                }),
                content_type="application/json",
                HTTP_AUTHORIZATION=f"Bearer {token}",
            )
            assert post_resp.status_code == 200

            ws_msg = _recv_until(ws, "worker_eval_updated")
            assert ws_msg["items"][0]["hash"] == first_item["hash"]
            assert ws_msg["items"][0]["summary"]["badgeText"] == "12.3m"
            assert ws_msg["items"][0]["topology"]["nodes"][0]["label"] == "imu_pitch"

    evals_resp = client.get(f"/api/rooms/{slug}/worker/evals/?goal={jdata['goal']}&generation=0")
    assert evals_resp.status_code == 200
    edata = evals_resp.json()
    assert len(edata["items"]) == 1
    assert edata["items"][0]["posterDataUrl"].startswith("data:image/png;base64,")
    assert len(edata["items"][0]["topology"]["edges"]) == 2


@pytest.mark.django_db
def test_vendored_mujoco_assets_and_network_viz_dag(client):
    from pathlib import Path
    import re
    import tempfile

    # 1. Bundled frontend & vendored MuJoCo WASM assets serve cleanly with proper MIME types
    asset_checks = [
        ("/assets/cppn_core.js", "javascript"),
        ("/assets/sim_worker.js", "javascript"),
        ("/assets/cheetah_renderer.js", "javascript"),
        ("/assets/mujoco_gl.js", "javascript"),
        ("/assets/network_viz.js", "javascript"),
        ("/assets/half_cheetah.xml", "xml"),
        ("/assets/vendor/mujoco/mujoco.js", "javascript"),
        ("/assets/vendor/mujoco/mujoco.wasm", "wasm"),
    ]
    for url, kind in asset_checks:
        resp = client.get(url)
        assert resp.status_code == 200, f"{url} returned {resp.status_code}"
        ctype = resp["Content-Type"].lower()
        assert kind in ctype, f"{url} unexpected Content-Type {ctype}"
    assert client.get("/assets/../settings.py").status_code == 404

    # Verify gzip pre-compression, immutable caching, & ETag 304 Not Modified on vendored WASM
    wasm_gz = client.get("/assets/vendor/mujoco/mujoco.wasm", HTTP_ACCEPT_ENCODING="gzip")
    assert wasm_gz.status_code == 200
    assert wasm_gz.get("Content-Encoding") == "gzip"
    assert "immutable" in wasm_gz.get("Cache-Control", "")
    assert len(wasm_gz.content) < 5 * 1024 * 1024  # 11.2MB WASM compressed under 5MB
    wasm_304 = client.get(
        "/assets/vendor/mujoco/mujoco.wasm",
        HTTP_ACCEPT_ENCODING="gzip",
        HTTP_IF_NONE_MATCH=wasm_gz["ETag"],
    )
    assert wasm_304.status_code == 304

    # 2. Index template exposes local MuJoCo WASM URL, lazy/pooled loader, 3D GL canvas, & Network Inspector
    page_html = client.get("/").content.decode("utf-8")
    assert "/assets/vendor/mujoco/mujoco.js" in page_html
    assert "/assets/cppn_core.js" in page_html
    assert "/assets/mujoco_gl.js" in page_html
    assert "/assets/network_viz.js" in page_html
    assert "ensureAssetScript" in page_html
    assert "callSimWorkerPoolEval" in page_html
    assert "scheduleLazyLineageThumbs" in page_html
    assert 'id="sim-mode-bar"' in page_html
    assert 'id="crowd-trust-pill"' in page_html
    assert 'id="det-crowd-config"' in page_html
    assert 'bottom: 0;' in page_html and 'position: fixed;' in page_html
    assert 'id="fs-view-tabs"' in page_html
    assert 'id="fs-network-panel"' in page_html
    assert 'id="fs-gl-canvas"' in page_html
    assert 'value="halfcheetah"' in page_html and 'value="external"' in page_html

    # 3. Verify all JS assets + inline <script> syntax cleanly under Node
    pkg_dir = Path(__file__).resolve().parent.parent / "direct_evolution_together"
    for fname in ("cppn_core.js", "sim_worker.js", "cheetah_renderer.js", "mujoco_gl.js", "network_viz.js"):
        subprocess.run(["node", "--check", str(pkg_dir / "frontend" / fname)], check=True)

    html_raw = (pkg_dir / "templates" / "direct_evolution_together" / "index.html").read_text()
    inline_js = html_raw.split("<script>\n", 1)[1].rsplit("</script>", 1)[0]
    inline_js = re.sub(r"\{\{[^}]+\}\}", "template_val", inline_js)
    with tempfile.NamedTemporaryFile(suffix=".js", delete=False, mode="w") as tf:
        tf.write(inline_js)
        tmp_js = tf.name
    try:
        subprocess.run(["node", "--check", tmp_js], check=True)
    finally:
        Path(tmp_js).unlink(missing_ok=True)

    # 4. Verify DetNetworkViz handles HyperNEAT substrates, CPPN graphs, and arbitrary NEAT DAGs + cycles
    nv_path = str(pkg_dir / "frontend" / "network_viz.js")
    dag_check = f"""
const nv = require({json.dumps(nv_path)});
const neat = {{
  nodes: [
    {{ id: "in0", kind: "sensor", label: "z" }},
    {{ id: "in1", kind: "bias", label: "1" }},
    {{ id: "h1", kind: "hidden", activation: "sin", bias: 0.2 }},
    {{ id: "h2", kind: "hidden", activation: "gauss", bias: -0.1 }},
    {{ id: "out0", kind: "output", activation: "tanh", bias: 0.0 }}
  ],
  edges: [
    {{ from: "in0", to: "h1", weight: 1.5, enabled: true }},
    {{ from: "h1", to: "h2", weight: -0.7, enabled: true }},
    {{ from: "in0", to: "out0", weight: 0.4, enabled: true }},
    {{ from: "h2", to: "out0", weight: 1.2, enabled: true }},
    {{ from: "h2", to: "h1", weight: -0.5, enabled: true }}
  ]
}};
const g = nv.buildNeatDagGraph(neat, {{ obsLabels: ["z", "1"], actionLabels: ["act"] }});
const byId = Object.fromEntries(g.nodes.map(n => [n.id, n]));
const rec = g.edges.filter(e => e.recurrent);
console.log(JSON.stringify({{
  dims: g.dims,
  layers: {{ in0: byId.in0.layer, h1: byId.h1.layer, h2: byId.h2.layer, out0: byId.out0.layer }},
  recurrentCount: rec.length,
  matrixShape: [g.matrices[0].dIn, g.matrices[0].dOut],
}}));
"""
    dag_res = json.loads(subprocess.check_output(["node", "-e", dag_check], text=True))
    assert dag_res["layers"]["in0"] == 0
    assert dag_res["layers"]["h1"] < dag_res["layers"]["h2"] < dag_res["layers"]["out0"]
    assert dag_res["recurrentCount"] == 1
    assert dag_res["matrixShape"] == [5, 5]


@pytest.mark.django_db
def test_centralized_config_client_lock_and_crowd_trust(client, monkeypatch, tmp_path):
    from django.test import Client
    from direct_evolution_together.config import reload_config
    from direct_evolution_together.environments import get_task_spec
    from direct_evolution_together.models import ContributorTrust, ExternalEvalResult

    # 1. Inspect default /api/config/ payload
    cfg_resp = client.get("/api/config/")
    assert cfg_resp.status_code == 200
    cdata = cfg_resp.json()
    assert cdata["ok"] is True
    assert cdata["trust"]["quorum"] == 2
    assert cdata["environments"]["halfcheetah"]["execution_policy"] == "client_locked"
    assert cdata["environments"]["halfcheetah"]["simModes"] == ["client"]
    assert cdata["environments"]["halfcheetah"]["crowd_contribute"] is True
    assert cdata["environments"]["external"]["enabled"] is True

    # 2. Centralized JSON file + env overrides via reload_config()
    cfg_file = tmp_path / "det_config.json"
    cfg_file.write_text(
        json.dumps({
            "trust": {"quorum": 3, "min_trusted_reputation": 0.4},
            "environments": {
                "halfcheetah": {"default_pop_size": 12},
                "external": {"enabled": False},
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("DET_CONFIG_FILE", str(cfg_file))
    monkeypatch.setenv("DET_CROWD_QUORUM", "2")
    monkeypatch.setenv("DET_EXECUTION_POLICY_HALFCHEETAH", "client_preferred")
    monkeypatch.setenv("DET_MIN_VOTES_HALFCHEETAH", "12")
    try:
        reload_config()
        hc = get_task_spec("halfcheetah")
        ext = get_task_spec("external")
        assert hc.default_pop_size == 12
        assert hc.default_min_votes == 12
        assert hc.execution_policy == "client_preferred"
        assert hc.sim_modes == ("client", "server")
        assert ext.enabled is False

        cfg_override = client.get("/api/config/").json()
        assert cfg_override["trust"]["quorum"] == 2
        assert cfg_override["environments"]["external"]["enabled"] is False
    finally:
        monkeypatch.delenv("DET_CONFIG_FILE", raising=False)
        monkeypatch.delenv("DET_CROWD_QUORUM", raising=False)
        monkeypatch.delenv("DET_EXECUTION_POLICY_HALFCHEETAH", raising=False)
        monkeypatch.delenv("DET_MIN_VOTES_HALFCHEETAH", raising=False)
        reload_config()

    # 3. Crowd consensus + reputation reward/penalty + Byzantine quarantine across 3 sessions
    peer1, peer2, peer3 = Client(), Client(), Client()
    for p in (peer1, peer2, peer3):
        init = p.get("/api/rooms/lobby/crowd/evals/?goal=HalfCheetah&generation=0")
        assert init.status_code == 200 and len(init.json()["contributorId"]) >= 8

    def _submit(peer, gh, val):
        return peer.post(
            "/api/rooms/lobby/crowd/results/",
            data=json.dumps({
                "goal": "HalfCheetah",
                "generation": 0,
                "results": [{
                    "hash": gh,
                    "summary": {
                        "primaryValue": val,
                        "meanDistance": val,
                        "meanUpright": 0.72,
                        "primaryDisplay": f"+{val:.1f}m",
                        "badgeText": f"+{val:.1f}m",
                        "tier": "mid",
                    },
                }],
            }),
            content_type="application/json",
        )

    gh1 = "aa11bb22cc33dd44ee55"
    r1 = _submit(peer1, gh1, 4.50).json()
    assert r1["verifiedCount"] == 0
    assert r1["items"][0]["trust"]["status"] == "pending"

    # Peer 2 submits Byzantine outlier (+99.9m) -> disputed
    r2 = _submit(peer2, gh1, 99.90).json()
    assert r2["verifiedCount"] == 0
    assert r2["items"][0]["trust"]["status"] == "disputed"

    # Peer 3 agrees with Peer 1 (+4.54m) -> quorum reached (median = 4.52), rewards Peer1/Peer3, penalizes Peer2
    r3 = _submit(peer3, gh1, 4.54).json()
    assert r3["verifiedCount"] == 1
    assert r3["items"][0]["trust"]["status"] == "verified"
    assert abs(r3["items"][0]["summary"]["primaryValue"] - 4.52) < 1e-3
    assert r3["contributorReputation"] > 0.80

    # Check trusted_only=1 returns only verified consensus
    trusted = peer1.get("/api/rooms/lobby/crowd/evals/?goal=HalfCheetah&generation=0&trusted_only=1").json()
    assert len(trusted["items"]) == 1 and trusted["items"][0]["hash"] == gh1

    # Two more outlier submissions quarantine Peer2 (0.80 - 0.18*3 = 0.26 < 0.35)
    for gh_k in ("bb11cc22dd33ee44ff55", "cc11dd22ee33ff44aa55"):
        _submit(peer1, gh_k, 3.0)
        _submit(peer2, gh_k, 88.0)
        _submit(peer3, gh_k, 3.04)

    peer2_check = _submit(peer2, "dd11ee22ff33aa44bb55", 55.0).json()
    assert peer2_check["quarantined"] is True
    assert peer2_check["accepted"] == 0
    assert not ExternalEvalResult.objects.filter(room="lobby", genome_hash="dd11ee22ff33aa44bb55").exists()
    assert ContributorTrust.objects.filter(room="lobby", quarantined=True).count() == 1


@pytest.mark.django_db(transaction=True)
def test_ws_cid_reload_no_duplicate_leave_and_ping_reaper():
    import base64
    import os
    import socket as py_socket
    import time

    with _ws_server(seed=True, pop_size=6, max_votes=4) as port:
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2.0) as h_resp:
            assert json.loads(h_resp.read().decode("utf-8")) == {"status": "ok", "service": "ws"}
        seed_from_db(default_pop_size=6, default_max_votes=4)
        mummy_ws.set_round_config(2, 120)

        with connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=observer&cid=obscid000001") as obs:
            assert _recv_until(obs, "identity")["username"] == "observer"
            _recv_until(obs, "evolution_state")
            assert _recv_until(obs, "presence")["users"] == ["observer"]

            # 1. Tab connects with cid, casts vote, then reloads BEFORE old socket closes:
            #    must keep exact username "alice" (never "alice-2"), no duplicate presence, and preserve vote
            with connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=alice&cid=alicecid0001") as ws1:
                assert _recv_until(ws1, "identity")["username"] == "alice"
                _recv_until(ws1, "evolution_state")
                pres1 = _recv_until(obs, "presence")
                assert sorted(pres1["users"]) == ["alice", "observer"]

                ws1.send(json.dumps({
                    "type": "cast_vote", "goal": "HalfCheetah", "generation": 0, "index": 0, "vote": 1,
                }))
                v1 = _recv_until(obs, "votes_updated")
                assert v1["totalVotes"] == 1
                assert v1["votes"][0]["upUsers"] == ["alice"]

                # Open replacement socket with same cid while ws1 is still open (reload race)
                with connect(
                    f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=alice&cid=alicecid0001&reconnect=1"
                ) as ws2:
                    assert _recv_until(ws2, "identity")["username"] == "alice"
                    evo2 = _recv_until(ws2, "evolution_state")
                    pres2 = _recv_until(ws2, "presence")
                    assert sorted(pres2["users"]) == ["alice", "observer"]
                    hc_goal = {g["name"]: g for g in evo2["goals"]}["HalfCheetah"]
                    assert hc_goal["votes"][0]["upUsers"] == ["alice"]

                    # Idempotent unvote via vote=0 works on the transferred vote
                    ws2.send(json.dumps({
                        "type": "cast_vote", "goal": "HalfCheetah", "generation": 0, "index": 0, "vote": 0,
                    }))
                    v_cleared = _recv_until(ws2, "votes_updated")
                    assert v_cleared["totalVotes"] == 0
                    assert v_cleared["votes"][0]["upUsers"] == []

                    # 2. RFC 6455 Ping -> Pong and JSON ping -> pong
                    pong_waiter = ws2.ping(b"keepalive")
                    assert pong_waiter.wait(timeout=2.0)
                    ws2.send(json.dumps({"type": "ping"}))
                    assert _recv_until(ws2, "pong")["type"] == "pong"

            # 3. Explicit {"type": "leave"} disconnects user immediately before TCP close
            with connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=bob&cid=bobcid000002") as ws_bob:
                assert _recv_until(ws_bob, "identity")["username"] == "bob"
                _recv_until(ws_bob, "evolution_state")
                while True:
                    p = _recv_until(obs, "presence")
                    if "bob" in p["users"]:
                        break

                ws_bob.send(json.dumps({"type": "leave"}))
                while True:
                    left_sys = _recv_until(obs, "system", timeout=2.0)
                    if left_sys["message"] == "bob left":
                        break
                p_after_leave = _recv_until(obs, "presence", timeout=2.0)
                assert "bob" not in p_after_leave["users"]

            # 4. Dead-socket reaper evicts silent half-open TCP clients while keeping responsive clients live
            mummy_ws.set_heartbeat_config(100, 350)
            raw_sock = py_socket.create_connection(("127.0.0.1", port), timeout=2.0)
            try:
                sec_key = base64.b64encode(os.urandom(16)).decode("ascii")
                handshake = (
                    f"GET /ws/chat/lobby/?username=ghost&cid=ghostcid0003 HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{port}\r\n"
                    f"Upgrade: websocket\r\n"
                    f"Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Key: {sec_key}\r\n"
                    f"Sec-WebSocket-Version: 13\r\n\r\n"
                ).encode("ascii")
                raw_sock.sendall(handshake)
                resp_bytes = raw_sock.recv(4096)
                assert b"101 Switching Protocols" in resp_bytes

                while True:
                    p = _recv_until(obs, "presence", timeout=2.0)
                    if "ghost" in p["users"]:
                        break

                # Do not read/reply on raw_sock; meanwhile obs answers Ping automatically via websockets
                deadline = time.monotonic() + 2.5
                evicted = False
                while time.monotonic() < deadline:
                    p = _recv_until(obs, "presence", timeout=2.0)
                    if "ghost" not in p["users"] and "observer" in p["users"]:
                        evicted = True
                        break
                assert evicted, "Unresponsive ghost socket was not reaped within timeout"
            finally:
                mummy_ws.set_heartbeat_config(20000, 60000)
                try:
                    raw_sock.close()
                except Exception:
                    pass


@pytest.mark.django_db(transaction=True)
def test_halfcheetah_fast_rollout_parity_and_fault_tolerant_evolve(monkeypatch):
    import time
    from pathlib import Path
    from direct_evolution_together import ws_bridge
    from direct_evolution_together.environments.adapters import HalfCheetahAdapter
    from direct_evolution_together.environments.substrate import HyperNEATSubstrate
    from direct_evolution_together.models import RoomEvolutionState

    # 1. Verify HyperNEAT NumPy decode parity vs PyTorch CPPN and HalfCheetah capture_frames=False parity
    import random
    random.seed(777)
    torch.manual_seed(777)
    genome = CPPN.random(777).to_dict()
    sub = HyperNEATSubstrate((21, 12, 6))
    sub._decode_cache.clear()
    layers_np = sub.decode(genome)
    layers_pt = sub.decode(CPPN.from_dict(genome))
    for (w_np, b_np), (w_pt, b_pt) in zip(layers_np, layers_pt):
        assert abs(float((w_np - w_pt).max())) < 1e-5
        assert abs(float((b_np - b_pt).max())) < 1e-5

    adapter = HalfCheetahAdapter()
    ro_full = adapter.rollout(genome, seed=0, max_steps=200, capture_frames=True)
    ro_fast = adapter.rollout(genome, seed=0, max_steps=200, capture_frames=False)
    assert len(ro_full["frames"]) == 201
    assert ro_fast["frames"] == []
    assert abs(ro_full["distance"] - ro_fast["distance"]) < 1e-12
    assert abs(ro_full["return"] - ro_fast["return"]) < 1e-12

    # Verify cheetah_renderer.js FK segment output is finite and well-formed
    renderer_js = str(
        Path(__file__).resolve().parent.parent / "direct_evolution_together" / "frontend" / "cheetah_renderer.js"
    )
    fk_out = json.loads(subprocess.check_output([
        "node", "-e",
        f"global.window = global; require({json.dumps(renderer_js)}); "
        f"const segs = window.DetCheetahRenderer.cheetahFkSegments({json.dumps(ro_full['frames'][-1])}); "
        f"console.log(JSON.stringify({{count: segs.length, names: segs.map(s => s[0])}}));",
    ], text=True))
    assert fk_out["count"] == 8
    assert fk_out["names"] == ["torso", "head", "bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]

    # 2. Verify Lobby generation > 0 control goals still sync configured minVotes & eliteFrac on boot
    RoomEvolutionState.objects.update_or_create(
        room="lobby",
        defaults={
            "state_json": json.dumps({
                "defaultParams": {
                    "maxVotesPerUser": 1,
                    "minVotes": 99,
                    "mutSigma": 0.30,
                    "eliteFrac": 0.25,
                    "crossoverRate": 0.75,
                },
                "goals": [
                    {
                        "name": "HalfCheetah",
                        "task": "halfcheetah",
                        "generation": 3,
                        "genomes": [
                            {**CPPN.random(100 + i).to_dict(), "id": i + 1, "gen": 3, "op": "mutation", "parents": []}
                            for i in range(6)
                        ],
                        "lineage": [],
                        "params": {
                            "maxVotesPerUser": 1,
                            "minVotes": 99,
                            "mutSigma": 0.30,
                            "eliteFrac": 0.25,
                            "crossoverRate": 0.75,
                        },
                    }
                ],
            }),
        },
    )

    with _ws_server(seed=False, pop_size=6, max_votes=1) as port:
        seed_from_db(default_pop_size=6, default_max_votes=1)
        mummy_ws.set_round_config(1, 80)

        snap = json.loads(mummy_ws.get_room_snapshot_json("lobby"))
        hc_snap = {g["name"]: g for g in snap["goals"]}["HalfCheetah"]
        assert hc_snap["generation"] == 3

        # 3. Simulate a transient error during evolve_population: room must NOT stay wedged on evolving=True
        real_evolve = ws_bridge.evolve_population
        fail_once = {"boom": True}

        def flaky_evolve(*args, **kwargs):
            if fail_once["boom"]:
                fail_once["boom"] = False
                raise RuntimeError("transient evolution error")
            return real_evolve(*args, **kwargs)

        monkeypatch.setattr(ws_bridge, "evolve_population", flaky_evolve)

        with connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=tester&cid=testcid00001") as ws:
            _recv_until(ws, "identity")
            evo = _recv_until(ws, "evolution_state")
            hc_evo = {g["name"]: g for g in evo["goals"]}["HalfCheetah"]
            assert hc_evo["generation"] == 3
            assert hc_evo["eliteFrac"] == 0.05

            ws.send(json.dumps({
                "type": "cast_vote", "goal": "HalfCheetah", "generation": 3, "index": 0, "vote": 1,
            }))
            # First countdown attempt triggers flaky_evolve -> cancel_goal_evolve resets evolving=False,
            # restarts countdown on preserved votes, and second attempt advances cleanly to generation 4!
            saw_recovery = False
            updated = None
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline:
                msg = _decode_ws_message(ws.recv(timeout=2.0))
                if msg.get("type") == "votes_updated" and msg.get("goal") == "HalfCheetah":
                    if not fail_once["boom"] and msg.get("evolving") is False:
                        saw_recovery = True
                elif msg.get("type") == "goal_updated" and msg.get("goal", {}).get("name") == "HalfCheetah":
                    updated = msg["goal"]
                    break
            assert not fail_once["boom"], "flaky_evolve was never invoked"
            assert saw_recovery, "Failed evolve job did not reset goal.evolving to False"
            assert updated is not None and updated["generation"] == 4


@pytest.mark.django_db
def test_crowd_trust_changed_items_dedup_and_ws_nginx_timeout():
    from pathlib import Path
    from direct_evolution_together.crowd_trust import submit_eval_attestations

    r1 = submit_eval_attestations(
        room="lobby",
        goal="HalfCheetah",
        generation=0,
        contributor_id="peer_alpha_01",
        results=[{
            "hash": "11223344556677889900",
            "summary": {"primaryValue": 5.2, "meanDistance": 5.2, "meanUpright": 0.8},
        }],
    )
    assert len(r1["changedItems"]) == 1
    assert r1["changedItems"][0]["trust"]["status"] == "pending"

    r2 = submit_eval_attestations(
        room="lobby",
        goal="HalfCheetah",
        generation=0,
        contributor_id="peer_beta_02",
        results=[{
            "hash": "11223344556677889900",
            "summary": {"primaryValue": 5.2, "meanDistance": 5.2, "meanUpright": 0.8},
        }],
    )
    assert len(r2["changedItems"]) == 1
    assert r2["changedItems"][0]["trust"]["status"] == "verified"

    # Subsequent spot-check with identical consensus summary does not trigger redundant broadcast items
    r3 = submit_eval_attestations(
        room="lobby",
        goal="HalfCheetah",
        generation=0,
        contributor_id="peer_gamma_03",
        results=[{
            "hash": "11223344556677889900",
            "summary": {"primaryValue": 5.2, "meanDistance": 5.2, "meanUpright": 0.8},
        }],
    )
    assert len(r3["items"]) == 1
    assert r3["changedItems"] == []

    nginx_tpl = (Path(__file__).resolve().parent.parent / "deploy" / "nginx.conf.tpl").read_text(encoding="utf-8")
    assert "proxy_read_timeout 120;" in nginx_tpl
    assert "proxy_send_timeout 86400;" in nginx_tpl


@pytest.mark.django_db
def test_general_scene_compiler_so101_and_microduck_assets(client, tmp_path):
    from pathlib import Path
    import struct
    import mujoco
    import numpy as np
    from direct_evolution_together.environments.scene_assets import (
        compile_mjcf_scene,
        unpack_vfs_entries,
    )

    # 1. Verify precompiled Microduck scene assets & endpoints serve cleanly with immutable ETag + gzip
    for url, kind in (
        ("/assets/scene_cache.js", "javascript"),
        ("/assets/scenes/microduck/manifest.json", "json"),
        ("/assets/scenes/microduck/physics.bin", "octet-stream"),
        ("/assets/scenes/microduck/visual.bin", "octet-stream"),
    ):
        resp = client.get(url, HTTP_ACCEPT_ENCODING="gzip")
        assert resp.status_code == 200, f"{url} returned {resp.status_code}"
        assert kind in resp["Content-Type"].lower()

    phys_gz = client.get("/assets/scenes/microduck/physics.bin", HTTP_ACCEPT_ENCODING="gzip")
    assert phys_gz.get("Content-Encoding") == "gzip"
    assert len(phys_gz.content) < 1_500_000  # ~1.1 MB gzipped down from 23 MB raw STLs
    phys_304 = client.get(
        "/assets/scenes/microduck/physics.bin",
        HTTP_ACCEPT_ENCODING="gzip",
        HTTP_IF_NONE_MATCH=phys_gz["ETag"],
    )
    assert phys_304.status_code == 304

    manifest = client.get("/api/sim/scene/microduck/").json()
    assert manifest["ok"] is True
    assert manifest["sceneId"] == "microduck"
    assert manifest["visualStats"]["bodyMeshCount"] == 15
    assert manifest["visualStats"]["totalTris"] > 10_000

    # 2. Generalizability test: compile a synthetic so101-style 4-DOF robotic arm MJCF + binary STL
    #    WITHOUT explicit <inertial> tags -> verify inertial baking, visual mesh stripping, & 0.0 physics error!
    def _write_tetra_stl(stl_path: Path, scale: float) -> None:
        v = np.array(
            [
                [0.0, 0.0, 0.0],
                [scale, 0.0, 0.0],
                [0.0, scale, 0.0],
                [0.0, 0.0, scale],
            ],
            dtype=np.float32,
        )
        faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
        buf = bytearray(80)
        buf.extend(struct.pack("<I", len(faces)))
        for i0, i1, i2 in faces:
            n = np.cross(v[i1] - v[i0], v[i2] - v[i0])
            n /= max(1e-8, float(np.linalg.norm(n)))
            buf.extend(struct.pack("<12fH", *n, *v[i0], *v[i1], *v[i2], 0))
        stl_path.write_bytes(bytes(buf))

    src_dir = tmp_path / "so101_src"
    meshes_dir = src_dir / "meshes"
    meshes_dir.mkdir(parents=True)
    _write_tetra_stl(meshes_dir / "visual_housing.stl", 0.09)
    _write_tetra_stl(meshes_dir / "gripper_tip.stl", 0.03)

    entry_xml = src_dir / "so101_scene.xml"
    entry_xml.write_text(
        """<mujoco model="so101_arm">
  <compiler angle="radian" meshdir="meshes"/>
  <option timestep="0.005"/>
  <asset>
    <material name="orange_abs" rgba="0.95 0.45 0.12 1"/>
    <material name="dark_metal" rgba="0.18 0.20 0.24 1"/>
    <mesh name="housing_vis" file="visual_housing.stl"/>
    <mesh name="tip_col" file="gripper_tip.stl"/>
  </asset>
  <worldbody>
    <geom name="table" type="plane" size="1 1 0.05"/>
    <body name="shoulder_pan" pos="0 0 0.08">
      <joint name="j1" type="hinge" axis="0 0 1" damping="0.2"/>
      <geom type="mesh" mesh="housing_vis" material="orange_abs" contype="0" conaffinity="0" mass="0.25"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.02" material="dark_metal"/>
      <body name="upper_arm" pos="0 0 0.12">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.15"/>
        <geom type="mesh" mesh="housing_vis" material="orange_abs" contype="0" conaffinity="0" mass="0.18"/>
        <geom type="box" pos="0 0 0.08" size="0.018 0.018 0.08" material="orange_abs"/>
        <body name="forearm" pos="0 0 0.16">
          <joint name="j3" type="hinge" axis="0 1 0" damping="0.10"/>
          <geom type="cylinder" pos="0 0 0.06" size="0.014 0.06" material="dark_metal"/>
          <body name="wrist_gripper" pos="0 0 0.12">
            <joint name="j4" type="hinge" axis="1 0 0" damping="0.05"/>
            <geom type="mesh" mesh="tip_col" material="dark_metal" contype="1" conaffinity="1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="15"/>
    <position name="a2" joint="j2" kp="15"/>
    <position name="a3" joint="j3" kp="10"/>
    <position name="a4" joint="j4" kp="5"/>
  </actuator>
</mujoco>
""",
        encoding="utf-8",
    )

    out_dir = tmp_path / "so101_compiled"
    so_manifest = compile_mjcf_scene("so101", entry_xml, out_dir)
    assert so_manifest["sceneId"] == "so101"
    assert so_manifest["nq"] == 4 and so_manifest["nu"] == 4
    assert so_manifest["visualStats"]["bodyMeshCount"] == 4

    vfs_files = unpack_vfs_entries((out_dir / "physics.bin").read_bytes())
    assert "meshes/visual_housing.stl" not in vfs_files  # visual-only STL stripped
    assert "meshes/gripper_tip.stl" in vfs_files  # collision STL preserved
    stripped_xml = vfs_files["so101_scene.xml"].decode("utf-8")
    assert "<inertial " in stripped_xml  # baked automatically from full MjModel!

    # Verify bit-exact physics parity between original MJCF and stripped DETVFS1 archive
    orig_m = mujoco.MjModel.from_xml_path(str(entry_xml))
    orig_d = mujoco.MjData(orig_m)
    unpacked_dir = tmp_path / "so101_vfs"
    for rel_p, payload in vfs_files.items():
        dst = unpacked_dir / rel_p
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(payload)
    comp_m = mujoco.MjModel.from_xml_path(str(unpacked_dir / "so101_scene.xml"))
    comp_d = mujoco.MjData(comp_m)
    for step in range(100):
        ctrl = np.array([0.4 * np.sin(0.1 * step + k) for k in range(4)], dtype=np.float64)
        orig_d.ctrl[:] = ctrl
        comp_d.ctrl[:] = ctrl
        mujoco.mj_step(orig_m, orig_d)
        mujoco.mj_step(comp_m, comp_d)
    assert float(np.max(np.abs(orig_d.qpos - comp_d.qpos))) < 1e-9
    assert float(np.max(np.abs(orig_d.qvel - comp_d.qvel))) < 1e-9

    # Verify scene_cache.js parseVisualMeshpack parses both so101 and microduck visual.bin cleanly
    pkg_dir = Path(__file__).resolve().parent.parent / "direct_evolution_together"
    scene_cache_js = str(pkg_dir / "frontend" / "scene_cache.js")
    so_vis_path = str(out_dir / "visual.bin")
    md_vis_path = str(pkg_dir / "assets" / "scenes" / "microduck" / "visual.bin")
    parse_check = f"""
const fs = require("fs");
const sc = require({json.dumps(scene_cache_js)});
const b1 = fs.readFileSync({json.dumps(so_vis_path)});
const b2 = fs.readFileSync({json.dumps(md_vis_path)});
const p1 = sc.parseVisualMeshpack(b1.buffer.slice(b1.byteOffset, b1.byteOffset + b1.byteLength), {{}});
const p2 = sc.parseVisualMeshpack(b2.buffer.slice(b2.byteOffset, b2.byteOffset + b2.byteLength), {json.dumps(manifest)});
console.log(JSON.stringify({{
  soBodies: p1.bodyMeshes.length,
  mdBodies: p2.bodyMeshes.length,
  mdTris: p2.bodyMeshes.reduce((a, b) => a + b.numTris, 0),
}}));
"""
    parsed = json.loads(subprocess.check_output(["node", "-e", parse_check], text=True))
    assert parsed["soBodies"] == 4
    assert parsed["mdBodies"] == 15
    assert parsed["mdTris"] == manifest["visualStats"]["totalTris"]


@pytest.mark.django_db(transaction=True)
def test_populora_checkpoint_roundtrip_and_multi_round_crowd_evolution(client, settings):
    import time
    from django.core.files.uploadedfile import SimpleUploadedFile
    from django.test import Client

    # 1. Base model API + client-lock enforcement on server eval/rollout/video for microduck
    bm_resp = client.get("/api/sim/base-model/?task=microduck")
    assert bm_resp.status_code == 200
    base_model = bm_resp.json()
    assert base_model["ok"] is True
    assert base_model["arch"] == "gru_mlp"
    assert base_model["obsDim"] == 55 and base_model["actionDim"] == 14
    assert len(base_model["baseHash"]) == 16

    room_resp = client.post(
        "/api/rooms/",
        data=json.dumps({"name": "Duck Crowd Lab", "task": "microduck", "population_size": 8}),
        content_type="application/json",
    )
    assert room_resp.status_code == 200
    rdata = room_resp.json()
    slug = rdata["slug"]
    assert rdata["task"] == "microduck"
    assert rdata["population_size"] == 8

    # Client-locked enforcement: server refuses heavy Microduck physics & video
    sample_resp = client.get(rdata["url"])
    assert sample_resp.status_code == 200
    assert 'id="auto-evolve-btn"' in sample_resp.content.decode("utf-8")

    # Export .pt & .zip from initial room state and validate/reseed in another room
    from direct_evolution_together.models import RoomEvolutionState

    init_state = json.loads(RoomEvolutionState.objects.get(room=slug).state_json)
    g0 = init_state["goals"][0]["genomes"][0]
    assert g0["kind"] == "populora" and g0["baseHash"] == base_model["baseHash"]

    for bad_ep, body in (
        ("/api/sim/eval-batch/", {"task": "microduck", "genomes": [g0]}),
        ("/api/sim/rollout/", {"task": "microduck", "genome": g0, "seed": 0}),
        ("/api/render/video/", {"task": "microduck", "genome": g0, "seed": 0}),
    ):
        r_lock = client.post(bad_ep, data=json.dumps(body), content_type="application/json")
        assert r_lock.status_code == 400

    pt_exp = client.post(
        "/export/weights/",
        data=json.dumps({"goal": "Duck Crowd Lab", "task": "microduck", "generation": 0, "index": 0, "genome": g0}),
        content_type="application/json",
    )
    assert pt_exp.status_code == 200

    zip_exp = client.post(
        "/export/bundle/",
        data=json.dumps({"goal": "Duck Crowd Lab", "task": "microduck", "generation": 0, "index": 0, "genome": g0}),
        content_type="application/json",
    )
    assert zip_exp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(zip_exp.content)) as zf:
        znames = zf.namelist()
        assert any(n.endswith("/weights.pt") for n in znames)
        run_py = [n for n in znames if n.endswith("/run.py")][0]
        compile(zf.read(run_py).decode("utf-8"), "run.py", "exec")

    val_resp = client.post(
        "/api/weights/validate/",
        {
            "task": "microduck",
            "weights": [
                SimpleUploadedFile("duck0.pt", pt_exp.content),
                SimpleUploadedFile("duck0_bundle.zip", zip_exp.content),
            ],
        },
    )
    assert val_resp.status_code == 200
    v_strains = val_resp.json()["strains"]
    assert len(v_strains) == 2
    assert v_strains[0]["genome"]["hash"] == v_strains[1]["genome"]["hash"]
    assert v_strains[0]["genome"]["baseHash"] == g0["baseHash"]

    # Verify exported merged state_dict loads cleanly with strict=True into GRUMLPPolicy
    import copy
    from direct_evolution_together.environments.microduck import MICRODUCK_POLICY_SPEC
    from direct_evolution_together.populora_engine import get_or_create_base_model, save_populora_checkpoint_bytes

    strict_policy = copy.deepcopy(get_or_create_base_model(MICRODUCK_POLICY_SPEC)[0])
    orig_sd = torch.load(io.BytesIO(pt_exp.content), map_location="cpu", weights_only=True)
    strict_policy.load_state_dict(orig_sd, strict=True)
    re_saved_sd = torch.load(
        io.BytesIO(save_populora_checkpoint_bytes(v_strains[0]["genome"], MICRODUCK_POLICY_SPEC)),
        map_location="cpu",
        weights_only=True,
    )
    for k in orig_sd:
        assert float((orig_sd[k].float() - re_saved_sd[k].float()).abs().max()) < 5e-3

    reseed_room = client.post(
        "/api/rooms/",
        {
            "name": "Duck Reseeded",
            "task": "microduck",
            "population_size": "8",
            "weights": [SimpleUploadedFile("duck0.pt", pt_exp.content)],
        },
    )
    assert reseed_room.status_code == 200 and reseed_room.json()["seeded"] == 1

    # 2. Live WebSocket + Crowd Simulation Phase + Multi-Round PopuLoRA Evolution (pop=8)
    with _ws_server(seed=True, pop_size=8, max_votes=4) as port:
        settings.WS_HOST = "127.0.0.1"
        settings.WS_PORT = port
        seed_from_db(default_pop_size=8, default_max_votes=4)
        mummy_ws.set_round_config(2, 80)

        peer_a, peer_b = Client(), Client()

        def _submit_crowd_batch(http_peer, goal_name: str, gen: int, genomes: list[dict], offset: float = 0.0):
            results = []
            for idx, cand in enumerate(genomes):
                ret = 12.0 + 3.5 * idx + offset
                dist = 0.15 + 0.08 * idx
                traj = [[round(0.05 * (idx + 1) + 0.01 * t + 0.002 * d, 4) for d in range(16)] for t in range(12)]
                pose0 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] + [float(idx) * 0.02, 0.0, 0.14, 1.0, 0.0, 0.0, 0.0] + [0.0] * (14 * 7)
                results.append({
                    "hash": cand["hash"],
                    "summary": {
                        "return": ret,
                        "primaryValue": ret,
                        "meanDistance": dist,
                        "minDistance": dist * 0.9,
                        "meanSteps": 220.0 + idx * 10.0,
                        "primaryDisplay": f"{ret:+.1f} ret ({dist:+.2f}m)",
                        "badgeText": f"{dist:+.2f}m",
                        "tier": "mid",
                    },
                    "trajSample": traj,
                    "replay": {"bodyPoses": [pose0, pose0], "poseStride": 2, "vx_trace": [0.2, 0.25]},
                })
            resp = http_peer.post(
                f"/api/rooms/{slug}/crowd/results/",
                data=json.dumps({"goal": goal_name, "generation": gen, "results": results}),
                content_type="application/json",
            )
            assert resp.status_code == 200
            return resp.json()

        # Phase A: 2 peers connected -> adaptive quorum = 2; Phase 1 blocks blind votes, unlocks Phase 2 at 8/8!
        with (
            connect(f"ws://127.0.0.1:{port}/ws/chat/{slug}/?username=alice&cid=alice0000001") as ws_a,
            connect(f"ws://127.0.0.1:{port}/ws/chat/{slug}/?username=bob&cid=bob000000002") as ws_b,
        ):
            _recv_until(ws_a, "identity")
            evo0 = _recv_until(ws_a, "evolution_state")
            _recv_until(ws_b, "evolution_state")
            goal0 = evo0["goals"][0]
            gname = goal0["name"]
            assert goal0["generation"] == 0
            assert goal0["phase"] == "simulating"
            assert goal0["simPhaseEnabled"] is True
            assert goal0["autoEvolve"] is False
            assert len(goal0["genomes"]) == 8

            # Blind vote during Phase 1 ("simulating") must be rejected with simulations-running throttle
            ws_a.send(json.dumps({"type": "cast_vote", "goal": gname, "generation": 0, "index": 0, "vote": 1}))
            throttled = _recv_until(ws_a, "vote_throttled")
            assert "Simulations still running" in throttled["message"]

            # Peer A submits all 8 -> with 2 peers in room, quorum=2 so verifiedCount == 0 (still simulating)
            sub_a = _submit_crowd_batch(peer_a, gname, 0, goal0["genomes"], offset=0.0)
            assert sub_a["quorum"] == 2
            assert sub_a["verifiedCount"] == 0

            # Peer B corroborates all 8 -> quorum reached (8/8 verified) -> phase transitions to "voting"!
            sub_b = _submit_crowd_batch(peer_b, gname, 0, goal0["genomes"], offset=0.02)
            assert sub_b["quorum"] == 2
            assert sub_b["verifiedCount"] == 8

            unlocked = None
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                msg = _decode_ws_message(ws_a.recv(timeout=2.0))
                if msg.get("type") == "votes_updated" and msg.get("goal") == gname and msg.get("phase") == "voting":
                    unlocked = msg
                    break
            assert unlocked is not None
            assert unlocked["simCompleted"] == 8 and unlocked["simTotal"] == 8

            # Crowd evals endpoint returns peer 3D replay poses cleanly
            evals_g0 = peer_a.get(f"/api/rooms/{slug}/crowd/evals/?goal={gname}&generation=0").json()
            assert len(evals_g0["items"]) == 8
            assert "replay" in evals_g0["items"][0] and len(evals_g0["items"][0]["replay"]["bodyPoses"]) == 2

            # Cast 4 votes (half of pop=8) in Phase 2 -> triggers PopuLoRA evolution -> Gen 1!
            assert goal0["minVotes"] == 4
            ws_a.send(json.dumps({"type": "cast_vote", "goal": gname, "generation": 0, "index": 7, "vote": 1}))
            ws_b.send(json.dumps({"type": "cast_vote", "goal": gname, "generation": 0, "index": 6, "vote": 1}))
            ws_a.send(json.dumps({"type": "cast_vote", "goal": gname, "generation": 0, "index": 5, "vote": 1}))
            ws_b.send(json.dumps({"type": "cast_vote", "goal": gname, "generation": 0, "index": 4, "vote": 1}))
            gen1 = _recv_until(ws_a, "goal_updated", timeout=8.0)["goal"]
            assert gen1["generation"] == 1
            assert gen1["phase"] == "simulating"
            assert gen1["simCompleted"] == 0
            assert len(gen1["genomes"]) == 8
            assert all(g["kind"] == "populora" and g["baseHash"] == base_model["baseHash"] for g in gen1["genomes"])
            # Slot 0 champion backup preserves top candidate's weights/hash or evolves diverse LoRA deltas
            assert len({g["hash"] for g in gen1["genomes"]}) >= 6

        # Phase B: Solo client (only alice connected -> adaptive quorum = 1) + Auto-Evolve across 3 full rounds!
        with connect(f"ws://127.0.0.1:{port}/ws/chat/{slug}/?username=alice&cid=alice0000001") as ws_solo:
            _recv_until(ws_solo, "identity")
            cur_goal = _recv_until(ws_solo, "evolution_state")["goals"][0]
            assert cur_goal["generation"] == 1

            # Enable hands-free Auto-Evolve over WebSocket
            ws_solo.send(json.dumps({"type": "set_auto_evolve", "goal": gname, "enabled": True}))
            ae_upd = _recv_until(ws_solo, "votes_updated")
            assert ae_upd["autoEvolve"] is True

            # Simulate 3 consecutive generations (Gen 1 -> Gen 2 -> Gen 3 -> Gen 4) solo without casting manual votes!
            for expected_next_gen in (2, 3, 4):
                cur_gen = expected_next_gen - 1
                sub_solo = _submit_crowd_batch(peer_a, gname, cur_gen, cur_goal["genomes"])
                assert sub_solo["quorum"] == 1
                assert sub_solo["verifiedCount"] == 8

                next_goal = None
                deadline = time.monotonic() + 8.0
                while time.monotonic() < deadline:
                    msg = _decode_ws_message(ws_solo.recv(timeout=3.0))
                    if msg.get("type") == "goal_updated" and msg.get("goal", {}).get("name") == gname:
                        next_goal = msg["goal"]
                        break
                assert next_goal is not None, f"Auto-Evolve did not advance to Gen {expected_next_gen}"
                assert next_goal["generation"] == expected_next_gen
                assert next_goal["autoEvolve"] is True
                assert next_goal["phase"] == "simulating"
                assert len(next_goal["genomes"]) == 8
                cur_goal = next_goal


def test_microduck_wasm_worker_and_populora_network_viz_in_node(tmp_path):
    from pathlib import Path
    from direct_evolution_together.environments import get_task_specs_payload
    from direct_evolution_together.environments.microduck import MICRODUCK_POLICY_SPEC
    from direct_evolution_together.populora_engine import (
        create_populora_population,
        get_or_create_base_model,
    )

    pkg_dir = Path(__file__).resolve().parent.parent / "direct_evolution_together"
    _, base_hash, base_payload = get_or_create_base_model(MICRODUCK_POLICY_SPEC)
    genomes = create_populora_population("microduck", MICRODUCK_POLICY_SPEC, pop_size=8)
    assert len(genomes) == 8 and all(g["baseHash"] == base_hash for g in genomes)

    # 1. Verify DetNetworkViz builds low-rank PopuLoRA adapter graph + delta-W matrices
    nv_js = str(pkg_dir / "frontend" / "network_viz.js")
    specs = get_task_specs_payload()
    nv_check = f"""
const nv = require({json.dumps(nv_js)});
const model = nv.buildGraphModel({json.dumps(genomes[0])}, {json.dumps(specs["microduck"])});
console.log(JSON.stringify({{
  kind: model.kind,
  nodeCount: model.nodes.length,
  edgeCount: model.edges.length,
  matrixCount: model.matrices.length,
  firstMat: [model.matrices[0].dIn, model.matrices[0].dOut],
}}));
"""
    nv_out = json.loads(subprocess.check_output(["node", "-e", nv_check], text=True))
    assert nv_out["kind"] == "populora"
    assert nv_out["matrixCount"] == 4
    assert nv_out["firstMat"] == [16, 16]
    assert nv_out["edgeCount"] > 0

    # 2. Verify frontend/sim_worker.js executes all 8 PopuLoRA Microduck genomes in @mujoco/mujoco WASM
    worker_src = (pkg_dir / "frontend" / "sim_worker.js").read_text(encoding="utf-8")
    phys_bin = str(pkg_dir / "assets" / "scenes" / "microduck" / "physics.bin")
    wasm_js = (pkg_dir / "assets" / "vendor" / "mujoco" / "mujoco.js").as_uri()
    data_json = tmp_path / "worker_input.json"
    data_json.write_text(
        json.dumps({"baseModel": base_payload, "specs": specs, "genomes": genomes}),
        encoding="utf-8",
    )
    harness_mjs = tmp_path / "worker_harness.mjs"
    harness_mjs.write_text(
        f"""
import fs from "fs";
const physBuf = fs.readFileSync({json.dumps(phys_bin)});
const inp = JSON.parse(fs.readFileSync({json.dumps(str(data_json))}, "utf8"));
const responses = new Map();
let listener = null;
globalThis.self = {{
  location: {{ origin: "http://localhost:8000" }},
  addEventListener: (ev, fn) => {{ if (ev === "message") listener = fn; }},
  postMessage: (msg) => {{
    const r = responses.get(msg.reqId);
    if (r) r(msg);
  }},
}};
{worker_src}
function send(reqId, payload) {{
  return new Promise((resolve) => {{
    responses.set(reqId, resolve);
    listener({{ data: {{ reqId, ...payload }} }});
  }});
}}
const initRes = await send(1, {{
  type: "init",
  taskSpecs: inp.specs,
  mujocoWasmUrl: {json.dumps(wasm_js)},
  warmMujoco: true,
}});
if (!initRes.ok) throw new Error(initRes.error);
const batchRes = await send(2, {{
  type: "evalBatch",
  taskId: "microduck",
  genomes: inp.genomes,
  seeds: [0, 1],
  baseModel: inp.baseModel,
  physicsBuffer: physBuf.buffer.slice(physBuf.byteOffset, physBuf.byteOffset + physBuf.byteLength),
}});
if (!batchRes.ok) throw new Error(batchRes.error);
const roRes = await send(3, {{
  type: "rollout",
  taskId: "microduck",
  genome: inp.genomes[0],
  seed: 0,
  maxSteps: 60,
  nudge: "fwd",
  includeActivations: true,
  includeGeoms: true,
}});
if (!roRes.ok) throw new Error(roRes.error);
console.log(JSON.stringify({{
  engine: batchRes.engine,
  count: batchRes.items.length,
  hasTraj: batchRes.items.every((x) => Array.isArray(x.trajSample) && x.trajSample.length > 0),
  hasPoses: batchRes.items.every((x) => Array.isArray(x.primaryRollout.bodyPoses) && x.primaryRollout.bodyPoses[0].length === 15 * 7),
  roFrames: roRes.rollout.bodyPoses.length,
  roActLen: roRes.rollout.stepActivations.length,
  roActLayers: roRes.rollout.stepActivations[0].length,
}}));
""",
        encoding="utf-8",
    )
    wasm_out = json.loads(
        subprocess.check_output(["node", str(harness_mjs)], text=True)
    )
    assert wasm_out["engine"] == "client-wasm"
    assert wasm_out["count"] == 8
    assert wasm_out["hasTraj"] is True
    assert wasm_out["hasPoses"] is True
    assert wasm_out["roFrames"] >= 2
    assert wasm_out["roActLen"] >= 1
    assert wasm_out["roActLayers"] == 5


@pytest.mark.django_db
def test_unitree_go2_scene_assets_populora_and_wasm_worker(client, tmp_path):
    import math
    from pathlib import Path
    from django.core.files.uploadedfile import SimpleUploadedFile
    from direct_evolution_together.environments import (
        UNITREE_GO2_POLICY_SPEC,
        create_task_population,
        get_scene_manifest,
        get_task_specs_payload,
        unpack_vfs_entries,
    )
    from direct_evolution_together.populora_engine import (
        evolve_populora_population,
        get_or_create_base_model,
        reset_populora_goal_runtime,
    )

    pkg_dir = Path(__file__).resolve().parent.parent / "direct_evolution_together"

    # 1. Verify precompiled Unitree Go2 assets (27.7MB raw OBJs -> <20KB physics.bin + 13-body visual.bin)
    m_resp = client.get("/api/sim/scene/unitree_go2/")
    assert m_resp.status_code == 200
    manifest = m_resp.json()
    assert manifest["sceneId"] == "unitree_go2"
    assert manifest["physicsBytes"] < 20_000
    assert manifest["collisionMeshes"] == []
    assert manifest["visualStats"]["bodyMeshCount"] == 13

    phys_resp = client.get(f"/assets/{manifest['physicsAsset']}", HTTP_ACCEPT_ENCODING="gzip")
    vis_resp = client.get(f"/assets/{manifest['visualAsset']}", HTTP_ACCEPT_ENCODING="gzip")
    assert phys_resp.status_code == 200 and vis_resp.status_code == 200

    raw_phys = (pkg_dir / "assets" / "scenes" / "unitree_go2" / "physics.bin").read_bytes()
    vfs = unpack_vfs_entries(raw_phys)
    assert set(vfs.keys()) == {"go2_walk.xml", "scene_walk.xml"}
    assert not any(k.endswith(".obj") for k in vfs)

    # 2. Verify Unitree Go2 PopuLoRA base model, .pt export/upload roundtrip, & client_locked policy
    bm_resp = client.get("/api/sim/base-model/?task=unitree_go2&seed=1")
    assert bm_resp.status_code == 200
    bm_json = bm_resp.json()
    assert bm_json["obsDim"] == 49
    assert bm_json["actionDim"] == 12

    torch.manual_seed(7)
    genomes = create_task_population("unitree_go2", 8)
    assert len(genomes) == 8
    assert all(g["kind"] == "populora" and g["task"] == "unitree_go2" for g in genomes)

    exp_resp = client.post(
        "/export/weights/",
        data=json.dumps({
            "genome": genomes[0],
            "goal": "Unitree Go2",
            "generation": 0,
            "index": 0,
            "task": "unitree_go2",
        }),
        content_type="application/json",
    )
    assert exp_resp.status_code == 200
    val_resp = client.post(
        "/api/weights/validate/",
        data={
            "task": "unitree_go2",
            "files": [SimpleUploadedFile("unitree_go2_gen0_0.pt", exp_resp.content)],
        },
    )
    assert val_resp.status_code == 200
    v_data = val_resp.json()
    assert v_data["ok"] is True
    assert v_data["strains"][0]["genome"]["task"] == "unitree_go2"

    # Server refuses server-side rollout & video for client_locked unitree_go2
    ro_deny = client.post(
        "/api/sim/rollout/",
        data=json.dumps({"task": "unitree_go2", "genome": genomes[0], "seed": 0}),
        content_type="application/json",
    )
    assert ro_deny.status_code == 400 and ro_deny.json()["ok"] is False

    # 3. Verify frontend/sim_worker.js evaluates all 8 Go2 PopuLoRA genomes in @mujoco/mujoco WASM
    _, _, base_payload = get_or_create_base_model(UNITREE_GO2_POLICY_SPEC, base_seed=1)
    specs = get_task_specs_payload()
    worker_src = (pkg_dir / "frontend" / "sim_worker.js").read_text(encoding="utf-8")
    phys_bin = str(pkg_dir / "assets" / "scenes" / "unitree_go2" / "physics.bin")
    wasm_js = (pkg_dir / "assets" / "vendor" / "mujoco" / "mujoco.js").as_uri()
    data_json = tmp_path / "go2_worker_input.json"
    data_json.write_text(
        json.dumps({"baseModel": base_payload, "specs": specs, "genomes": genomes}),
        encoding="utf-8",
    )
    harness_mjs = tmp_path / "go2_worker_harness.mjs"
    harness_mjs.write_text(
        f"""
import fs from "fs";
const physBuf = fs.readFileSync({json.dumps(phys_bin)});
const inp = JSON.parse(fs.readFileSync({json.dumps(str(data_json))}, "utf8"));
const responses = new Map();
let listener = null;
globalThis.self = {{
  location: {{ origin: "http://localhost:8000" }},
  addEventListener: (ev, fn) => {{ if (ev === "message") listener = fn; }},
  postMessage: (msg) => {{
    const r = responses.get(msg.reqId);
    if (r) r(msg);
  }},
}};
{worker_src}
function send(reqId, payload) {{
  return new Promise((resolve) => {{
    responses.set(reqId, resolve);
    listener({{ data: {{ reqId, ...payload }} }});
  }});
}}
const initRes = await send(1, {{
  type: "init",
  taskSpecs: inp.specs,
  mujocoWasmUrl: {json.dumps(wasm_js)},
  warmMujoco: true,
  warmTaskId: "unitree_go2",
  physicsBuffer: physBuf.buffer.slice(physBuf.byteOffset, physBuf.byteOffset + physBuf.byteLength),
}});
if (!initRes.ok) throw new Error(initRes.error);
const batchRes = await send(2, {{
  type: "evalBatch",
  taskId: "unitree_go2",
  genomes: inp.genomes,
  seeds: [0, 1],
  baseModel: inp.baseModel,
  physicsBuffer: physBuf.buffer.slice(physBuf.byteOffset, physBuf.byteOffset + physBuf.byteLength),
}});
if (!batchRes.ok) throw new Error(batchRes.error);
const roRes = await send(3, {{
  type: "rollout",
  taskId: "unitree_go2",
  genome: inp.genomes[0],
  seed: 0,
  maxSteps: 60,
  nudge: "fwd",
  includeActivations: true,
}});
if (!roRes.ok) throw new Error(roRes.error);
console.log(JSON.stringify({{
  engine: batchRes.engine,
  count: batchRes.items.length,
  summaries: batchRes.items.map((x) => x.summary),
  trajs: batchRes.items.map((x) => x.trajSample),
  poseRowLen: batchRes.items[0].primaryRollout.bodyPoses[0].length,
  roFrames: roRes.rollout.bodyPoses.length,
  roStepTraceLen: roRes.rollout.stepTrace[0].length,
  roActLayers: roRes.rollout.stepActivations[0].length,
}}));
""",
        encoding="utf-8",
    )
    wasm_out = json.loads(subprocess.check_output(["node", str(harness_mjs)], text=True))
    assert wasm_out["engine"] == "client-wasm"
    assert wasm_out["count"] == 8
    assert wasm_out["poseRowLen"] == 13 * 7
    assert wasm_out["roFrames"] >= 2
    assert wasm_out["roStepTraceLen"] == 5
    assert wasm_out["roActLayers"] == 5
    assert all(len(t) == 16 and len(t[0]) == 22 for t in wasm_out["trajs"])
    assert all(
        isinstance(s["return"], (int, float)) and math.isfinite(s["return"]) and s["minSteps"] > 0
        for s in wasm_out["summaries"]
    )

    # 4. Verify PopuLoRA evolution step accepts the 22-D WASM Go2 trajectories cleanly
    reset_populora_goal_runtime("lobby", "Unitree Go2")
    eval_map = {
        genomes[i]["hash"]: {
            "summary": wasm_out["summaries"][i],
            "trajSample": wasm_out["trajs"][i],
        }
        for i in range(8)
    }
    next_genomes, top_score = evolve_populora_population(
        room="lobby",
        goal="Unitree Go2",
        task_id="unitree_go2",
        generation=0,
        policy_spec=UNITREE_GO2_POLICY_SPEC,
        genome_dicts=genomes,
        scores=[0] * 8,
        up_counts=[0] * 8,
        eval_by_hash=eval_map,
        pop_size=8,
        elite_frac=0.25,
    )
    assert len(next_genomes) == 8
    assert all(g["task"] == "unitree_go2" for g in next_genomes)
    assert isinstance(top_score, int) and top_score > -200


@pytest.mark.django_db
def test_so101_arm_scene_assets_populora_and_wasm_worker(client, tmp_path):
    import math
    from pathlib import Path
    from django.core.files.uploadedfile import SimpleUploadedFile
    from direct_evolution_together.environments import (
        SO101_POLICY_SPEC,
        create_task_population,
        get_task_specs_payload,
        unpack_vfs_entries,
    )
    from direct_evolution_together.populora_engine import (
        evolve_populora_population,
        get_or_create_base_model,
        reset_populora_goal_runtime,
    )
    from direct_evolution_together.ws_bridge import _ensure_lobby_default_goals

    pkg_dir = Path(__file__).resolve().parent.parent / "direct_evolution_together"

    # 1. Verify _ensure_lobby_default_goals ALWAYS keeps Microduck as default tab #0
    partial_lobby = json.dumps({
        "defaultParams": {"maxVotesPerUser": 20, "minVotes": 3, "mutSigma": 0.3, "eliteFrac": 0.25, "crossoverRate": 0.75},
        "goals": [
            {"name": "Unitree Go2", "task": "unitree_go2", "generation": 1, "genomes": create_task_population("unitree_go2", 8), "lineage": [], "params": {"maxVotesPerUser": 20, "minVotes": 3, "mutSigma": 0.3, "eliteFrac": 0.25, "crossoverRate": 0.75}},
            {"name": "Microduck", "task": "microduck", "generation": 0, "genomes": create_task_population("microduck", 8), "lineage": [], "params": {"maxVotesPerUser": 20, "minVotes": 3, "mutSigma": 0.3, "eliteFrac": 0.25, "crossoverRate": 0.75}},
        ],
    })
    upgraded_json, changed = _ensure_lobby_default_goals(partial_lobby, default_pop_size=8, default_max_votes=20)
    assert changed is True
    upgraded_names = [g["name"] for g in json.loads(upgraded_json)["goals"]]
    assert upgraded_names[:3] == ["Microduck", "Unitree Go2", "SO-102 Arm"]

    # 2. Verify precompiled SO-101 assets (17.9MB raw STLs -> <15KB physics.bin + 8-body visual.bin with cube)
    m_resp = client.get("/api/sim/scene/so101/")
    assert m_resp.status_code == 200
    manifest = m_resp.json()
    assert manifest["sceneId"] == "so101"
    assert manifest["physicsBytes"] < 15_000
    assert manifest["collisionMeshes"] == []
    assert manifest["visualStats"]["bodyMeshCount"] == 8
    assert "cube" in manifest["bodyNames"]

    raw_phys = (pkg_dir / "assets" / "scenes" / "so101" / "physics.bin").read_bytes()
    vfs = unpack_vfs_entries(raw_phys)
    assert set(vfs.keys()) == {"scene_pick.xml", "so101_arm.xml"}
    assert not any(k.endswith(".stl") for k in vfs)

    # 3. Verify SO-101 PopuLoRA base model & .pt export/upload roundtrip
    bm_resp = client.get("/api/sim/base-model/?task=so101&seed=1")
    assert bm_resp.status_code == 200
    bm_json = bm_resp.json()
    assert bm_json["obsDim"] == 33
    assert bm_json["actionDim"] == 6

    genomes = create_task_population("so101", 8)
    assert len(genomes) == 8
    assert all(g["kind"] == "populora" and g["task"] == "so101" for g in genomes)

    exp_resp = client.post(
        "/export/weights/",
        data=json.dumps({
            "genome": genomes[0],
            "goal": "SO-101 Arm",
            "generation": 0,
            "index": 0,
            "task": "so101",
        }),
        content_type="application/json",
    )
    assert exp_resp.status_code == 200
    val_resp = client.post(
        "/api/weights/validate/",
        data={
            "task": "so101",
            "files": [SimpleUploadedFile("so101_gen0_0.pt", exp_resp.content)],
        },
    )
    assert val_resp.status_code == 200
    assert val_resp.json()["strains"][0]["genome"]["task"] == "so101"

    # 4. Verify frontend/sim_worker.js evaluates all 8 SO-101 PopuLoRA genomes in @mujoco/mujoco WASM
    _, _, base_payload = get_or_create_base_model(SO101_POLICY_SPEC, base_seed=1)
    specs = get_task_specs_payload()
    worker_src = (pkg_dir / "frontend" / "sim_worker.js").read_text(encoding="utf-8")
    phys_bin = str(pkg_dir / "assets" / "scenes" / "so101" / "physics.bin")
    wasm_js = (pkg_dir / "assets" / "vendor" / "mujoco" / "mujoco.js").as_uri()
    data_json = tmp_path / "so101_worker_input.json"
    data_json.write_text(
        json.dumps({"baseModel": base_payload, "specs": specs, "genomes": genomes}),
        encoding="utf-8",
    )
    harness_mjs = tmp_path / "so101_worker_harness.mjs"
    harness_mjs.write_text(
        f"""
import fs from "fs";
const physBuf = fs.readFileSync({json.dumps(phys_bin)});
const inp = JSON.parse(fs.readFileSync({json.dumps(str(data_json))}, "utf8"));
const responses = new Map();
let listener = null;
globalThis.self = {{
  location: {{ origin: "http://localhost:8000" }},
  addEventListener: (ev, fn) => {{ if (ev === "message") listener = fn; }},
  postMessage: (msg) => {{
    const r = responses.get(msg.reqId);
    if (r) r(msg);
  }},
}};
{worker_src}
function send(reqId, payload) {{
  return new Promise((resolve) => {{
    responses.set(reqId, resolve);
    listener({{ data: {{ reqId, ...payload }} }});
  }});
}}
const initRes = await send(1, {{
  type: "init",
  taskSpecs: inp.specs,
  mujocoWasmUrl: {json.dumps(wasm_js)},
  warmMujoco: true,
  warmTaskId: "so101",
  envParams: {{ arm_model: "so101" }},
  physicsBuffer: physBuf.buffer.slice(physBuf.byteOffset, physBuf.byteOffset + physBuf.byteLength),
}});
if (!initRes.ok) throw new Error(initRes.error);
const batchRes = await send(2, {{
  type: "evalBatch",
  taskId: "so101",
  envParams: {{ arm_model: "so101" }},
  genomes: inp.genomes,
  seeds: [0, 1],
  baseModel: inp.baseModel,
  physicsBuffer: physBuf.buffer.slice(physBuf.byteOffset, physBuf.byteOffset + physBuf.byteLength),
}});
if (!batchRes.ok) throw new Error(batchRes.error);
const roRes = await send(3, {{
  type: "rollout",
  taskId: "so101",
  envParams: {{ arm_model: "so101" }},
  genome: inp.genomes[0],
  seed: 0,
  maxSteps: 60,
  nudge: "left",
  includeActivations: true,
}});
if (!roRes.ok) throw new Error(roRes.error);
console.log(JSON.stringify({{
  engine: batchRes.engine,
  count: batchRes.items.length,
  summaries: batchRes.items.map((x) => x.summary),
  trajs: batchRes.items.map((x) => x.trajSample),
  poseRowLen: batchRes.items[0].primaryRollout.bodyPoses[0].length,
  roFrames: roRes.rollout.bodyPoses.length,
  roStepTraceLen: roRes.rollout.stepTrace[0].length,
  roActLayers: roRes.rollout.stepActivations[0].length,
}}));
""",
        encoding="utf-8",
    )
    wasm_out = json.loads(subprocess.check_output(["node", str(harness_mjs)], text=True))
    assert wasm_out["engine"] == "client-wasm"
    assert wasm_out["count"] == 8
    assert wasm_out["poseRowLen"] == 8 * 7
    assert wasm_out["roFrames"] >= 2
    assert wasm_out["roStepTraceLen"] == 5
    assert wasm_out["roActLayers"] == 5
    assert all(len(t) == 16 and len(t[0]) == 15 for t in wasm_out["trajs"])
    assert all(
        isinstance(s["return"], (int, float)) and math.isfinite(s["return"])
        for s in wasm_out["summaries"]
    )

    # 5. Verify PopuLoRA evolution step accepts the 15-D WASM SO-101 trajectories cleanly
    reset_populora_goal_runtime("lobby", "SO-101 Arm")
    eval_map = {
        genomes[i]["hash"]: {
            "summary": wasm_out["summaries"][i],
            "trajSample": wasm_out["trajs"][i],
        }
        for i in range(8)
    }
    next_genomes, _ = evolve_populora_population(
        room="lobby",
        goal="SO-101 Arm",
        task_id="so101",
        generation=0,
        policy_spec=SO101_POLICY_SPEC,
        genome_dicts=genomes,
        scores=[0] * 8,
        up_counts=[0] * 8,
        eval_by_hash=eval_map,
        pop_size=8,
        elite_frac=0.25,
    )
    assert len(next_genomes) == 8
    assert all(g["task"] == "so101" for g in next_genomes)


@pytest.mark.django_db(transaction=True)
def test_lazy_active_tab_ws_loading_and_compact_crowd_wire(client):
    import time
    from direct_evolution_together.crowd_trust import fetch_room_eval_map

    def _recv_raw_type(ws, target_type: str, timeout: float = 5.0) -> tuple[str, dict]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            raw = ws.recv(timeout=max(0.05, deadline - time.monotonic()))
            text = gzip.decompress(raw).decode("utf-8") if isinstance(raw, bytes) else raw
            msg = json.loads(text)
            if msg.get("type") == target_type:
                return text, msg
        raise AssertionError(f"Timed out waiting for {target_type}")

    with _ws_server(seed=True) as port:
        mummy_ws.set_round_config(2, 80)
        with (
            connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=eager") as ws_eager,
            connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=lazy_def&lazy=1") as ws_lazy,
            connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=lazy_bfly&lazy=1&focus=Butterfly") as ws_bfly,
        ):
            raw_eager, evo_eager = _recv_raw_type(ws_eager, "evolution_state")
            raw_lazy, evo_lazy = _recv_raw_type(ws_lazy, "evolution_state")
            raw_bfly, evo_bfly = _recv_raw_type(ws_bfly, "evolution_state")

            assert all(g["genomesLoaded"] is True and len(g["genomes"]) > 0 for g in evo_eager["goals"])
            assert len(raw_eager) > 950_000

            # Default lazy client loads ONLY active first tab ("Microduck")
            assert evo_lazy["goals"][0]["name"] == "Microduck"
            assert evo_lazy["goals"][0]["genomesLoaded"] is True
            assert len(evo_lazy["goals"][0]["genomes"]) == 8
            for stub in evo_lazy["goals"][1:]:
                assert stub["genomesLoaded"] is False
                assert stub["genomes"] == []
                assert stub["lineage"] == []
                assert stub["populationSize"] >= 8
            assert len(raw_lazy) < 360_000

            # Focused lazy client loads ONLY "Butterfly" (< 75 KB raw vs > 950 KB eager)
            loaded_bfly = [g["name"] for g in evo_bfly["goals"] if g["genomesLoaded"]]
            assert loaded_bfly == ["Butterfly"]
            assert len(raw_bfly) < 75_000

            # Switching tab via select_goal hydrates "SO-102 Arm" on demand
            ws_lazy.send(json.dumps({"type": "select_goal", "goal": "SO-102 Arm", "needGenomes": True}))
            _, so_upd = _recv_raw_type(ws_lazy, "goal_updated")
            assert so_upd["goal"]["name"] == "SO-102 Arm"
            assert so_upd["goal"]["genomesLoaded"] is True
            assert len(so_upd["goal"]["genomes"]) == 8

            # Evolving "Butterfly" sends full Gen 1 to ws_bfly, but a tiny stub (< 650 B) to ws_lazy on "SO-102 Arm"
            ws_bfly.send(json.dumps({"type": "cast_vote", "goal": "Butterfly", "generation": 0, "index": 0, "vote": 1}))
            ws_eager.send(json.dumps({"type": "cast_vote", "goal": "Butterfly", "generation": 0, "index": 1, "vote": 1}))

            raw_stub_upd, msg_stub_upd = _recv_raw_type(ws_lazy, "goal_updated", timeout=6.0)
            raw_full_upd, msg_full_upd = _recv_raw_type(ws_bfly, "goal_updated", timeout=6.0)
            assert msg_stub_upd["goal"]["name"] == "Butterfly"
            assert msg_stub_upd["goal"]["generation"] == 1
            assert msg_stub_upd["goal"]["genomesLoaded"] is False
            assert msg_stub_upd["goal"]["genomes"] == []
            assert len(raw_stub_upd) < 650

            assert msg_full_upd["goal"]["name"] == "Butterfly"
            assert msg_full_upd["goal"]["generation"] == 1
            assert msg_full_upd["goal"]["genomesLoaded"] is True
            assert len(msg_full_upd["goal"]["genomes"]) == 30
            assert len(raw_full_upd) > 15_000

            # Compact crowd wire: 201-frame replay subsampled <= 32 keyframes, ACK omits replay/trajSample,
            # GET /crowd/evals omits server-only trajSample while server fetch_room_eval_map retains it.
            md_hash = evo_lazy["goals"][0]["genomes"][0]["hash"]
            dense_poses = [[float(k * 0.01 + j * 0.001) for j in range(105)] for k in range(201)]
            traj_sample = [[float(r * 0.1 + c * 0.01) for c in range(18)] for r in range(16)]
            post_resp = client.post(
                "/api/rooms/lobby/crowd/results/",
                data=json.dumps({
                    "goal": "Microduck",
                    "generation": 0,
                    "results": [{
                        "hash": md_hash,
                        "summary": {"primaryValue": 124.5, "return": 124.5, "meanDistance": 1.42},
                        "trajSample": traj_sample,
                        "replay": {"bodyPoses": dense_poses, "poseStride": 2, "vx_trace": [0.4] * 200},
                    }],
                }),
                content_type="application/json",
            )
            assert post_resp.status_code == 200
            assert len(post_resp.content) < 2500
            ack_item = post_resp.json()["items"][0]
            assert "replay" not in ack_item and "trajSample" not in ack_item

            get_resp = client.get("/api/rooms/lobby/crowd/evals/?goal=Microduck&generation=0")
            assert get_resp.status_code == 200
            fetched_item = next(x for x in get_resp.json()["items"] if x["hash"] == md_hash)
            assert "trajSample" not in fetched_item
            assert len(fetched_item["replay"]["bodyPoses"]) <= 32
            assert fetched_item["replay"]["poseStride"] >= 12
            assert len(fetched_item["replay"]["vx_trace"]) <= 48

            srv_map = fetch_room_eval_map("lobby", "Microduck", 0, trusted_only=False)
            assert len(srv_map[md_hash]["trajSample"]) == 16


@pytest.mark.django_db
def test_public_assets_staging_nginx_template_and_static_base_model(client, settings, tmp_path):
    from direct_evolution_together.environments import SO101_POLICY_SPEC
    from direct_evolution_together.populora_engine import get_or_create_base_model
    from direct_evolution_together.public_assets import stage_public_assets
    import fabfile

    settings.MEDIA_ROOT = tmp_path / "media"

    staged_dir = tmp_path / "staged_assets"
    stats1 = stage_public_assets(staged_dir, precompress=True)
    assert stats1["total"] >= 12
    assert stats1["copied"] == stats1["total"]
    assert stats1["gzipped"] >= 8

    for rel in (
        "sim_worker.js",
        "mujoco_gl.js",
        "scene_cache.js",
        "scenes/microduck/physics.bin",
        "scenes/unitree_go2/visual.bin",
        "scenes/so101/visual.bin",
        "vendor/mujoco/mujoco.wasm",
    ):
        raw_p = staged_dir / rel
        gz_p = staged_dir / f"{rel}.gz"
        assert raw_p.is_file() and gz_p.is_file()
        assert gz_p.stat().st_size < raw_p.stat().st_size
        assert gz_p.stat().st_mtime_ns == raw_p.stat().st_mtime_ns
        assert gzip.decompress(gz_p.read_bytes()) == raw_p.read_bytes()

    stats2 = stage_public_assets(staged_dir, precompress=True)
    assert stats2["copied"] == 0 and stats2["gzipped"] == 0

    # Verify Nginx config renders /assets/ + /media/ static locations with gzip_static & immutable headers
    cfg = fabfile.DeployConfig(
        host="example.com",
        user="ubuntu",
        app_name="det",
        deploy_dir="/var/www/det",
        domain="example.com",
        subpath="/det",
        http_port=8000,
        ws_port=8001,
        uvicorn_workers=2,
        python_version="3.11",
        cuda_visible_devices="",
        nginx_vhost_file="",
        secret_key="s",
        admin_password="a",
        allowed_hosts="example.com",
        csrf_trusted_origins="https://example.com",
        pop_size=30,
        min_votes="20",
        max_votes=20,
        mut_sigma="0.30",
        elite_frac="0.20",
        crossover_rate="0.75",
    )
    nginx_conf = fabfile._render_nginx_conf(cfg)
    assert "location ^~ /det/assets/ {" in nginx_conf
    assert "alias /var/www/det/assets/;" in nginx_conf
    assert "gzip_static on;" in nginx_conf
    assert "application/wasm" in nginx_conf
    assert "application/octet-stream" in nginx_conf
    assert 'Cache-Control "public, max-age=31536000, immutable"' in nginx_conf

    # Verify PopuLoRA base model static JSON + .gz sidecar served via /media/base_models/<hash>.json
    _, b_hash, _ = get_or_create_base_model(SO101_POLICY_SPEC, base_seed=1)
    media_resp = client.get(f"/media/base_models/{b_hash}.json", HTTP_ACCEPT_ENCODING="gzip")
    assert media_resp.status_code == 200
    assert media_resp["Content-Encoding"] == "gzip"
    assert "immutable" in media_resp["Cache-Control"]
    decoded = json.loads(gzip.decompress(media_resp.content).decode("utf-8"))
    assert decoded["baseHash"] == b_hash and "tensors" in decoded


@pytest.mark.django_db(transaction=True)
def test_individual_short_uuid_permalinks_and_expandable_left_panel(client, settings):
    from direct_evolution_together.cppn import SHORT_UUID_RE
    from direct_evolution_together.environments import create_task_population

    # 1. Every CPPN & PopuLoRA individual gets a unique 8-char lowercase hex UUID
    cppn_pop = create_random_population(pop_size=6)
    pl_pop = create_task_population("microduck", 6)
    cppn_uuids = [g["uuid"] for g in cppn_pop]
    pl_uuids = [g["uuid"] for g in pl_pop]
    assert len(set(cppn_uuids)) == 6 and all(SHORT_UUID_RE.match(u) for u in cppn_uuids)
    assert len(set(pl_uuids)) == 6 and all(SHORT_UUID_RE.match(u) for u in pl_uuids)

    restored_cppn = CPPN.from_dict(cppn_pop[0])
    assert restored_cppn.uuid == cppn_pop[0]["uuid"]
    next_pop, _ = evolve_population(cppn_pop, [2, 1, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0], pop_size=6)
    next_uuids = [g["uuid"] for g in next_pop]
    assert len(set(next_uuids)) == 6 and all(SHORT_UUID_RE.match(u) for u in next_uuids)
    assert set(next_uuids).isdisjoint(set(cppn_uuids))

    # 2. Index HTML renders expandable left-hand Individuals panel, hide/show controls, & UUID link badges
    lobby_html = client.get("/").content.decode("utf-8")
    for required_marker in (
        'id="individuals-panel"',
        'id="left-rail-expand-btn"',
        'id="left-rail-collapse-btn"',
        'id="individuals-list"',
        'id="ind-show-all-btn"',
        'id="ind-search-input"',
        'id="fs-uuid-badge"',
        'id="fs-copy-link-btn"',
        'id="det-initial-ind"',
        "toggleHideIndividual",
        "focusOrOpenIndividualByUuid",
    ):
        assert required_marker in lobby_html, f"Missing {required_marker} in index.html"

    # 3. Live WS + HTTP permalinks (/i/<uuid>/, /p/<room>/i/<uuid>/) & JSON API for live and ancestor individuals
    with _ws_server(seed=True, pop_size=6, max_votes=4) as port:
        settings.WS_HOST = "127.0.0.1"
        settings.WS_PORT = port
        seed_from_db(default_pop_size=6, default_max_votes=4)
        mummy_ws.set_round_config(1, 60)

        with (
            connect(f"ws://127.0.0.1:{port}/ws/chat/lobby/?username=alice") as ws_lob,
            connect(f"ws://127.0.0.1:{port}/ws/chat/p-uuidroom/?username=bob&goal=Iris&pop=5") as ws_priv,
        ):
            evo_lob = _recv_until(ws_lob, "evolution_state")
            evo_priv = _recv_until(ws_priv, "evolution_state")

            bfly0 = next(g for g in evo_lob["goals"] if g["name"] == "Butterfly")
            founder_uuid = bfly0["genomes"][0]["uuid"]
            assert SHORT_UUID_RE.match(founder_uuid)

            priv_goal = evo_priv["goals"][0]
            priv_uuid = priv_goal["genomes"][2]["uuid"]
            assert SHORT_UUID_RE.match(priv_uuid)

            # Evolve Butterfly to Gen 1 so founder_uuid becomes an archived lineage ancestor
            ws_lob.send(json.dumps({"type": "cast_vote", "goal": "Butterfly", "generation": 0, "index": 0, "vote": 1}))
            bfly1 = _recv_until(ws_lob, "goal_updated", timeout=4.0)["goal"]
            assert bfly1["generation"] == 1
            live_gen1_uuid = bfly1["genomes"][0]["uuid"]
            assert SHORT_UUID_RE.match(live_gen1_uuid) and live_gen1_uuid != founder_uuid

            # Resolve live Gen 1 individual via global & room-scoped JSON API
            api_live = client.get(f"/api/individuals/{live_gen1_uuid}/").json()
            assert api_live["ok"] is True
            assert api_live["uuid"] == live_gen1_uuid
            assert api_live["room"] == "lobby" and api_live["goal"] == "Butterfly"
            assert api_live["generation"] == 1 and api_live["isLive"] is True
            assert api_live["permalink"] == f"/i/{live_gen1_uuid}/"
            assert len(api_live["genome"]["weights"]) > 100

            # Resolve archived Gen 0 founder from lineage via UUID!
            api_anc = client.get(f"/api/rooms/lobby/individuals/{founder_uuid}/").json()
            assert api_anc["ok"] is True
            assert api_anc["uuid"] == founder_uuid
            assert api_anc["generation"] == 0 and api_anc["isLive"] is False

            # Resolve private room individual + HTML permalinks
            api_priv = client.get(f"/api/rooms/p-uuidroom/individuals/{priv_uuid}/").json()
            assert api_priv["ok"] is True
            assert api_priv["room"] == "p-uuidroom" and api_priv["index"] == 2
            assert api_priv["permalink"] == f"/p/p-uuidroom/i/{priv_uuid}/"

            perm_lob = client.get(f"/i/{live_gen1_uuid}/")
            assert perm_lob.status_code == 200
            perm_lob_html = perm_lob.content.decode("utf-8")
            assert f'id="det-initial-ind" type="application/json">"{live_gen1_uuid}"<' in perm_lob_html
            assert 'id="det-initial-goal" type="application/json">"Butterfly"<' in perm_lob_html

            perm_priv = client.get(f"/p/p-uuidroom/i/{priv_uuid}/")
            assert perm_priv.status_code == 200
            perm_priv_html = perm_priv.content.decode("utf-8")
            assert f'id="det-initial-ind" type="application/json">"{priv_uuid}"<' in perm_priv_html
            assert 'id="det-initial-goal" type="application/json">"Iris"<' in perm_priv_html

            # Unknown UUID returns 404 cleanly
            assert client.get("/api/individuals/deadbeef/").status_code == 404
            assert client.get("/i/deadbeef/").status_code == 404


@pytest.mark.django_db(transaction=True)
def test_so102_arm_environment_hyperparameter_assets_and_wasm_worker(client, settings, tmp_path):
    import math
    from pathlib import Path
    from direct_evolution_together.environments import (
        SO101_POLICY_SPEC,
        create_task_population,
        evaluate_populora_genome,
        get_task_specs_payload,
        unpack_vfs_entries,
    )
    from direct_evolution_together.populora_engine import get_or_create_base_model
    from direct_evolution_together.public_assets import stage_public_assets

    pkg_dir = Path(__file__).resolve().parent.parent / "direct_evolution_together"

    # 1. Task specs expose arm_model environment hyperparameter pointing at SO-102 URDF + STL files
    specs = get_task_specs_payload()
    so_spec = specs["so101"]
    assert so_spec["sceneId"] == "so102"
    assert so_spec["envParams"]["arm_model"] == "so102"
    hspecs = so_spec["hyperparamSpecs"]
    assert len(hspecs) >= 1 and hspecs[0]["key"] == "arm_model" and hspecs[0]["default"] == "so102"
    opt_by_val = {o["value"]: o for o in hspecs[0]["options"]}
    assert set(opt_by_val) == {"so101", "so102"}
    assert opt_by_val["so102"]["sceneId"] == "so102"
    assert opt_by_val["so102"]["urdfAsset"] == "scenes/so102/so_arm_102.urdf"
    so102_var = so_spec["variants"]["so102"]
    assert so102_var["gripperKind"] == "parallel"
    assert len(so102_var["stlAssets"]) == 8
    assert all(p.startswith("scenes/so102/meshes/") and p.endswith(".stl") for p in so102_var["stlAssets"])

    # 2. Verify SO-102 URDF, STL meshes, physics.bin, visual.bin, & public asset staging
    m_resp = client.get("/api/sim/scene/so102/")
    assert m_resp.status_code == 200
    manifest = m_resp.json()
    assert manifest["sceneId"] == "so102"
    assert manifest["nbody"] == 10
    assert manifest["nq"] == 14 and manifest["nv"] == 13 and manifest["nu"] == 6
    assert manifest["visualStats"]["bodyMeshCount"] == 9
    assert {"clamp_left", "clamp_right", "cube"} <= set(manifest["bodyNames"])

    for asset_url, expected_ct in (
        ("/assets/scenes/so102/so_arm_102.urdf", "xml"),
        ("/assets/scenes/so102/meshes/clamp1_1.stl", "stl"),
        ("/assets/scenes/so102/physics.bin", "octet-stream"),
        ("/assets/scenes/so102/visual.bin", "octet-stream"),
    ):
        a_resp = client.get(asset_url)
        assert a_resp.status_code == 200, f"{asset_url} returned {a_resp.status_code}"
        assert expected_ct in a_resp["Content-Type"].lower()

    vfs_so102 = unpack_vfs_entries((pkg_dir / "assets" / "scenes" / "so102" / "physics.bin").read_bytes())
    assert set(vfs_so102.keys()) == {"scene_pick.xml", "so102_arm.xml"}

    staged_dir = tmp_path / "staged_so102"
    stage_public_assets(staged_dir, precompress=True)
    assert (staged_dir / "scenes" / "so102" / "so_arm_102.urdf").is_file()
    assert (staged_dir / "scenes" / "so102" / "so_arm_102.urdf.gz").is_file()
    assert (staged_dir / "scenes" / "so102" / "meshes" / "clamp2_1.stl").is_file()

    # 3. Create room with arm_model="so102", verify WS envParams & bundle export with URDF + license
    room_resp = client.post(
        "/api/rooms/",
        data=json.dumps({
            "name": "SO102 Parallel Jaw Lab",
            "task": "so101",
            "env_params": {"arm_model": "so102"},
            "population_size": 6,
        }),
        content_type="application/json",
    )
    assert room_resp.status_code == 200
    rdata = room_resp.json()
    slug = rdata["slug"]
    assert rdata["env_params"]["arm_model"] == "so102"
    assert rdata["env_params"]["sceneId"] == "so102"

    with _ws_server(seed=True, pop_size=6, max_votes=4) as port:
        settings.WS_HOST = "127.0.0.1"
        settings.WS_PORT = port
        seed_from_db(default_pop_size=6, default_max_votes=4)
        with connect(f"ws://127.0.0.1:{port}/ws/chat/{slug}/?username=operator") as ws:
            evo = _recv_until(ws, "evolution_state")
            g_so102 = evo["goals"][0]
            assert g_so102["task"] == "so101"
            assert g_so102["envParams"]["arm_model"] == "so102"
            assert g_so102["envParams"]["sceneId"] == "so102"
            g0 = g_so102["genomes"][0]

    bun_resp = client.post(
        "/export/bundle/",
        data=json.dumps({
            "goal": "SO102 Parallel Jaw Lab",
            "task": "so101",
            "envParams": {"arm_model": "so102"},
            "generation": 0,
            "index": 0,
            "genome": g0,
        }),
        content_type="application/json",
    )
    assert bun_resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(bun_resp.content)) as zf:
        znames = zf.namelist()
        assert any(n.endswith("/so_arm_102.urdf") for n in znames)
        assert any(n.endswith("/SCENE_LICENSE") for n in znames)
        run_py = [n for n in znames if n.endswith("/run.py")][0]
        run_src = zf.read(run_py).decode("utf-8")
        assert "so102" in run_src
        compile(run_src, "run.py", "exec")

    # 4. Verify BOTH native Python MuJoCo simulator & frontend/sim_worker.js WASM run SO-102 cleanly
    genomes = create_task_population("so101", 4)
    py_eval = evaluate_populora_genome("so101", genomes[0], env_params={"arm_model": "so102"})
    assert py_eval["summary"]["armModel"] == "so102"
    assert math.isfinite(py_eval["summary"]["return"])
    assert len(py_eval["trajSample"]) == 16 and len(py_eval["trajSample"][0]) == 15

    _, _, base_payload = get_or_create_base_model(SO101_POLICY_SPEC, base_seed=1)
    worker_src = (pkg_dir / "frontend" / "sim_worker.js").read_text(encoding="utf-8")
    phys_bin = str(pkg_dir / "assets" / "scenes" / "so102" / "physics.bin")
    wasm_js = (pkg_dir / "assets" / "vendor" / "mujoco" / "mujoco.js").as_uri()
    data_json = tmp_path / "so102_worker_input.json"
    data_json.write_text(
        json.dumps({"baseModel": base_payload, "specs": specs, "genomes": genomes}),
        encoding="utf-8",
    )
    harness_mjs = tmp_path / "so102_worker_harness.mjs"
    harness_mjs.write_text(
        f"""
import fs from "fs";
const physBuf = fs.readFileSync({json.dumps(phys_bin)});
const inp = JSON.parse(fs.readFileSync({json.dumps(str(data_json))}, "utf8"));
const responses = new Map();
let listener = null;
globalThis.self = {{
  location: {{ origin: "http://localhost:8000" }},
  addEventListener: (ev, fn) => {{ if (ev === "message") listener = fn; }},
  postMessage: (msg) => {{
    const r = responses.get(msg.reqId);
    if (r) r(msg);
  }},
}};
{worker_src}
function send(reqId, payload) {{
  return new Promise((resolve) => {{
    responses.set(reqId, resolve);
    listener({{ data: {{ reqId, ...payload }} }});
  }});
}}
const initRes = await send(1, {{
  type: "init",
  taskSpecs: inp.specs,
  mujocoWasmUrl: {json.dumps(wasm_js)},
  warmMujoco: true,
  warmTaskId: "so101",
  envParams: {{ arm_model: "so102" }},
  physicsBuffer: physBuf.buffer.slice(physBuf.byteOffset, physBuf.byteOffset + physBuf.byteLength),
}});
if (!initRes.ok) throw new Error(initRes.error);
const batchRes = await send(2, {{
  type: "evalBatch",
  taskId: "so101",
  envParams: {{ arm_model: "so102" }},
  genomes: inp.genomes,
  seeds: [0, 1],
  baseModel: inp.baseModel,
  physicsBuffer: physBuf.buffer.slice(physBuf.byteOffset, physBuf.byteOffset + physBuf.byteLength),
}});
if (!batchRes.ok) throw new Error(batchRes.error);
const roRes = await send(3, {{
  type: "rollout",
  taskId: "so101",
  envParams: {{ arm_model: "so102" }},
  genome: inp.genomes[0],
  seed: 0,
  maxSteps: 50,
  nudge: "fwd",
  includeActivations: true,
}});
if (!roRes.ok) throw new Error(roRes.error);
console.log(JSON.stringify({{
  engine: batchRes.engine,
  count: batchRes.items.length,
  summaries: batchRes.items.map((x) => x.summary),
  poseRowLen: batchRes.items[0].primaryRollout.bodyPoses[0].length,
  roArmModel: roRes.rollout.armModel,
  roFrames: roRes.rollout.bodyPoses.length,
}}));
""",
        encoding="utf-8",
    )
    wasm_out = json.loads(subprocess.check_output(["node", str(harness_mjs)], text=True))
    assert wasm_out["engine"] == "client-wasm"
    assert wasm_out["count"] == 4
    assert wasm_out["poseRowLen"] == 9 * 7
    assert wasm_out["roArmModel"] == "so102"
    assert wasm_out["roFrames"] >= 2
    assert all(s["armModel"] == "so102" and math.isfinite(s["return"]) for s in wasm_out["summaries"])


@pytest.mark.django_db(transaction=True)
def test_crowdsourced_large_population_sharding_ranking_and_evolution(client, settings):
    import time
    from django.test import Client
    from direct_evolution_together.environments import evaluate_populora_shard, rank_eval_items

    # Create a large MuJoCo population (pop=20 > crowd_shard_threshold=16) using SO-102 arm
    room_resp = client.post(
        "/api/rooms/",
        data=json.dumps({
            "name": "Crowd Sharded Arm Swarm",
            "task": "so101",
            "env_params": {"arm_model": "so102"},
            "population_size": 20,
            "min_votes": 2,
            "max_votes_per_user": 10,
        }),
        content_type="application/json",
    )
    assert room_resp.status_code == 200
    rdata = room_resp.json()
    slug = rdata["slug"]
    assert rdata["population_size"] == 20

    # Standalone crowd_worker.py download endpoint serves valid PEP 723 runner script
    cw_resp = client.get(f"/export/crowd-worker/?room={slug}")
    assert cw_resp.status_code == 200
    cw_src = cw_resp.content.decode("utf-8")
    assert "/crowd/claim/" in cw_src and "/crowd/results/" in cw_src
    compile(cw_src, "crowd_worker.py", "exec")

    with _ws_server(seed=True, pop_size=20, max_votes=10) as port:
        settings.WS_HOST = "127.0.0.1"
        settings.WS_PORT = port
        seed_from_db(default_pop_size=20, default_max_votes=10)
        mummy_ws.set_round_config(2, 80)

        with connect(f"ws://127.0.0.1:{port}/ws/chat/{slug}/?username=lead_researcher&cid=leadcid00001") as ws:
            _recv_until(ws, "identity")
            evo0 = _recv_until(ws, "evolution_state")
            goal0 = evo0["goals"][0]
            gname = goal0["name"]

            # 1. Because pop_size=20 > crowd_shard_threshold=16, WS omits bulky LoRA adapters!
            assert goal0["generation"] == 0
            assert goal0["phase"] == "simulating"
            assert goal0["genomesSharded"] is True
            assert goal0["crowdShardThreshold"] == 16
            assert len(goal0["genomes"]) == 20
            assert all("adapters" not in g and len(g["uuid"]) == 8 and g.get("hash") for g in goal0["genomes"])

            # Single-individual UUID endpoint hydrates full adapter weights on demand
            sample_uuid = goal0["genomes"][0]["uuid"]
            hyd_resp = client.get(f"/api/rooms/{slug}/individuals/{sample_uuid}/").json()
            assert hyd_resp["ok"] is True
            assert isinstance(hyd_resp["genome"].get("adapters"), dict) and len(hyd_resp["genome"]["adapters"]) == 6

            # Voting before crowd shards finish simulating is gated
            ws.send(json.dumps({"type": "cast_vote", "goal": gname, "generation": 0, "index": 0, "vote": 1}))
            assert "Simulations still running" in _recv_until(ws, "vote_throttled")["message"]

            # 2. Three separate clients divvy up the 20 MuJoCo simulations, run local physics, rank, & report back
            workers = [("worker_alpha", Client()), ("worker_beta", Client()), ("worker_gamma", Client())]
            claimed_by_worker: dict[str, list[int]] = {name: [] for name, _ in workers}
            local_shard_rankings: dict[str, list[dict]] = {}

            # Round-robin shard claiming across 3 workers until all 20 individuals are simulated
            for _round in range(4):
                any_claimed = False
                for w_name, w_http in workers:
                    claim_res = w_http.post(
                        f"/api/rooms/{slug}/crowd/claim/",
                        data=json.dumps({"goal": gname, "batchSize": 3, "peerId": w_name}),
                        content_type="application/json",
                    )
                    assert claim_res.status_code == 200
                    claim_data = claim_res.json()
                    assert claim_data["sharded"] is True
                    assert claim_data["envParams"]["arm_model"] == "so102"
                    shard = claim_data["shard"]
                    if not shard:
                        continue
                    any_claimed = True
                    claimed_by_worker[w_name].extend(int(s["index"]) for s in shard)

                    # Run real local MuJoCo SO-102 rollouts on the claimed shard & rank locally
                    evaluated = evaluate_populora_shard(
                        claim_data["task"],
                        shard,
                        seeds=(0,),
                        max_steps=40,
                        env_params=claim_data["envParams"],
                    )
                    eval_batch = evaluated["results"]
                    shard_ranked = evaluated["shardRanking"]
                    assert rank_eval_items(eval_batch) == shard_ranked
                    local_shard_rankings.setdefault(w_name, []).extend(shard_ranked)
                    assert [it["rank"] for it in shard_ranked] == list(range(1, len(shard) + 1))

                    post_res = w_http.post(
                        f"/api/rooms/{slug}/crowd/results/",
                        data=json.dumps({
                            "goal": gname,
                            "generation": 0,
                            "peerId": w_name,
                            "sharded": True,
                            "results": eval_batch,
                            "shardRanking": shard_ranked,
                        }),
                        content_type="application/json",
                    )
                    assert post_res.status_code == 200
                    assert post_res.json()["ok"] is True
                if not any_claimed:
                    break

            # Verify all 3 clients participated and partitioned all 20 individuals with ZERO overlap!
            all_claimed = []
            for w_name, idxs in claimed_by_worker.items():
                assert len(idxs) > 0, f"{w_name} did not receive any shard"
                all_claimed.extend(idxs)
            assert len(all_claimed) == 20
            assert set(all_claimed) == set(range(20))

            # 3. Verify global rankings across all 20 individuals (ranks 1..20, sorted descending by primaryValue)
            evals_resp = client.get(f"/api/rooms/{slug}/crowd/evals/?goal={gname}&generation=0").json()
            assert evals_resp["ok"] is True
            assert evals_resp["sharded"] is True
            assert len(evals_resp["items"]) == 20
            rankings = evals_resp["rankings"]
            assert len(rankings) == 20
            assert [r["rank"] for r in rankings] == list(range(1, 21))
            primary_vals = [float(r["primaryValue"]) for r in rankings]
            assert primary_vals == sorted(primary_vals, reverse=True)

            # 4. Verify WS transitioned to Phase 2 ("voting") once all 20/20 shards completed
            voting_ready = False
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                msg = _decode_ws_message(ws.recv(timeout=1.5))
                if msg.get("type") == "votes_updated" and msg.get("goal") == gname and msg.get("phase") == "voting":
                    assert msg["simCompleted"] == 20 and msg["simTotal"] == 20
                    voting_ready = True
                    break
            assert voting_ready, "Room did not transition to voting after 20/20 crowd shards finished"

            # 5. Researchers vote on the top-2 globally ranked individuals -> evolution advances to Generation 1!
            top1_idx = int(rankings[0]["index"])
            top2_idx = int(rankings[1]["index"])
            ws.send(json.dumps({"type": "cast_vote", "goal": gname, "generation": 0, "index": top1_idx, "vote": 1}))
            ws.send(json.dumps({"type": "cast_vote", "goal": gname, "generation": 0, "index": top2_idx, "vote": 1}))

            gen1_goal = None
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                msg = _decode_ws_message(ws.recv(timeout=3.0))
                if msg.get("type") == "goal_updated" and msg.get("goal", {}).get("name") == gname:
                    gen1_goal = msg["goal"]
                    break
            assert gen1_goal is not None, "Evolution did not advance to Generation 1"
            assert gen1_goal["generation"] == 1
            assert gen1_goal["phase"] == "simulating"
            assert gen1_goal["genomesSharded"] is True
            assert len(gen1_goal["genomes"]) == 20
            assert gen1_goal["lineageTotal"] == 40
            gen0_uuids = {g["uuid"] for g in goal0["genomes"]}
            gen1_uuids = {g["uuid"] for g in gen1_goal["genomes"]}
            assert len(gen1_uuids) == 20
            assert gen1_uuids.isdisjoint(gen0_uuids)


@pytest.mark.django_db(transaction=True)
def test_centralized_hyperparameters_and_population_scaling_constraints(client, monkeypatch):
    from direct_evolution_together.config import EnvPolicyConfig, build_config, reload_config
    from direct_evolution_together.environments import get_task_spec, sanitize_task_env_params

    # 1. Generic centralized env-var hyperparameter overrides (JSON + individual key) with bounds clamping
    monkeypatch.setenv("DET_HP_MICRODUCK_ACTION_SCALE", "2.5")
    monkeypatch.setenv("DET_HYPERPARAMS_UNITREE_GO2", json.dumps({"actionScale": 0.05}))
    monkeypatch.setenv("DET_HP_SO101_ARM_MODEL", "so101")
    cfg = build_config()
    assert cfg.get_env("unitree_go2").env_hyperparams["actionScale"] == 0.05
    assert cfg.get_env("so101").env_hyperparams["arm_model"] == "so101"
    reload_config()
    assert get_task_spec("microduck").env_params["actionScale"] == 0.85  # clamped to [0.20, 0.85]
    assert get_task_spec("unitree_go2").env_params["actionScale"] == 0.20  # clamped to [0.20, 0.75]
    assert get_task_spec("so101").env_params["arm_model"] == "so101"

    monkeypatch.delenv("DET_HP_MICRODUCK_ACTION_SCALE", raising=False)
    monkeypatch.delenv("DET_HYPERPARAMS_UNITREE_GO2", raising=False)
    monkeypatch.delenv("DET_HP_SO101_ARM_MODEL", raising=False)
    reload_config()

    # 2. Room creator hyperparameter overrides are validated and clamped to declared [min, max] bounds
    assert sanitize_task_env_params("microduck", {"actionScale": 99.0})["actionScale"] == 0.85
    assert sanitize_task_env_params("microduck", {"actionScale": -1.0})["actionScale"] == 0.20
    assert sanitize_task_env_params("unitree_go2", {"actionScale": 0.99})["actionScale"] == 0.75
    assert sanitize_task_env_params("so101", {})["arm_model"] == "so102"
    assert sanitize_task_env_params("so101", {"arm_model": "so101"})["arm_model"] == "so101"

    room_resp = client.post(
        "/api/rooms/",
        data=json.dumps({
            "name": "Bounded Duck Swarm",
            "task": "microduck",
            "env_params": {"actionScale": 42.0},
            "population_size": 24,
            "min_votes": 50,
            "max_votes_per_user": 6,
        }),
        content_type="application/json",
    )
    assert room_resp.status_code == 200
    rdata = room_resp.json()
    assert rdata["population_size"] == 24
    assert rdata["env_params"]["actionScale"] == 0.85

    # 3. Population-scaling constraints: unsharded control policy caps effective_max_pop_size before scaling
    unsharded = EnvPolicyConfig(
        task="microduck",
        crowd_contribute=False,
        sim_phase_enabled=False,
        max_pop_size=128,
        crowd_shard_threshold=16,
    )
    assert unsharded.can_shard_population is False
    assert unsharded.effective_max_pop_size == 32
    assert unsharded.constrain_room_scale(80, 40, 10, is_control=True) == {
        "population_size": 32,
        "min_votes": 10,
        "max_votes_per_user": 10,
        "sharded": False,
        "crowd_shard_threshold": 16,
        "crowd_shard_size": 4,
    }

    sharded_spec = get_task_spec("so101")
    assert sharded_spec.max_pop_size == 128
    assert sharded_spec.crowd_shard_threshold == 16
    assert sharded_spec.crowd_shard_size == 4

