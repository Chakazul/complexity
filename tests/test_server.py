"""The UI server's protocol and its settings for sharing: compressed frames, limits,
password, session cap (server.py)."""

import base64
import zlib

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from complexity.app import server

RUN = {"system": "life", "pattern": "glider", "size": [32, 32]}


@pytest.fixture
def client(monkeypatch):
    for var in ("COMPLEXITY_PASSWORD", "COMPLEXITY_PUBLIC", "COMPLEXITY_MAX_SESSIONS"):
        monkeypatch.delenv(var, raising=False)
    return TestClient(server.app)


def start(ws, runs, **extra):
    ws.send_json({"type": "start", "id": 1, "runs": runs, "measures": ["lzma"], **extra})
    return ws.receive_json()


def test_ticks_carry_compressed_frames(client):
    with client.websocket_connect("/ws") as ws:
        assert start(ws, [RUN])["type"] == "init"
        tick = ws.receive_json()
        frame = zlib.decompress(base64.b64decode(tick["runs"][0]["frame"]))
        assert tick["type"] == "tick" and len(frame) == 32 * 32


def test_limits_on_runs_sizes_and_speed(client):
    with client.websocket_connect("/ws") as ws:
        assert "runs" in start(ws, [RUN] * 9)["message"]
        assert "grid sides" in start(ws, [{**RUN, "size": [1024, 1024]}])["message"]
        assert "grid sides" in start(ws, [{**RUN, "size": [32]}])["message"]
    with pytest.raises(ValueError):
        server.check_start({"runs": []})


def test_public_servers_cap_measure_parameters(client, monkeypatch):
    msg = {"runs": [RUN], "measure_params": {"functional_information": {"samples": 10**6}}}
    server.check_start(msg)  # a private server allows it
    monkeypatch.setenv("COMPLEXITY_PUBLIC", "1")
    with pytest.raises(ValueError, match="samples"):
        server.check_start(msg)
    server.check_start({"runs": [RUN], "measure_params": {"functional_information": {"samples": 1024}}})


def test_password_admits_the_page_and_then_the_websocket(client, monkeypatch):
    monkeypatch.setenv("COMPLEXITY_PASSWORD", "s3cret")
    assert client.get("/").status_code == 401
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws") as ws:
        ws.receive_json()
    assert client.get("/", auth=("anyone", "wrong")).status_code == 401
    assert client.get("/", auth=("anyone", "s3cret")).status_code == 200  # sets the cookie
    assert client.get("/api/catalog").status_code == 200
    with client.websocket_connect("/ws") as ws:
        assert start(ws, [RUN])["type"] == "init"


def test_session_cap(client, monkeypatch):
    monkeypatch.setenv("COMPLEXITY_MAX_SESSIONS", "1")
    with client.websocket_connect("/ws") as first:
        assert start(first, [RUN])["type"] == "init"
        with client.websocket_connect("/ws") as second:
            assert "server full" in second.receive_json()["message"]
    with client.websocket_connect("/ws") as again:  # the first one has left
        assert start(again, [RUN])["type"] == "init"
