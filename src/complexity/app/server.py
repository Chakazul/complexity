"""Interactive UI server: streams frames and measure values over a WebSocket.

Protocol (JSON messages):
  client -> server
    {"type": "start", "id"?: int, "runs": [{"system", "pattern", "size"?, "seed"?}], "measures": [keys],
     "measure_params"?: {key: {param: value}}}
    {"type": "play"} | {"type": "pause"} | {"type": "step"}
    {"type": "speed", "steps_per_tick": int, "fps": int}
    {"type": "map", "measure": key | null}  (heatmap of one measure, MEASURES.md §4)
  server -> client (init and tick echo the start's id, so the client can drop stale ticks)
    {"type": "init", "id": int, "runs": [{"label", "system", "n_states", "rgb", "height", "width", "measures",
                               "skipped", "config": {key: {param: value}}}]}
    {"type": "tick", "id": int, "t": int, "runs": [{"frame": base64 zlib uint8 (h w, or h w 3 if rgb), "values": {}, "ms": {},
                                          "warming_up": [keys],
                                          "map"?: {"h", "w", "data": base64 zlib float32}}]}
    {"type": "error", "message": str}

Frames and maps are zlib-compressed (the browser inflates them with DecompressionStream):
uncompressed, eight 128² runs at 30 FPS are ~5 MB/s, too much to share over the internet.

Sharing (README "Sharing the UI"): the server checks what a client asks for (LIMITS), and,
configured by `complexity serve` through environment variables, asks for a password
(COMPLEXITY_PASSWORD: HTTP Basic auth, then a cookie that also admits the WebSocket),
caps measure parameters at PARAM_FACTOR times their defaults (COMPLEXITY_PUBLIC) and the
number of simultaneous sessions (COMPLEXITY_MAX_SESSIONS).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import os
import secrets
import time
import zlib
from pathlib import Path

import jax
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from complexity import registry
from complexity.measures.base import RuleMeasure
from complexity.session import SPECIAL_MAPS, Run, RunSpec

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="complexity")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# What one client may ask for, as in the UI (8 runs, sizes up to 256, speed sliders).
LIMITS = {"runs": 8, "min_size": 8, "max_size": 256, "steps_per_tick": 64, "fps": 60}
PARAM_FACTOR = 16  # public servers: numeric measure parameters at most this times the default
COOKIE = "complexity_auth"
sessions = 0  # open WebSocket connections


def password() -> str:
    return os.environ.get("COMPLEXITY_PASSWORD", "")


def token() -> str:
    """The cookie value that proves the password was given (an HMAC of it)."""
    return hmac.new(password().encode(), b"complexity", hashlib.sha256).hexdigest()


def basic_auth_ok(header: str | None) -> bool:
    """An HTTP Basic `Authorization` header with the password (any user name)."""
    if not header or not header.startswith("Basic "):
        return False
    try:
        _, _, given = base64.b64decode(header[6:]).decode().partition(":")
    except ValueError:
        return False
    return secrets.compare_digest(given.encode(), password().encode())


@app.middleware("http")
async def require_password(request: Request, call_next):
    if not password() or secrets.compare_digest(request.cookies.get(COOKIE, ""), token()):
        return await call_next(request)
    if not basic_auth_ok(request.headers.get("authorization")):
        return Response("password required", status_code=401,
                        headers={"WWW-Authenticate": 'Basic realm="complexity"'})
    response = await call_next(request)
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    response.set_cookie(COOKIE, token(), httponly=True, samesite="strict", secure=secure)
    return response


def check_start(msg: dict) -> None:
    """Raise ValueError if a start message asks for more than LIMITS allows, or, on a public
    server, for measure parameters beyond PARAM_FACTOR times their defaults."""
    runs = msg["runs"]
    if not 1 <= len(runs) <= LIMITS["runs"]:
        raise ValueError(f"1 to {LIMITS['runs']} runs")
    for r in runs:
        size = r.get("size")
        if size is not None and not (len(size) == 2 and all(
                isinstance(n, int) and LIMITS["min_size"] <= n <= LIMITS["max_size"] for n in size)):
            raise ValueError(f"grid sides from {LIMITS['min_size']} to {LIMITS['max_size']}")
    if os.environ.get("COMPLEXITY_PUBLIC"):
        measures = registry.measures()
        for key, params in (msg.get("measure_params") or {}).items():
            for name, value in params.items():
                default = getattr(measures[key], name, None) if key in measures else None
                if (isinstance(default, (int, float)) and not isinstance(default, bool)
                        and isinstance(value, (int, float)) and value > PARAM_FACTOR * default):
                    raise ValueError(f"{key}: {name} at most {PARAM_FACTOR * default} here")


def pack(data: bytes) -> str:
    return base64.b64encode(zlib.compress(data, 1)).decode()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/catalog")
def catalog():
    systems = registry.systems()
    probes = {key: cls() for key, cls in systems.items()}
    device = jax.devices()[0]
    return {
        "device": {"backend": jax.default_backend(), "name": device.device_kind},
        "maps": [{"key": key, "name": name, "unit": unit} for key, (name, unit) in SPECIAL_MAPS.items()],
        "systems": [
            {"key": key, "name": cls.name, "patterns": cls.patterns,
             "n_states": {p: probes[key].n_states(p) for p in cls.patterns},
             "default_size": cls.default_size}
            for key, cls in systems.items()
        ],
        "measures": [
            {"key": key, "name": cls.name, "unit": cls.unit, "description": cls.description,
             "group": cls.group,
             "window": cls.window, "granularity": cls.granularity, "scope": cls.scope,
             "temporal": cls.temporal, "backend": cls.backend,
             "map": cls.map, "map_unit": cls.map_unit or cls.unit,
             "ratio": cls.ratio, "ratio_unit": cls.ratio_unit, "icon": cls.icon,
             "equation": cls.equation, "reference": cls.reference,
             "subject": cls.subject, "estimated_from": cls.estimated_from,
             "per_rule": issubclass(cls, RuleMeasure),
             "needs_discrete": cls.needs_discrete, "peaks_at": cls.peaks_at,
             "params": cls().config(), "choices": cls.choices, "applies_to": cls.applies_to,
             "accepts": [s for s, probe in probes.items() if cls().accepts(probe)]}
            for key, cls in registry.measures().items()
        ],
    }


class Session:
    def __init__(self, runs: list[dict], measures: list[str], measure_params: dict | None = None,
                 id: int = 0, normalized: bool = False):
        self.id = id
        self.map_key: str | None = None
        self.runs = [
            Run(RunSpec(r["system"], r["pattern"], r.get("size"), r.get("seed", 0)), measures,
                measure_params, normalized)
            for r in runs
        ]

    def init_message(self) -> dict:
        return {"type": "init", "id": self.id, "runs": [
            {"label": r.spec.label, "system": r.spec.system, "n_states": r.n_states, "rgb": r.rgb,
             "height": r.size[0], "width": r.size[1],
             "measures": [m.key for m in r.measures], "skipped": r.skipped,
             "config": {m.key: m.config() for m in r.measures}}
            for r in self.runs
        ]}

    def tick(self, steps: int) -> dict:
        out = []
        for run in self.runs:
            run.step(steps)
            values, ms = run.measure()
            frame = pack(run.frame_u8().tobytes())
            item = {"frame": frame, "values": values, "ms": ms, "warming_up": run.warming_up()}
            heat = run.map(self.map_key) if self.map_key else None
            if heat is not None:
                item["map"] = {"h": heat.shape[0], "w": heat.shape[1],
                               "data": pack(heat.astype("<f4").tobytes())}
            out.append(item)
        return {"type": "tick", "id": self.id, "t": self.runs[0].t if self.runs else 0, "runs": out}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global sessions
    if password() and not secrets.compare_digest(ws.cookies.get(COOKIE, ""), token()):
        await ws.close(code=1008)  # policy violation: no password given
        return
    await ws.accept()
    max_sessions = int(os.environ.get("COMPLEXITY_MAX_SESSIONS") or 0)
    if max_sessions and sessions >= max_sessions:
        await ws.send_json({"type": "error", "message": f"server full ({max_sessions} sessions): retrying"})
        await ws.close(code=1013)  # try again later
        return
    sessions += 1
    state = {"session": None, "playing": False, "steps": 1, "fps": 30}
    lock = asyncio.Lock()

    async def tick(steps: int) -> None:
        async with lock:
            msg = await asyncio.to_thread(state["session"].tick, steps)
        await ws.send_json(msg)

    async def ticker() -> None:
        while True:
            if state["session"] and state["playing"]:
                t0 = time.perf_counter()
                await tick(state["steps"])
                await asyncio.sleep(max(0.0, 1 / state["fps"] - (time.perf_counter() - t0)))
            else:
                await asyncio.sleep(0.05)

    task = asyncio.create_task(ticker())
    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            try:
                if kind == "start":
                    check_start(msg)
                    async with lock:
                        state["session"] = Session(msg["runs"], msg["measures"],
                                                   msg.get("measure_params"), msg.get("id", 0),
                                                   bool(msg.get("normalized", False)))
                        state["session"].map_key = state.get("map")
                    await ws.send_json(state["session"].init_message())
                    await tick(0)
                    state["playing"] = True
                elif kind == "play":
                    state["playing"] = True
                elif kind == "pause":
                    state["playing"] = False
                elif kind == "step" and state["session"] and not state["playing"]:
                    await tick(1)
                elif kind == "map":
                    state["map"] = msg.get("measure")
                    if state["session"]:
                        state["session"].map_key = state["map"]
                    if state["session"] and not state["playing"]:
                        await tick(0)  # show it now, also when paused
                elif kind == "speed":
                    state["steps"] = min(LIMITS["steps_per_tick"], max(1, int(msg.get("steps_per_tick", 1))))
                    state["fps"] = min(LIMITS["fps"], max(1, int(msg.get("fps", 30))))
            except (KeyError, ValueError, TypeError) as e:
                await ws.send_json({"type": "error", "message": f"{type(e).__name__}: {e}"})
    except WebSocketDisconnect:
        pass
    finally:
        sessions -= 1
        task.cancel()
