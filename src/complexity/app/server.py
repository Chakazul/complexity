"""Interactive UI server: streams frames and measure values over a WebSocket.

Protocol (JSON messages):
  client -> server
    {"type": "start", "id"?: int, "runs": [{"system", "pattern", "size"?, "seed"?}], "measures": [keys],
     "measure_params"?: {key: {param: value}}}
    {"type": "play"} | {"type": "pause"} | {"type": "step"}
    {"type": "speed", "steps_per_tick": int, "fps": int}
    {"type": "map", "measure": key | null}  (heatmap of one measure, docs/measures.md §4)
  server -> client (init and tick echo the start's id, so the client can drop stale ticks)
    {"type": "init", "id": int, "runs": [{"label", "system", "n_states", "rgb", "height", "width", "measures",
                               "skipped", "config": {key: {param: value}}}]}
    {"type": "tick", "id": int, "t": int, "runs": [{"frame": base64 uint8 (h w, or h w 3 if rgb), "values": {}, "ms": {},
                                          "warming_up": [keys],
                                          "map"?: {"h", "w", "data": base64 float32}}]}
    {"type": "error", "message": str}
"""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path

import jax
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from complexity import registry
from complexity.session import Run, RunSpec

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="complexity")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


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
             "needs_discrete": cls.needs_discrete, "peaks_at": cls.peaks_at,
             "params": cls().config(), "choices": cls.choices, "applies_to": cls.applies_to,
             "accepts": [s for s, probe in probes.items() if cls().accepts(probe)]}
            for key, cls in registry.measures().items()
        ],
    }


class Session:
    def __init__(self, runs: list[dict], measures: list[str], measure_params: dict | None = None,
                 id: int = 0):
        self.id = id
        self.map_key: str | None = None
        self.runs = [
            Run(RunSpec(r["system"], r["pattern"], r.get("size"), r.get("seed", 0)), measures,
                measure_params)
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
            frame = base64.b64encode(run.frame_u8().tobytes()).decode()
            item = {"frame": frame, "values": values, "ms": ms, "warming_up": run.warming_up()}
            heat = run.map(self.map_key) if self.map_key else None
            if heat is not None:
                item["map"] = {"h": heat.shape[0], "w": heat.shape[1],
                               "data": base64.b64encode(heat.astype("<f4").tobytes()).decode()}
            out.append(item)
        return {"type": "tick", "id": self.id, "t": self.runs[0].t if self.runs else 0, "runs": out}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
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
                    async with lock:
                        state["session"] = Session(msg["runs"], msg["measures"],
                                                   msg.get("measure_params"), msg.get("id", 0))
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
                    state["steps"] = max(1, int(msg.get("steps_per_tick", 1)))
                    state["fps"] = max(1, int(msg.get("fps", 30)))
            except (KeyError, ValueError) as e:
                await ws.send_json({"type": "error", "message": f"{type(e).__name__}: {e}"})
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
