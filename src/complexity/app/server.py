"""Interactive UI server: streams frames and measure values over a WebSocket.

Protocol (JSON messages):
  client -> server
    {"type": "start", "runs": [{"system", "pattern", "size"?, "seed"?}], "measures": [keys]}
    {"type": "play"} | {"type": "pause"} | {"type": "step"}
    {"type": "speed", "steps_per_tick": int, "fps": int}
  server -> client
    {"type": "init", "runs": [{"label", "system", "height", "width", "measures", "skipped"}]}
    {"type": "tick", "t": int, "runs": [{"frame": base64 uint8, "values": {}, "ms": {}}]}
    {"type": "error", "message": str}
"""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path

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
    return {
        "systems": [
            {"key": key, "name": cls.name, "state_type": cls.state_type,
             "patterns": cls.patterns, "default_size": cls.default_size}
            for key, cls in systems.items()
        ],
        "measures": [
            {"key": key, "name": cls.name, "unit": cls.unit, "description": cls.description,
             "window": cls.window, "granularity": cls.granularity, "scope": cls.scope,
             "temporal": cls.temporal,
             "accepts": [s for s, probe in probes.items() if cls().accepts(probe)]}
            for key, cls in registry.measures().items()
        ],
    }


class Session:
    def __init__(self, runs: list[dict], measures: list[str]):
        self.runs = [
            Run(RunSpec(r["system"], r["pattern"], r.get("size"), r.get("seed", 0)), measures)
            for r in runs
        ]

    def init_message(self) -> dict:
        return {"type": "init", "runs": [
            {"label": r.spec.label, "system": r.spec.system,
             "height": r.system.size[0], "width": r.system.size[1],
             "measures": [m.key for m in r.measures], "skipped": r.skipped}
            for r in self.runs
        ]}

    def tick(self, steps: int) -> dict:
        out = []
        for run in self.runs:
            run.step(steps)
            values, ms = run.measure()
            frame = base64.b64encode(run.frame_u8().tobytes()).decode()
            out.append({"frame": frame, "values": values, "ms": ms})
        return {"type": "tick", "t": self.runs[0].t if self.runs else 0, "runs": out}


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
                        state["session"] = Session(msg["runs"], msg["measures"])
                    await ws.send_json(state["session"].init_message())
                    await tick(0)
                    state["playing"] = True
                elif kind == "play":
                    state["playing"] = True
                elif kind == "pause":
                    state["playing"] = False
                elif kind == "step" and state["session"] and not state["playing"]:
                    await tick(1)
                elif kind == "speed":
                    state["steps"] = max(1, int(msg.get("steps_per_tick", 1)))
                    state["fps"] = max(1, int(msg.get("fps", 30)))
            except (KeyError, ValueError) as e:
                await ws.send_json({"type": "error", "message": f"{type(e).__name__}: {e}"})
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
