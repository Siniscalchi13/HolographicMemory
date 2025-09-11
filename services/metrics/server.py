from __future__ import annotations

import asyncio
import os
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .metrics_collector import collect_metrics
from .websocket_broadcaster import Broadcaster
from .data_storage import RingBuffer


app = FastAPI(title="Metrics Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

broadcaster = Broadcaster()
history = RingBuffer(maxlen=2000)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/history")
async def get_history():
    return history.all()


@app.websocket("/ws")
async def ws_metrics(ws: WebSocket):
    await broadcaster.add(ws)
    try:
        # Send initial snapshot
        await broadcaster.broadcast({"type": "hello", "history": history.all()[-20:]})
        while True:
            await asyncio.sleep(1.0)
    except Exception:
        pass
    finally:
        await broadcaster.remove(ws)


async def producer_loop():
    while True:
        data = collect_metrics()
        history.append(data)
        await broadcaster.broadcast({"type": "metrics", **data})
        await asyncio.sleep(1.0)


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(producer_loop())


def main() -> None:
    port = int(os.environ.get("METRICS_SERVICE_PORT", "0") or "0")
    import argparse

    parser = argparse.ArgumentParser(description="Metrics Service")
    parser.add_argument("--port", type=int, default=port or 9100)
    args = parser.parse_args()
    uvicorn.run("services.metrics.server:app", host="0.0.0.0", port=args.port, reload=False)


if __name__ == "__main__":
    main()

