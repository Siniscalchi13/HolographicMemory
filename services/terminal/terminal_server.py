from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .shell_manager import ShellSession
from .security import validate_token


logger = logging.getLogger("terminal_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="SmartHaus Terminal Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self):
        self.active: Dict[WebSocket, ShellSession] = {}

    async def connect(self, websocket: WebSocket, token: str | None) -> None:
        await websocket.accept()
        if not validate_token(token):
            await websocket.send_text("[SECURITY] Unauthorized. Set TERMINAL_AUTH_TOKEN or allow dev mode.")
            await websocket.close()
            return

        session = ShellSession()
        session.spawn()
        self.active[websocket] = session

        async def send_data(data: bytes) -> None:
            try:
                await websocket.send_bytes(data)
            except Exception:
                pass

        websocket.state.reader_task = asyncio.create_task(session.read_loop(send_data))

    async def disconnect(self, websocket: WebSocket) -> None:
        session = self.active.pop(websocket, None)
        try:
            if getattr(websocket.state, "reader_task", None):
                websocket.state.reader_task.cancel()
        except Exception:
            pass
        if session:
            session.terminate()


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("token")
    await manager.connect(websocket, token)
    if websocket not in manager.active:
        return
    session = manager.active[websocket]
    try:
        while True:
            msg = await websocket.receive()
            if "text" in msg and msg["text"] is not None:
                txt = msg["text"]
                # Try to parse control JSON for resize
                try:
                    import json as _json
                    payload = _json.loads(txt)
                    if isinstance(payload, dict) and payload.get("type") == "resize":
                        rows = int(payload.get("rows", 32))
                        cols = int(payload.get("cols", 120))
                        session.set_winsize(rows, cols)
                        continue
                except Exception:
                    pass
                await session.write(txt.encode())
            elif "bytes" in msg and msg["bytes"] is not None:
                await session.write(msg["bytes"])
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        await manager.disconnect(websocket)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


def main() -> None:
    port = int(os.environ.get("TERMINAL_SERVICE_PORT", "0") or "0")
    import argparse

    parser = argparse.ArgumentParser(description="Terminal WebSocket Service")
    parser.add_argument("--port", type=int, default=port or 9000)
    args = parser.parse_args()

    uvicorn.run("services.terminal.terminal_server:app", host="0.0.0.0", port=args.port, reload=False)


if __name__ == "__main__":
    main()
