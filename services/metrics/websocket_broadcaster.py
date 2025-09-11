from __future__ import annotations

import asyncio
import json
from typing import Set

from fastapi import WebSocket

class Broadcaster:
    def __init__(self):
        self.clients: Set[WebSocket] = set()

    async def add(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)

    async def remove(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def broadcast(self, data) -> None:
        message = data if isinstance(data, (bytes, str)) else json.dumps(data)
        for ws in list(self.clients):
            try:
                await ws.send_text(message)  # send as text JSON
            except Exception:
                try:
                    await ws.close()
                except Exception:
                    pass
                self.clients.discard(ws)

