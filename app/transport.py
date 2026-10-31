from __future__ import annotations

from typing import Protocol

from app.events import ClientEvent, ServerEvent


class EventTransport(Protocol):
    async def receive(self) -> ClientEvent: ...

    async def send(self, event: ServerEvent) -> None: ...

    async def close(self, code: int) -> None: ...
