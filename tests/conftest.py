from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from app.api.events import ClientEvent, ServerEvent


class FakeTransport:
    def __init__(self) -> None:
        self.inbound: asyncio.Queue[ClientEvent] = asyncio.Queue()
        self.sent: list[ServerEvent] = []
        self.attempted: asyncio.Queue[ServerEvent] = asyncio.Queue()
        self.close_code: int | None = None
        self._blocked_type: str | None = None
        self._send_gate = asyncio.Event()
        self._send_gate.set()

    async def receive(self) -> ClientEvent:
        return await self.inbound.get()

    async def send(self, event: ServerEvent) -> None:
        await self.attempted.put(event)
        if event.type == self._blocked_type:
            await self._send_gate.wait()
        self.sent.append(event)

    async def close(self, code: int) -> None:
        self.close_code = code

    def block_on(self, event_type: str) -> None:
        self._blocked_type = event_type
        self._send_gate.clear()

    def release_sends(self) -> None:
        self._send_gate.set()

    async def wait_for_sent(self, event_type: str, timeout: float = 1) -> ServerEvent:
        async def find() -> ServerEvent:
            while True:
                for event in self.sent:
                    if event.type == event_type:
                        return event
                await asyncio.sleep(0)

        return await asyncio.wait_for(find(), timeout=timeout)

    async def wait_for_attempt(self, event_type: str, timeout: float = 1) -> ServerEvent:
        async def find() -> ServerEvent:
            while True:
                event = await self.attempted.get()
                if event.type == event_type:
                    return event

        return await asyncio.wait_for(find(), timeout=timeout)


def event_fields(event_id: str, session_id: str = "session-1") -> dict[str, object]:
    return {
        "event_id": event_id,
        "session_id": session_id,
        "timestamp": datetime.now(UTC),
    }


def live_session_tasks(session_id: str) -> list[asyncio.Task[object]]:
    prefix = f"voice-session-{session_id}:"
    current = asyncio.current_task()
    return [
        task
        for task in asyncio.all_tasks()
        if task is not current and not task.done() and task.get_name().startswith(prefix)
    ]
