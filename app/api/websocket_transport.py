from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError
from starlette.websockets import WebSocket, WebSocketDisconnect

from app.api.events import ClientEvent, ServerEvent, parse_client_event


class EventTransport(Protocol):
    async def receive(self) -> ClientEvent: ...

    async def send(self, event: ServerEvent) -> None: ...

    async def close(self, code: int) -> None: ...


@dataclass(frozen=True, slots=True)
class ProtocolError(Exception):
    code: str
    safe_message: str
    fatal: bool
    close_code: int | None = None
    emit_error: bool = True


class TransportDisconnected(Exception):
    pass


class WebSocketEventTransport:
    def __init__(self, websocket: WebSocket, max_message_bytes: int) -> None:
        self._websocket = websocket
        self._max_message_bytes = max_message_bytes
        self._closed = False

    async def receive(self) -> ClientEvent:
        try:
            raw = await self._websocket.receive_text()
        except WebSocketDisconnect as exc:
            raise TransportDisconnected from exc
        if len(raw.encode("utf-8")) > self._max_message_bytes:
            raise ProtocolError(
                code="message_too_large",
                safe_message="message exceeds configured size limit",
                fatal=True,
                close_code=1009,
                emit_error=False,
            )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProtocolError(
                code="invalid_json",
                safe_message="message must be valid JSON",
                fatal=False,
            ) from exc
        try:
            return parse_client_event(payload)
        except ValidationError as exc:
            raise ProtocolError(
                code="invalid_event",
                safe_message="message does not match a supported event schema",
                fatal=False,
            ) from exc

    async def send(self, event: ServerEvent) -> None:
        try:
            await self._websocket.send_json(event.model_dump(mode="json"))
        except WebSocketDisconnect as exc:
            raise TransportDisconnected from exc

    async def close(self, code: int) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._websocket.close(code=code)
        except (RuntimeError, WebSocketDisconnect):
            pass
