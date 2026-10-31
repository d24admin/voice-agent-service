from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, WebSocket
from pydantic import BaseModel

from app.clock import Clock, SystemClock
from app.config import Settings
from app.metrics import MetricsRegistry
from app.providers.llm import LLMProvider, MockLLMProvider
from app.session import CallSession
from app.transport import TransportDisconnected, WebSocketEventTransport


ProviderFactory = Callable[[str], LLMProvider]


class HealthResponse(BaseModel):
    status: str
    service: str


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, CallSession] = {}

    def register(self, session: CallSession) -> None:
        if session.session_id in self._sessions:
            raise ValueError(f"session already registered: {session.session_id}")
        self._sessions[session.session_id] = session

    def unregister(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def close_all(self, timeout_s: float) -> None:
        sessions = tuple(self._sessions.values())
        if not sessions:
            return
        try:
            async with asyncio.timeout(timeout_s):
                await asyncio.gather(
                    *(session.request_shutdown() for session in sessions)
                )
                await asyncio.gather(*(session.wait_closed() for session in sessions))
        except TimeoutError:
            for session in sessions:
                session.force_cancel()
        finally:
            self._sessions.clear()


def create_app(
    settings: Settings | None = None,
    provider_factory: ProviderFactory | None = None,
    clock: Clock | None = None,
) -> FastAPI:
    selected_settings = Settings() if settings is None else settings
    selected_clock = SystemClock() if clock is None else clock
    registry = MetricsRegistry(
        session_capacity=selected_settings.metrics_session_capacity,
        turns_per_session=selected_settings.metrics_turns_per_session,
    )
    manager = SessionManager()

    if provider_factory is None:
        def default_provider(_session_id: str) -> LLMProvider:
            return MockLLMProvider(
                responses=(
                    "Thanks for calling. I can help with that request.",
                    "I heard your update. Let us continue from there.",
                ),
                token_delay_s=selected_settings.llm_token_delay_ms / 1_000,
            )

        selected_provider_factory = default_provider
    else:
        selected_provider_factory = provider_factory

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await manager.close_all(selected_settings.shutdown_grace_ms / 1_000)

    app = FastAPI(title="voice-agent-service", lifespan=lifespan)
    app.state.settings = selected_settings
    app.state.metrics = registry
    app.state.sessions = manager

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", service="voice-agent-service")

    @app.get("/metrics")
    async def metrics(
        session_id: Annotated[
            str | None,
            Query(min_length=1, max_length=128),
        ] = None,
    ) -> dict[str, object]:
        summary = registry.summary(session_id)
        if summary is None:
            raise HTTPException(status_code=404, detail="session not found")
        return summary

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        if not 1 <= len(session_id) <= 128:
            await websocket.close(code=1008)
            return
        transport = WebSocketEventTransport(
            websocket,
            max_message_bytes=selected_settings.max_message_bytes,
        )
        session = CallSession(
            session_id=session_id,
            settings=selected_settings,
            transport=transport,
            provider=selected_provider_factory(session_id),
            registry=registry,
            clock=selected_clock,
        )
        try:
            manager.register(session)
        except ValueError:
            await transport.close(1008)
            return
        try:
            await session.run()
        except (ConnectionError, RuntimeError, TransportDisconnected):
            pass
        finally:
            manager.unregister(session_id)

    return app


app = create_app()
