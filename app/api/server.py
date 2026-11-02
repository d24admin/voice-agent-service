from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, WebSocket
from pydantic import BaseModel

from app.clock import Clock, SystemClock
from app.config import Settings
from app.conversation.call_session import CallSession
from app.conversation.session_manager import SessionManager
from app.metrics import MetricsRegistry
from app.providers.llm import LLMProvider, MockLLMProvider
from app.api.websocket_transport import (
    TransportDisconnected,
    WebSocketEventTransport,
)


ProviderFactory = Callable[[str], LLMProvider]


@dataclass(frozen=True, slots=True)
class SessionRuntime:
    settings: Settings
    provider: LLMProvider


RuntimeFactory = Callable[[str, str | None], SessionRuntime]


class HealthResponse(BaseModel):
    status: str
    service: str


def create_app(
    settings: Settings | None = None,
    provider_factory: ProviderFactory | None = None,
    clock: Clock | None = None,
    runtime_factory: RuntimeFactory | None = None,
) -> FastAPI:
    if provider_factory is not None and runtime_factory is not None:
        raise ValueError("provider_factory and runtime_factory are mutually exclusive")
    selected_settings = Settings() if settings is None else settings
    selected_clock = SystemClock() if clock is None else clock
    registry = MetricsRegistry()
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
        if manager.contains(session_id):
            await websocket.close(code=1008)
            return
        if runtime_factory is None:
            session_settings = selected_settings
            provider = selected_provider_factory(session_id)
        else:
            try:
                runtime = runtime_factory(
                    session_id,
                    websocket.query_params.get("scenario_id"),
                )
            except KeyError:
                await websocket.close(code=1008)
                return
            session_settings = runtime.settings
            provider = runtime.provider
        transport = WebSocketEventTransport(
            websocket,
            max_message_bytes=session_settings.max_message_bytes,
        )
        session = CallSession(
            session_id=session_id,
            settings=session_settings,
            transport=transport,
            provider=provider,
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
