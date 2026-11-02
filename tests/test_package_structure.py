from __future__ import annotations

import importlib.util

from fastapi import FastAPI


def test_provider_modules_are_grouped() -> None:
    from app.providers.llm import ChatMessage, LLMProvider, MockLLMProvider
    from app.providers.stt import FinalResult, PartialResult, SimulatedSTT
    from app.providers.tts import SentenceBuffer

    assert ChatMessage.__module__ == "app.providers.llm"
    assert LLMProvider.__module__ == "app.providers.llm"
    assert MockLLMProvider.__module__ == "app.providers.llm"
    assert FinalResult.__module__ == "app.providers.stt"
    assert PartialResult.__module__ == "app.providers.stt"
    assert SimulatedSTT.__module__ == "app.providers.stt"
    assert SentenceBuffer.__module__ == "app.providers.tts"


def test_conversation_modules_are_grouped() -> None:
    from app.conversation.call_session import CallSession
    from app.conversation.turn_pipeline import ActiveTurnRunner
    from app.conversation.turn_state import TurnOutcome, TurnState
    from app.metrics import TurnOutcome as MetricsTurnOutcome

    assert CallSession.__module__ == "app.conversation.call_session"
    assert ActiveTurnRunner.__module__ == "app.conversation.turn_pipeline"
    assert TurnState.__module__ == "app.conversation.turn_state"
    assert TurnOutcome is MetricsTurnOutcome


def test_api_modules_are_grouped_and_flat_modules_removed() -> None:
    from app.api.events import SessionStart, parse_client_event
    from app.api.server import create_app
    from app.api.websocket_transport import WebSocketEventTransport
    from app.conversation.session_manager import SessionManager

    assert isinstance(create_app(), FastAPI)
    assert SessionStart.__module__ == "app.api.events"
    assert parse_client_event.__module__ == "app.api.events"
    assert WebSocketEventTransport.__module__ == "app.api.websocket_transport"
    assert SessionManager.__module__ == "app.conversation.session_manager"

    for removed_module in (
        "app.events",
        "app.main",
        "app.outbound",
        "app.conversation.outbound_buffer",
        "app.session",
        "app.stt",
        "app.transport",
        "app.tts",
        "app.turn",
    ):
        assert importlib.util.find_spec(removed_module) is None
