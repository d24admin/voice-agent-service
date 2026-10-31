from __future__ import annotations


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
    from app.conversation.outbound_buffer import OutboundBuffer
    from app.conversation.turn_pipeline import ActiveTurnRunner
    from app.conversation.turn_state import TurnOutcome, TurnState
    from app.metrics import TurnOutcome as MetricsTurnOutcome

    assert CallSession.__module__ == "app.conversation.call_session"
    assert OutboundBuffer.__module__ == "app.conversation.outbound_buffer"
    assert ActiveTurnRunner.__module__ == "app.conversation.turn_pipeline"
    assert TurnState.__module__ == "app.conversation.turn_state"
    assert TurnOutcome is MetricsTurnOutcome
