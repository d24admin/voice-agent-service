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
