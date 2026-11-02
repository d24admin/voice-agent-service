import asyncio

import pytest

from app.providers.llm import ChatMessage, MockLLMProvider
from app.providers.tts import SentenceBuffer


MESSAGES = [ChatMessage(role="user", content="hello")]


async def collect(provider: MockLLMProvider, turn_id: int) -> list[str]:
    return [token async for token in provider.stream(MESSAGES, turn_id)]


@pytest.mark.asyncio
async def test_mock_provider_streams_words_with_punctuation_and_delay() -> None:
    delays: list[float] = []

    async def record_delay(seconds: float) -> None:
        delays.append(seconds)

    provider = MockLLMProvider(
        ["Hello there. How are you?"], token_delay_s=0.04, sleep=record_delay
    )

    assert await collect(provider, 1) == ["Hello", "there.", "How", "are", "you?"]
    assert delays == [0.04, 0.04, 0.04, 0.04, 0.04]


@pytest.mark.asyncio
async def test_interrupted_stream_consumes_its_response_slot() -> None:
    async def no_delay(_: float) -> None:
        return None

    provider = MockLLMProvider(
        ["first response", "second response"], token_delay_s=0.01, sleep=no_delay
    )
    stream = provider.stream(MESSAGES, 1)

    assert await anext(stream) == "first"
    await stream.aclose()

    assert await collect(provider, 2) == ["second", "response"]
    assert await collect(provider, 3) == ["second", "response"]


@pytest.mark.asyncio
async def test_mock_provider_propagates_cancellation() -> None:
    async def cancelled(_: float) -> None:
        raise asyncio.CancelledError

    provider = MockLLMProvider(["never emitted"], token_delay_s=0.01, sleep=cancelled)

    with pytest.raises(asyncio.CancelledError):
        await anext(provider.stream(MESSAGES, 1))


def test_sentence_buffer_handles_boundaries_across_tokens() -> None:
    buffer = SentenceBuffer()

    assert buffer.push("Hello") == []
    assert buffer.push("there.") == ["Hello there."]
    assert buffer.push("Are") == []
    assert buffer.push("you") == []
    assert buffer.push("?") == ["Are you?"]


def test_sentence_buffer_emits_multiple_sentences_from_one_token() -> None:
    buffer = SentenceBuffer()

    assert buffer.push("One. Two? Three!") == ["One.", "Two?", "Three!"]
    assert buffer.flush() is None


def test_sentence_buffer_normalizes_whitespace_without_joining_words() -> None:
    buffer = SentenceBuffer()

    assert buffer.push("  hello   world  ") == []
    assert buffer.push("again.") == ["hello world again."]


def test_sentence_buffer_flushes_trailing_fragment_once() -> None:
    buffer = SentenceBuffer()
    buffer.push("unfinished")
    buffer.push("answer")

    assert buffer.flush() == "unfinished answer"
    assert buffer.flush() is None
