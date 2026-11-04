import asyncio
import json
from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from app.providers.llm import (
    ChatMessage,
    GeminiLLMProvider,
    LLMProviderError,
    LLMProviderUnavailable,
)


MESSAGES = (
    ChatMessage(role="system", content="Be concise."),
    ChatMessage(role="user", content="Hello"),
    ChatMessage(role="assistant", content="Hi there."),
    ChatMessage(role="user", content="Can you help?"),
)


def sse(*payloads: object) -> bytes:
    return b"".join(
        f"data: {json.dumps(payload)}\n\n".encode() for payload in payloads
    )


def gemini_chunk(text: str, *, done: bool = False) -> dict[str, object]:
    candidate: dict[str, object] = {
        "content": {
            "role": "model",
            "parts": [{"text": text}],
        }
    }
    if done:
        candidate["finishReason"] = "STOP"
    return {
        "candidates": [candidate],
        "usageMetadata": {
            "promptTokenCount": 3,
            "candidatesTokenCount": 2,
            "totalTokenCount": 5,
        },
    }


def provider_with(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    max_retries: int = 1,
    retry_delay_s: float = 0.25,
    sleep: Callable[[float], object] | None = None,
) -> GeminiLLMProvider:
    async def default_sleep(_: float) -> None:
        return None

    return GeminiLLMProvider(
        api_key="test-secret",
        model="gemini-test",
        base_url="https://example.test/v1beta",
        max_retries=max_retries,
        retry_delay_s=retry_delay_s,
        sleep=default_sleep if sleep is None else sleep,  # type: ignore[arg-type]
        transport=httpx.MockTransport(handler),
    )


async def collect(provider: GeminiLLMProvider) -> list[str]:
    return [token async for token in provider.stream(MESSAGES, 7, "session-1")]


@pytest.mark.asyncio
async def test_gemini_maps_chat_request_and_streams_complete_words() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse(
                gemini_chunk("Hel"),
                gemini_chunk("lo world. ", done=True),
            ),
        )

    provider = provider_with(handler)
    try:
        assert await collect(provider) == ["Hello", "world."]
    finally:
        await provider.aclose()

    assert len(requests) == 1
    request = requests[0]
    assert request.url == (
        "https://example.test/v1beta/models/"
        "gemini-test:streamGenerateContent?alt=sse"
    )
    assert request.headers["x-goog-api-key"] == "test-secret"
    assert request.headers["x-goog-api-client"] == "voice-agent-service/0.1.0"
    assert json.loads(request.content) == {
        "systemInstruction": {"parts": [{"text": "Be concise."}]},
        "contents": [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi there."}]},
            {"role": "user", "parts": [{"text": "Can you help?"}]},
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [408, 429, 500, 503])
async def test_gemini_retries_transient_http_failure_before_output(
    status: int,
) -> None:
    attempts = 0
    delays: list[float] = []

    async def record_sleep(seconds: float) -> None:
        delays.append(seconds)

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(status, content=b"provider detail")
        return httpx.Response(
            200, content=sse(gemini_chunk("Recovered. ", done=True))
        )

    provider = provider_with(handler, sleep=record_sleep)
    try:
        assert await collect(provider) == ["Recovered."]
    finally:
        await provider.aclose()

    assert attempts == 2
    assert delays == [0.25]


@pytest.mark.asyncio
async def test_gemini_retries_connection_failure_before_output() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("private connection detail", request=request)
        return httpx.Response(
            200, content=sse(gemini_chunk("Ready. ", done=True))
        )

    provider = provider_with(handler)
    try:
        assert await collect(provider) == ["Ready."]
    finally:
        await provider.aclose()

    assert attempts == 2


@pytest.mark.asyncio
async def test_gemini_exhaustion_raises_sanitized_unavailable_error() -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, content=b"secret upstream response")

    provider = provider_with(handler)
    try:
        with pytest.raises(LLMProviderUnavailable) as raised:
            await collect(provider)
    finally:
        await provider.aclose()

    assert attempts == 2
    assert str(raised.value) == "LLM provider unavailable"
    assert "secret" not in repr(raised.value)
    assert "test-secret" not in repr(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_gemini_does_not_retry_permanent_http_failure(status: int) -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(status, content=b"private provider response")

    provider = provider_with(handler)
    try:
        with pytest.raises(LLMProviderUnavailable):
            await collect(provider)
    finally:
        await provider.aclose()

    assert attempts == 1


@pytest.mark.asyncio
async def test_gemini_retries_unusable_stream_before_output() -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(200, content=b"data: not-json\n\n")
        return httpx.Response(
            200, content=sse(gemini_chunk("Valid. ", done=True))
        )

    provider = provider_with(handler)
    try:
        assert await collect(provider) == ["Valid."]
    finally:
        await provider.aclose()

    assert attempts == 2


@pytest.mark.asyncio
async def test_gemini_retries_finished_stream_without_text() -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                200, content=sse(gemini_chunk("", done=True))
            )
        return httpx.Response(
            200, content=sse(gemini_chunk("Available. ", done=True))
        )

    provider = provider_with(handler)
    try:
        assert await collect(provider) == ["Available."]
    finally:
        await provider.aclose()

    assert attempts == 2


@pytest.mark.asyncio
async def test_gemini_does_not_retry_failure_after_first_token() -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            content=sse(gemini_chunk("Partial ")) + b"data: not-json\n\n",
        )

    provider = provider_with(handler)
    stream = provider.stream(MESSAGES, 7, "session-1")
    try:
        assert await anext(stream) == "Partial"
        with pytest.raises(LLMProviderError):
            await anext(stream)
    finally:
        await stream.aclose()
        await provider.aclose()

    assert attempts == 1


class BlockingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.closed.set()
        if False:
            yield b""

    async def aclose(self) -> None:
        self.closed.set()


class YieldingStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.closed = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.content
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed.set()


@pytest.mark.asyncio
async def test_gemini_closing_public_stream_awaits_response_close() -> None:
    body = YieldingStream(sse(gemini_chunk("Partial ")))

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=body)

    provider = provider_with(handler)
    stream = provider.stream(MESSAGES, 7, "session-1")
    try:
        assert await anext(stream) == "Partial"
        await stream.aclose()
        assert body.closed.is_set()
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_gemini_cancellation_closes_stream_without_retry() -> None:
    attempts = 0
    body = BlockingStream()

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, stream=body)

    provider = provider_with(handler)
    task = asyncio.create_task(anext(provider.stream(MESSAGES, 7, "session-1")))
    await asyncio.wait_for(body.started.wait(), timeout=1)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(body.closed.wait(), timeout=1)
    finally:
        await provider.aclose()

    assert attempts == 1
