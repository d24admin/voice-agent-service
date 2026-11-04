from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias
from urllib.parse import quote

import httpx

from app import __version__


AsyncSleep: TypeAlias = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


class LLMProvider(Protocol):
    def stream(
        self,
        messages: Sequence[ChatMessage],
        turn_id: int,
        session_id: str | None = None,
    ) -> AsyncIterator[str]: ...


class LLMProviderError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("LLM provider error")


class LLMProviderUnavailable(LLMProviderError):
    def __init__(self) -> None:
        RuntimeError.__init__(self, "LLM provider unavailable")


class _RetryableGeminiError(Exception):
    pass


class _PermanentGeminiError(Exception):
    pass


class _StreamingWords:
    def __init__(self) -> None:
        self._pending = ""

    def push(self, text: str) -> list[str]:
        self._pending += text
        tokens: list[str] = []
        while match := re.search(r"\s+", self._pending):
            token = self._pending[: match.start()]
            self._pending = self._pending[match.end() :]
            if token:
                tokens.append(token)
        return tokens

    def flush(self) -> str | None:
        token = self._pending.strip()
        self._pending = ""
        return token or None


class GeminiLLMProvider:
    """Stream Gemini chat text with safe pre-output retries."""

    _RETRYABLE_STATUSES = {408, 429}
    _SUCCESS_FINISH_REASONS = {"STOP", "MAX_TOKENS"}

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        max_retries: int = 1,
        retry_delay_s: float = 0.25,
        sleep: AsyncSleep = asyncio.sleep,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Gemini API key cannot be blank")
        if not model.strip():
            raise ValueError("Gemini model cannot be blank")
        if max_retries < 0:
            raise ValueError("max retries cannot be negative")
        if retry_delay_s < 0:
            raise ValueError("retry delay cannot be negative")
        self._api_key = api_key
        self._model = model.strip()
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._retry_delay_s = retry_delay_s
        self._sleep = sleep
        self._client = httpx.AsyncClient(timeout=None, transport=transport)

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        turn_id: int,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        del turn_id, session_id
        attempts = 0
        while True:
            emitted = False
            try:
                async with aclosing(self._stream_once(messages)) as stream:
                    async for token in stream:
                        emitted = True
                        yield token
                return
            except asyncio.CancelledError:
                raise
            except _PermanentGeminiError:
                if emitted:
                    raise LLMProviderError from None
                raise LLMProviderUnavailable from None
            except _RetryableGeminiError:
                if emitted:
                    raise LLMProviderError from None
                if attempts >= self._max_retries:
                    raise LLMProviderUnavailable from None
                attempts += 1
                await self._sleep(self._retry_delay_s)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _stream_once(
        self, messages: Sequence[ChatMessage]
    ) -> AsyncIterator[str]:
        try:
            async with self._client.stream(
                "POST",
                self._stream_url(),
                headers={
                    "accept": "text/event-stream",
                    "content-type": "application/json",
                    "x-goog-api-key": self._api_key,
                    "x-goog-api-client": f"voice-agent-service/{__version__}",
                },
                json=self._request_body(messages),
            ) as response:
                if response.status_code < 200 or response.status_code >= 300:
                    if self._is_retryable_status(response.status_code):
                        raise _RetryableGeminiError
                    raise _PermanentGeminiError

                words = _StreamingWords()
                finished = False
                saw_text = False
                async for line in response.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        raise _RetryableGeminiError
                    try:
                        payload = json.loads(line.removeprefix("data:").strip())
                        texts, finish_reason = self._parse_payload(payload)
                    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                        raise _RetryableGeminiError from None
                    for text in texts:
                        saw_text = saw_text or bool(text.strip())
                        for token in words.push(text):
                            yield token
                    if finish_reason is not None:
                        if finish_reason not in self._SUCCESS_FINISH_REASONS:
                            raise _PermanentGeminiError
                        finished = True

                if not finished:
                    raise _RetryableGeminiError
                if not saw_text:
                    raise _RetryableGeminiError
                trailing = words.flush()
                if trailing is not None:
                    yield trailing
        except asyncio.CancelledError:
            raise
        except (_RetryableGeminiError, _PermanentGeminiError):
            raise
        except httpx.HTTPError:
            raise _RetryableGeminiError from None

    def _stream_url(self) -> str:
        model = quote(self._model, safe="")
        return (
            f"{self._base_url}/models/{model}:streamGenerateContent?alt=sse"
        )

    @staticmethod
    def _request_body(messages: Sequence[ChatMessage]) -> dict[str, object]:
        system_text = "\n\n".join(
            message.content for message in messages if message.role == "system"
        )
        contents = [
            {
                "role": "model" if message.role == "assistant" else "user",
                "parts": [{"text": message.content}],
            }
            for message in messages
            if message.role != "system"
        ]
        body: dict[str, object] = {"contents": contents}
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        return body

    @staticmethod
    def _parse_payload(payload: object) -> tuple[list[str], str | None]:
        if not isinstance(payload, dict):
            raise TypeError
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            prompt_feedback = payload.get("promptFeedback")
            if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
                raise _PermanentGeminiError
            return [], None
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise TypeError
        finish_reason = candidate.get("finishReason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise TypeError
        content = candidate.get("content")
        if content is None:
            return [], finish_reason
        if not isinstance(content, dict):
            raise TypeError
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            raise TypeError
        texts: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                raise TypeError
            text = part.get("text")
            if text is not None:
                if not isinstance(text, str):
                    raise TypeError
                texts.append(text)
        return texts, finish_reason

    @classmethod
    def _is_retryable_status(cls, status_code: int) -> bool:
        return status_code in cls._RETRYABLE_STATUSES or status_code >= 500


class MockLLMProvider:
    def __init__(
        self,
        responses: Sequence[str],
        token_delay_s: float,
        sleep: AsyncSleep = asyncio.sleep,
    ) -> None:
        normalized = tuple(response.strip() for response in responses if response.strip())
        if not normalized:
            raise ValueError("at least one non-empty mock response is required")
        if token_delay_s < 0:
            raise ValueError("token delay cannot be negative")
        self._responses = normalized
        self._token_delay_s = token_delay_s
        self._sleep = sleep
        self._cursor = 0

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        turn_id: int,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        del messages, turn_id, session_id
        response_index = min(self._cursor, len(self._responses) - 1)
        self._cursor += 1
        for token in self._responses[response_index].split():
            await self._sleep(self._token_delay_s)
            yield token
