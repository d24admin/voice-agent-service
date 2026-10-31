from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias


AsyncSleep: TypeAlias = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


class LLMProvider(Protocol):
    def stream(
        self, messages: Sequence[ChatMessage], turn_id: int
    ) -> AsyncIterator[str]: ...


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
        self, messages: Sequence[ChatMessage], turn_id: int
    ) -> AsyncIterator[str]:
        del messages, turn_id
        response_index = min(self._cursor, len(self._responses) - 1)
        self._cursor += 1
        for token in self._responses[response_index].split():
            await self._sleep(self._token_delay_s)
            yield token
