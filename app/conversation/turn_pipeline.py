from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from app.api.events import AudioChunk, LLMToken, ServerEvent
from app.clock import Clock, SystemClock
from app.config import Settings
from app.conversation.turn_state import TurnState
from app.metrics import TurnOutcome
from app.providers.llm import (
    ChatMessage,
    LLMProvider,
    LLMProviderError,
    LLMProviderUnavailable,
)
from app.providers.tts import SentenceBuffer

__all__ = ["ActiveTurnRunner"]


Sleep = Callable[[float], Awaitable[None]]
EventEmitter = Callable[[ServerEvent, TurnState], Awaitable[bool]]
TurnFinisher = Callable[
    [TurnState, TurnOutcome, str | None], Awaitable[None]
]


@dataclass(frozen=True, slots=True)
class _EndOfStream:
    pass


END_OF_STREAM = _EndOfStream()


class _TurnInvalidated(Exception):
    pass


class _FirstTokenTimeout(Exception):
    pass


class _GenerationTimeout(Exception):
    pass


class ActiveTurnRunner:
    """Stream one answer through concurrent LLM and sentence-TTS tasks."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        emit: EventEmitter,
        finish: TurnFinisher,
        settings: Settings,
        session_id: str,
        clock: Clock | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._provider = provider
        self._emit = emit
        self._finish = finish
        self._settings = settings
        self._session_id = session_id
        self._clock = SystemClock() if clock is None else clock
        self._sleep = sleep

    async def run(
        self,
        turn: TurnState,
        messages: Sequence[ChatMessage],
    ) -> None:
        token_queue: asyncio.Queue[str | _EndOfStream] = asyncio.Queue(
            maxsize=self._settings.token_queue_size
        )
        provider_unavailable = asyncio.Event()
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(
                    self._produce(
                        turn,
                        messages,
                        token_queue,
                        provider_unavailable,
                    ),
                    name=f"turn-{turn.turn_id}-llm",
                )
                group.create_task(
                    self._speak(turn, token_queue),
                    name=f"turn-{turn.turn_id}-tts",
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._contains(exc, _TurnInvalidated):
                return
            await self._finish(turn, TurnOutcome.FAILED, self._failure_reason(exc))
            return

        if turn.output_valid:
            if provider_unavailable.is_set():
                await self._finish(turn, TurnOutcome.FAILED, "provider_error")
            else:
                await self._finish(turn, TurnOutcome.COMPLETED, None)

    async def _produce(
        self,
        turn: TurnState,
        messages: Sequence[ChatMessage],
        token_queue: asyncio.Queue[str | _EndOfStream],
        provider_unavailable: asyncio.Event,
    ) -> None:
        try:
            stream = self._provider.stream(messages, turn.turn_id, self._session_id)
        except TypeError:
            stream = self._provider.stream(messages, turn.turn_id)
        iterator = aiter(stream)
        published = False
        try:
            try:
                async with asyncio.timeout(
                    self._settings.generation_timeout_ms / 1_000
                ):
                    first = await self._first_token(iterator)
                    if first is not END_OF_STREAM:
                        await self._publish(turn, token_queue, first)
                        published = True
                        async for token in iterator:
                            await self._publish(turn, token_queue, token)
                            published = True
            except LLMProviderUnavailable:
                if published:
                    raise LLMProviderError from None
                provider_unavailable.set()
                for token in self._settings.llm_failure_message.split():
                    await self._publish(turn, token_queue, token)
            except _FirstTokenTimeout:
                raise
            except TimeoutError as exc:
                raise _GenerationTimeout from exc

            if not turn.output_valid:
                raise _TurnInvalidated
            await token_queue.put(END_OF_STREAM)
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result

    async def _first_token(self, iterator: object) -> str | _EndOfStream:
        try:
            async with asyncio.timeout(
                self._settings.first_token_timeout_ms / 1_000
            ):
                return await anext(iterator)  # type: ignore[arg-type]
        except StopAsyncIteration:
            return END_OF_STREAM
        except TimeoutError as exc:
            raise _FirstTokenTimeout from exc

    async def _publish(
        self,
        turn: TurnState,
        token_queue: asyncio.Queue[str | _EndOfStream],
        token: str,
    ) -> None:
        if not turn.output_valid:
            raise _TurnInvalidated
        accepted = await self._emit(
            LLMToken(
                event_id=self._event_id(),
                session_id=self._session_id,
                timestamp=self._clock.utc_now(),
                turn_id=turn.turn_id,
                token=token,
            ),
            turn,
        )
        if not accepted or not turn.output_valid:
            raise _TurnInvalidated
        turn.response_tokens.append(token)
        await token_queue.put(token)

    async def _speak(
        self,
        turn: TurnState,
        token_queue: asyncio.Queue[str | _EndOfStream],
    ) -> None:
        sentence_buffer = SentenceBuffer()
        sequence = 0
        while True:
            item = await token_queue.get()
            if item is END_OF_STREAM:
                trailing = sentence_buffer.flush()
                if trailing is not None:
                    await self._synthesize(turn, trailing, sequence)
                return
            for sentence in sentence_buffer.push(item):
                await self._synthesize(turn, sentence, sequence)
                sequence += 1

    async def _synthesize(
        self,
        turn: TurnState,
        sentence: str,
        sequence: int,
    ) -> None:
        if not turn.output_valid:
            raise _TurnInvalidated
        await self._sleep(self._settings.tts_sentence_delay_ms / 1_000)
        if not turn.output_valid:
            raise _TurnInvalidated
        accepted = await self._emit(
            AudioChunk(
                event_id=self._event_id(),
                session_id=self._session_id,
                timestamp=self._clock.utc_now(),
                turn_id=turn.turn_id,
                text=sentence,
                sequence=sequence,
            ),
            turn,
        )
        if not accepted:
            raise _TurnInvalidated

    @staticmethod
    def _contains(exc: BaseException, kind: type[BaseException]) -> bool:
        if isinstance(exc, kind):
            return True
        if isinstance(exc, BaseExceptionGroup):
            return any(
                ActiveTurnRunner._contains(child, kind)
                for child in exc.exceptions
            )
        return False

    @classmethod
    def _failure_reason(cls, exc: BaseException) -> str:
        if cls._contains(exc, _FirstTokenTimeout):
            return "first_token_timeout"
        if cls._contains(exc, _GenerationTimeout):
            return "generation_timeout"
        return "provider_error"

    @staticmethod
    def _event_id() -> str:
        return uuid.uuid4().hex
