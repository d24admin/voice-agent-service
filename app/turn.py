from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from app.clock import Clock, SystemClock
from app.config import Settings
from app.events import AudioChunk, LLMToken
from app.metrics import TurnMetrics, TurnOutcome
from app.outbound import LifecycleReservation, OutboundBuffer, TerminalCommand
from app.providers.llm import ChatMessage, LLMProvider
from app.providers.tts import SentenceBuffer

__all__ = ["ActiveTurnRunner", "TurnOutcome", "TurnState"]


Sleep = Callable[[float], Awaitable[None]]
QueueObserver = Callable[[str, int], None]


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


class _BackpressureTimeout(Exception):
    pass


@dataclass(slots=True)
class TurnState:
    turn_id: int
    metrics: TurnMetrics
    outcome: TurnOutcome = TurnOutcome.OPEN
    reason: str | None = None
    generation_open: bool = True
    playback_open: bool = True
    output_valid: bool = True
    lifecycle_reservation: LifecycleReservation | None = None
    user_text: str = ""
    response_tokens: list[str] = field(default_factory=list)
    history_committed: bool = False

    def interrupt(self, reason: str) -> bool:
        return self._claim_terminal(TurnOutcome.INTERRUPTED, reason, invalidate=True)

    def fail(self, reason: str) -> bool:
        return self._claim_terminal(TurnOutcome.FAILED, reason, invalidate=True)

    def mark_generation_finished(self) -> None:
        self.generation_open = False

    def complete_from_writer(self) -> bool:
        if self.outcome is not TurnOutcome.OPEN or self.generation_open:
            return False
        self.playback_open = False
        return self._claim_terminal(TurnOutcome.COMPLETED, None, invalidate=False)

    def _claim_terminal(
        self, outcome: TurnOutcome, reason: str | None, *, invalidate: bool
    ) -> bool:
        if self.outcome is not TurnOutcome.OPEN:
            return False
        self.outcome = outcome
        self.reason = reason
        if invalidate:
            self.output_valid = False
        return True


class ActiveTurnRunner:
    """Own one turn's concurrent LLM producer and sentence TTS consumer."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        outbound: OutboundBuffer,
        settings: Settings,
        session_id: str,
        clock: Clock | None = None,
        sleep: Sleep = asyncio.sleep,
        observe_queue: QueueObserver | None = None,
    ) -> None:
        self._provider = provider
        self._outbound = outbound
        self._settings = settings
        self._session_id = session_id
        self._clock = SystemClock() if clock is None else clock
        self._sleep = sleep
        self._observe_queue = observe_queue

    async def run(
        self, turn: TurnState, messages: Sequence[ChatMessage]
    ) -> None:
        if turn.lifecycle_reservation is None:
            try:
                turn.lifecycle_reservation = await self._outbound.reserve_turn(
                    turn.turn_id,
                    self._settings.backpressure_timeout_ms / 1_000,
                )
            except TimeoutError:
                turn.fail("lifecycle_capacity_timeout")
                return

        token_queue: asyncio.Queue[str | _EndOfStream] = asyncio.Queue(
            maxsize=self._settings.token_queue_size
        )
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(
                    self._llm_producer(turn, messages, token_queue),
                    name=f"turn-{turn.turn_id}-llm",
                )
                group.create_task(
                    self._tts_consumer(turn, token_queue),
                    name=f"turn-{turn.turn_id}-tts",
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._contains(exc, _TurnInvalidated):
                return
            reason = self._failure_reason(exc)
            if turn.fail(reason):
                self._queue_terminal(turn)

    async def _llm_producer(
        self,
        turn: TurnState,
        messages: Sequence[ChatMessage],
        token_queue: asyncio.Queue[str | _EndOfStream],
    ) -> None:
        try:
            stream = self._provider.stream(messages, turn.turn_id, self._session_id)
        except TypeError:
            # Backward-compatible adapter for simple two-argument assessment fakes.
            stream = self._provider.stream(messages, turn.turn_id)
        iterator = aiter(stream)
        try:
            try:
                async with asyncio.timeout(
                    self._settings.generation_timeout_ms / 1_000
                ):
                    first = await self._first_token(iterator)
                    if first is not END_OF_STREAM:
                        await self._publish_token(turn, token_queue, first)
                        async for token in iterator:
                            await self._publish_token(turn, token_queue, token)
            except _FirstTokenTimeout:
                raise
            except TimeoutError as exc:
                raise _GenerationTimeout from exc

            if not turn.output_valid:
                raise _TurnInvalidated
            turn.mark_generation_finished()
            await self._put_token_queue(token_queue, END_OF_STREAM)
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result

    async def _first_token(
        self, iterator: object
    ) -> str | _EndOfStream:
        try:
            async with asyncio.timeout(
                self._settings.first_token_timeout_ms / 1_000
            ):
                return await anext(iterator)  # type: ignore[arg-type]
        except StopAsyncIteration:
            return END_OF_STREAM
        except TimeoutError as exc:
            raise _FirstTokenTimeout from exc

    async def _publish_token(
        self,
        turn: TurnState,
        token_queue: asyncio.Queue[str | _EndOfStream],
        token: str,
    ) -> None:
        if not turn.output_valid:
            raise _TurnInvalidated
        await self._put_outbound(
            LLMToken(
                event_id=self._event_id(),
                session_id=self._session_id,
                timestamp=self._clock.utc_now(),
                turn_id=turn.turn_id,
                token=token,
            ),
            turn.turn_id,
        )
        turn.response_tokens.append(token)
        if not turn.output_valid:
            raise _TurnInvalidated
        await self._put_token_queue(token_queue, token)

    async def _tts_consumer(
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
                if not turn.output_valid:
                    raise _TurnInvalidated
                await self._put_output_complete(turn.turn_id)
                return

            for sentence in sentence_buffer.push(item):
                await self._synthesize(turn, sentence, sequence)
                sequence += 1

    async def _synthesize(
        self, turn: TurnState, sentence: str, sequence: int
    ) -> None:
        if not turn.output_valid:
            raise _TurnInvalidated
        await self._sleep(self._settings.tts_sentence_delay_ms / 1_000)
        if not turn.output_valid:
            raise _TurnInvalidated
        await self._put_outbound(
            AudioChunk(
                event_id=self._event_id(),
                session_id=self._session_id,
                timestamp=self._clock.utc_now(),
                turn_id=turn.turn_id,
                text=sentence,
                sequence=sequence,
            ),
            turn.turn_id,
        )

    async def _put_token_queue(
        self,
        queue: asyncio.Queue[str | _EndOfStream],
        item: str | _EndOfStream,
    ) -> None:
        try:
            async with asyncio.timeout(
                self._settings.backpressure_timeout_ms / 1_000
            ):
                await queue.put(item)
                if self._observe_queue is not None:
                    self._observe_queue("token", queue.qsize())
        except TimeoutError as exc:
            raise _BackpressureTimeout from exc

    async def _put_outbound(self, event: LLMToken | AudioChunk, turn_id: int) -> None:
        try:
            async with asyncio.timeout(
                self._settings.backpressure_timeout_ms / 1_000
            ):
                await self._outbound.put_data(
                    event,
                    turn_id=turn_id,
                    fenced=True,
                )
                if self._observe_queue is not None:
                    self._observe_queue("outbound_data", self._outbound.data_size)
        except TimeoutError as exc:
            raise _BackpressureTimeout from exc

    async def _put_output_complete(self, turn_id: int) -> None:
        try:
            async with asyncio.timeout(
                self._settings.backpressure_timeout_ms / 1_000
            ):
                await self._outbound.put_output_complete(turn_id)
        except TimeoutError as exc:
            raise _BackpressureTimeout from exc

    def _queue_terminal(self, turn: TurnState) -> None:
        reservation = turn.lifecycle_reservation
        if reservation is None:
            return
        self._outbound.put_lifecycle_nowait(
            reservation,
            TerminalCommand(
                turn_id=turn.turn_id,
                outcome=turn.outcome,
                reason=turn.reason,
            ),
        )

    @staticmethod
    def _contains(exc: BaseException, kind: type[BaseException]) -> bool:
        if isinstance(exc, kind):
            return True
        if isinstance(exc, BaseExceptionGroup):
            return any(ActiveTurnRunner._contains(child, kind) for child in exc.exceptions)
        return False

    @classmethod
    def _failure_reason(cls, exc: BaseException) -> str:
        if cls._contains(exc, _FirstTokenTimeout):
            return "first_token_timeout"
        if cls._contains(exc, _GenerationTimeout):
            return "generation_timeout"
        if cls._contains(exc, _BackpressureTimeout):
            return "backpressure_timeout"
        return "provider_error"

    @staticmethod
    def _event_id() -> str:
        return uuid.uuid4().hex
