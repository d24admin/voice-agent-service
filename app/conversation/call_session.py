from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from dataclasses import dataclass

from app.api.events import (
    AudioChunk,
    AudioEnd,
    AudioFrame,
    ClientEvent,
    ErrorEvent,
    LLMToken,
    Pong,
    ServerEvent,
    SessionReady,
    SessionStart,
    SessionStop,
    TranscriptFinal,
    TranscriptPartial,
    TurnCompleted,
    TurnFailed,
    TurnInterrupted,
)
from app.api.websocket_transport import (
    EventTransport,
    ProtocolError,
    TransportDisconnected,
)
from app.clock import Clock
from app.config import Settings
from app.conversation.turn_pipeline import ActiveTurnRunner
from app.conversation.turn_state import TurnState
from app.metrics import MetricsRegistry, SessionMetrics, TurnMetrics, TurnOutcome
from app.providers.llm import ChatMessage, LLMProvider
from app.providers.stt import FinalResult, SimulatedSTT


logger = logging.getLogger("voice_agent")


@dataclass(frozen=True, slots=True)
class _QueuedEvent:
    event: ServerEvent
    turn_id: int | None = None
    fenced: bool = False


@dataclass(frozen=True, slots=True)
class _Terminal:
    turn_id: int
    outcome: TurnOutcome
    reason: str | None


@dataclass(frozen=True, slots=True)
class _StopWriter:
    pass


STOP_WRITER = _StopWriter()
OutboundItem = _QueuedEvent | _Terminal | _StopWriter


class CallSession:
    """Own one WebSocket call, its active answer, and all child tasks."""

    def __init__(
        self,
        *,
        session_id: str,
        settings: Settings,
        transport: EventTransport,
        provider: LLMProvider,
        registry: MetricsRegistry,
        clock: Clock,
    ) -> None:
        self.session_id = session_id
        self._settings = settings
        self._transport = transport
        self._provider = provider
        self._clock = clock
        self._session_metrics: SessionMetrics = registry.start_session(session_id)
        self._output: asyncio.Queue[OutboundItem] = asyncio.Queue(
            maxsize=settings.output_queue_size
        )
        self._stt = SimulatedSTT()
        self._silence_deadline: float | None = None
        self._turns: dict[int, TurnState] = {}
        self._turn_tasks: set[asyncio.Task[None]] = set()
        self._active_turn_task: asyncio.Task[None] | None = None
        self._current_turn_id: int | None = None
        self._next_turn_id = 1
        self._history = [
            ChatMessage(role="system", content=settings.system_prompt)
        ]
        self._protocol_state = "awaiting_start"
        self._requested_close_code = 1000
        self._shutdown_requested = asyncio.Event()
        self._closed = asyncio.Event()
        self._run_task: asyncio.Task[None] | None = None

    @property
    def turns(self) -> dict[int, TurnState]:
        return dict(self._turns)

    @property
    def output_queue_size(self) -> int:
        return self._output.qsize()

    async def request_shutdown(self) -> None:
        self._shutdown_requested.set()

    async def wait_closed(self) -> None:
        await self._closed.wait()

    def force_cancel(self) -> None:
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()

    async def run(self) -> None:
        self._run_task = asyncio.current_task()
        prefix = f"voice-session-{self.session_id}:"
        writer = asyncio.create_task(self._write_loop(), name=f"{prefix}writer")
        receiver = asyncio.create_task(
            self._transport.receive(), name=f"{prefix}receive"
        )
        shutdown = asyncio.create_task(
            self._shutdown_requested.wait(), name=f"{prefix}shutdown-wait"
        )
        try:
            while True:
                timeout = self._silence_timeout()
                done, _ = await asyncio.wait(
                    {receiver, shutdown, writer},
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    self._silence_deadline = None
                    final = self._stt.finalize()
                    if final is not None:
                        await self._start_turn(final)
                    continue
                if writer in done:
                    await writer
                    raise RuntimeError("outbound writer stopped unexpectedly")
                if shutdown in done:
                    break
                if receiver in done:
                    try:
                        event = receiver.result()
                    except ProtocolError as exc:
                        should_close = await self._handle_protocol_error(exc)
                        if should_close:
                            break
                    except TransportDisconnected:
                        self._requested_close_code = 1001
                        break
                    else:
                        received_ns = self._clock.monotonic_ns()
                        if await self._handle_event(event, received_ns):
                            break
                    receiver = asyncio.create_task(
                        self._transport.receive(), name=f"{prefix}receive"
                    )
        except asyncio.CancelledError:
            self._requested_close_code = 1001
            raise
        except Exception:
            self._requested_close_code = 1011
            raise
        finally:
            cleanup_cancelled = False
            try:
                async with asyncio.timeout(
                    self._settings.shutdown_grace_ms / 1_000
                ):
                    await self._graceful_cleanup(writer, receiver, shutdown)
            except TimeoutError:
                logger.warning(
                    "session_cleanup_timeout session_id=%s", self.session_id
                )
            except asyncio.CancelledError:
                cleanup_cancelled = True
            finally:
                owned = (receiver, shutdown, writer, *tuple(self._turn_tasks))
                for task in owned:
                    if not task.done():
                        task.cancel()
                try:
                    await asyncio.gather(*owned, return_exceptions=True)
                finally:
                    self._record_unwritten_turns()
                    try:
                        await self._transport.close(self._requested_close_code)
                    finally:
                        self._closed.set()
            logger.info(
                "session_closed session_id=%s close_code=%s",
                self.session_id,
                self._requested_close_code,
            )
            if cleanup_cancelled:
                raise asyncio.CancelledError

    async def _graceful_cleanup(
        self,
        writer: asyncio.Task[None],
        receiver: asyncio.Task[ClientEvent],
        shutdown: asyncio.Task[bool],
    ) -> None:
        for task in (receiver, shutdown):
            if not task.done():
                task.cancel()
        await asyncio.gather(receiver, shutdown, return_exceptions=True)

        if self._current_turn_id is not None:
            await self._interrupt_current("session_ended")
        for task in tuple(self._turn_tasks):
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._turn_tasks, return_exceptions=True)
        await self._close_provider()

        if writer.done():
            await asyncio.gather(writer, return_exceptions=True)
            return
        await self._output.join()
        await self._output.put(STOP_WRITER)
        await writer

    def _silence_timeout(self) -> float | None:
        if self._silence_deadline is None:
            return None
        return max(0.0, self._silence_deadline - asyncio.get_running_loop().time())

    async def _handle_event(self, event: ClientEvent, received_ns: int) -> bool:
        if event.session_id != self.session_id:
            await self._fatal_policy(
                "session_id_mismatch",
                "event session_id does not match the connection",
            )
            return True
        if event.type == "ping":
            await self._put_event(
                Pong(
                    event_id=self._event_id(),
                    session_id=self.session_id,
                    timestamp=self._clock.utc_now(),
                )
            )
            return False
        if isinstance(event, SessionStart):
            if self._protocol_state != "awaiting_start":
                await self._fatal_policy(
                    "duplicate_start", "session.start is only valid once"
                )
                return True
            self._protocol_state = "active"
            await self._put_event(
                SessionReady(
                    event_id=self._event_id(),
                    session_id=self.session_id,
                    timestamp=self._clock.utc_now(),
                )
            )
            logger.info("session_started session_id=%s", self.session_id)
            return False
        if isinstance(event, SessionStop):
            self._protocol_state = "closing"
            self._silence_deadline = None
            return True
        if self._protocol_state != "active":
            await self._fatal_policy(
                "invalid_state", "audio events require an active session"
            )
            return True
        if isinstance(event, AudioFrame):
            await self._handle_audio_frame(event, received_ns)
            return False
        if isinstance(event, AudioEnd):
            self._silence_deadline = None
            final = self._stt.finalize()
            if final is not None:
                await self._start_turn(final, baseline_ns=received_ns)
        return False

    async def _handle_protocol_error(self, error: ProtocolError) -> bool:
        if error.emit_error:
            await self._put_event(
                ErrorEvent(
                    event_id=self._event_id(),
                    session_id=self.session_id,
                    timestamp=self._clock.utc_now(),
                    code=error.code,
                    message=error.safe_message,
                    fatal=error.fatal,
                )
            )
        if not error.fatal:
            return False
        self._requested_close_code = error.close_code or 1008
        self._protocol_state = "closing"
        await self._fail_current("session_aborted")
        return True

    async def _fatal_policy(self, code: str, message: str) -> None:
        await self._put_event(
            ErrorEvent(
                event_id=self._event_id(),
                session_id=self.session_id,
                timestamp=self._clock.utc_now(),
                code=code,
                message=message,
                fatal=True,
            )
        )
        self._requested_close_code = 1008
        self._protocol_state = "closing"
        await self._fail_current("session_aborted")

    async def _handle_audio_frame(
        self,
        event: AudioFrame,
        received_ns: int,
    ) -> None:
        await self._interrupt_current("barge_in")
        partial = self._stt.accept_frame(event.text, received_ns)
        self._silence_deadline = (
            asyncio.get_running_loop().time()
            + self._settings.silence_timeout_ms / 1_000
        )
        await asyncio.sleep(self._settings.stt_partial_delay_ms / 1_000)
        await self._put_event(
            TranscriptPartial(
                event_id=self._event_id(),
                session_id=self.session_id,
                timestamp=self._clock.utc_now(),
                utterance_id=partial.utterance_id,
                text=partial.text,
            )
        )

    async def _start_turn(
        self,
        final: FinalResult,
        baseline_ns: int | None = None,
    ) -> None:
        await self._put_event(
            TranscriptFinal(
                event_id=self._event_id(),
                session_id=self.session_id,
                timestamp=self._clock.utc_now(),
                utterance_id=final.utterance_id,
                text=final.text,
            )
        )
        turn_id = self._next_turn_id
        self._next_turn_id += 1
        turn = TurnState(
            turn_id=turn_id,
            metrics=TurnMetrics(
                turn_id=turn_id,
                baseline_ns=(
                    final.last_frame_ns if baseline_ns is None else baseline_ns
                ),
            ),
            user_text=final.text,
        )
        self._turns[turn_id] = turn
        self._session_metrics.mark_started(turn_id)
        runner = ActiveTurnRunner(
            provider=self._provider,
            emit=self._emit_stream,
            finish=self._finish_turn,
            settings=self._settings,
            session_id=self.session_id,
            clock=self._clock,
        )
        messages = (*self._history, ChatMessage(role="user", content=final.text))
        task = asyncio.create_task(
            runner.run(turn, messages),
            name=f"voice-session-{self.session_id}:turn-{turn_id}",
        )
        self._turn_tasks.add(task)
        task.add_done_callback(self._turn_task_done)
        self._active_turn_task = task
        self._current_turn_id = turn_id
        logger.info(
            "turn_started session_id=%s turn_id=%s", self.session_id, turn_id
        )

    async def _emit_stream(self, event: ServerEvent, turn: TurnState) -> bool:
        if not turn.output_valid:
            return False
        await self._output.put(
            _QueuedEvent(event=event, turn_id=turn.turn_id, fenced=True)
        )
        return turn.output_valid

    async def _finish_turn(
        self,
        turn: TurnState,
        outcome: TurnOutcome,
        reason: str | None,
    ) -> None:
        await self._output.put(_Terminal(turn.turn_id, outcome, reason))

    async def _interrupt_current(self, reason: str) -> TurnState | None:
        turn_id = self._current_turn_id
        if turn_id is None:
            return None
        turn = self._turns[turn_id]
        if not turn.interrupt(reason):
            return None
        task = self._active_turn_task
        if task is not None and not task.done():
            task.cancel()
        await self._output.put(
            _Terminal(turn.turn_id, TurnOutcome.INTERRUPTED, reason)
        )
        return turn

    async def _fail_current(self, reason: str) -> TurnState | None:
        turn_id = self._current_turn_id
        if turn_id is None:
            return None
        turn = self._turns[turn_id]
        if not turn.fail(reason):
            return None
        task = self._active_turn_task
        if task is not None and not task.done():
            task.cancel()
        await self._output.put(_Terminal(turn.turn_id, TurnOutcome.FAILED, reason))
        return turn

    async def _put_event(self, event: ServerEvent) -> None:
        await self._output.put(_QueuedEvent(event))

    async def _write_loop(self) -> None:
        while True:
            item = await self._output.get()
            try:
                if item is STOP_WRITER:
                    return
                if isinstance(item, _QueuedEvent):
                    await self._write_event(item)
                else:
                    await self._write_terminal(item)
            finally:
                self._output.task_done()

    async def _write_event(self, item: _QueuedEvent) -> None:
        if item.fenced and item.turn_id is not None:
            turn = self._turns[item.turn_id]
            if not turn.output_valid:
                return
        await self._transport.send(item.event)
        if item.turn_id is None:
            return
        turn = self._turns[item.turn_id]
        now_ns = self._clock.monotonic_ns()
        if isinstance(item.event, LLMToken):
            turn.metrics.record_token_sent(now_ns)
        elif isinstance(item.event, AudioChunk):
            turn.metrics.record_audio_sent(now_ns)

    async def _write_terminal(self, item: _Terminal) -> None:
        turn = self._turns[item.turn_id]
        if turn.terminal_claimed:
            return
        if item.outcome is TurnOutcome.COMPLETED:
            if not turn.complete():
                return
        elif item.outcome is TurnOutcome.FAILED:
            if turn.outcome is TurnOutcome.OPEN:
                if not turn.fail(item.reason or "provider_error"):
                    return
            elif turn.outcome is not TurnOutcome.FAILED:
                return
        elif turn.outcome is not item.outcome:
            return

        turn.terminal_claimed = True
        turn.metrics.finalize(turn.outcome, turn.reason, self._clock.monotonic_ns())
        terminal = self._terminal_event(turn)
        await self._transport.send(terminal)
        self._session_metrics.record_turn(turn.metrics)
        if turn.outcome is TurnOutcome.COMPLETED:
            self._commit_history(turn)
        if self._current_turn_id == turn.turn_id:
            self._current_turn_id = None
            self._active_turn_task = None
        logger.info(
            "turn_terminal session_id=%s turn_id=%s outcome=%s reason=%s",
            self.session_id,
            turn.turn_id,
            turn.outcome.value,
            turn.reason or "none",
        )

    def _terminal_event(self, turn: TurnState) -> ServerEvent:
        common = {
            "event_id": self._event_id(),
            "session_id": self.session_id,
            "timestamp": self._clock.utc_now(),
            "turn_id": turn.turn_id,
            "metrics": turn.metrics.model_dump(mode="json"),
        }
        if turn.outcome is TurnOutcome.COMPLETED:
            return TurnCompleted(**common)
        if turn.outcome is TurnOutcome.INTERRUPTED:
            return TurnInterrupted(reason=turn.reason or "interrupted", **common)
        return TurnFailed(reason=turn.reason or "provider_error", **common)

    def _commit_history(self, turn: TurnState) -> None:
        if turn.history_committed:
            return
        response = " ".join(turn.response_tokens).strip()
        self._history.append(ChatMessage(role="user", content=turn.user_text))
        if response:
            self._history.append(ChatMessage(role="assistant", content=response))
        turn.history_committed = True

    def _turn_task_done(self, task: asyncio.Task[None]) -> None:
        self._turn_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    def _record_unwritten_turns(self) -> None:
        for turn in self._turns.values():
            if turn.metrics.total_latency_ms is None:
                if turn.outcome is TurnOutcome.OPEN:
                    turn.interrupt("session_ended")
                turn.metrics.finalize(
                    turn.outcome, turn.reason, self._clock.monotonic_ns()
                )
            self._session_metrics.record_turn(turn.metrics)

    async def _close_provider(self) -> None:
        close = getattr(self._provider, "aclose", None)
        if close is None:
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.warning(
                "provider_cleanup_failed session_id=%s", self.session_id
            )

    @staticmethod
    def _event_id() -> str:
        return uuid.uuid4().hex
