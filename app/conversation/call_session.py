from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field

from app.clock import Clock
from app.config import Settings
from app.events import (
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
from app.metrics import MetricsRegistry, SessionMetrics, TurnMetrics, TurnOutcome
from app.conversation.outbound_buffer import (
    DropAction,
    OutboundBuffer,
    OutboundClosedError,
    OutputCompleteAction,
    SendAction,
    TerminalCommand,
)
from app.providers.llm import ChatMessage, LLMProvider
from app.providers.stt import FinalResult, SimulatedSTT
from app.transport import EventTransport, ProtocolError, TransportDisconnected
from app.conversation.turn_pipeline import ActiveTurnRunner
from app.conversation.turn_state import TurnState


@dataclass(frozen=True, slots=True)
class InboundEnvelope:
    event: ClientEvent
    received_ns: int


@dataclass(frozen=True, slots=True)
class SilenceExpired:
    timer_generation: int


@dataclass(slots=True)
class ProtocolIssue:
    code: str
    message: str
    fatal: bool
    close_code: int | None
    emit_error: bool
    delivered: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass(frozen=True, slots=True)
class ShutdownRequested:
    close_code: int = 1000


InboundMessage = InboundEnvelope | SilenceExpired | ProtocolIssue | ShutdownRequested


logger = logging.getLogger("voice_agent")


class CallSession:
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
        self._session_metrics = registry.get_or_create(session_id)
        self._inbound: asyncio.Queue[InboundMessage] = asyncio.Queue(
            maxsize=settings.inbound_queue_size
        )
        self._outbound = OutboundBuffer(
            data_capacity=settings.outbound_data_capacity,
            lifecycle_capacity=settings.outbound_lifecycle_capacity,
        )
        self._stt = SimulatedSTT()
        self._silence_timer: asyncio.Task[None] | None = None
        self._active_turn_task: asyncio.Task[None] | None = None
        self._current_turn_id: int | None = None
        self._supervisors: set[asyncio.Task[None]] = set()
        self._retiring_turns: set[asyncio.Task[None]] = set()
        self._retire_monitors: set[asyncio.Task[None]] = set()
        self._turns: dict[int, TurnState] = {}
        self._next_turn_id = self._session_metrics.next_turn_id
        self._history = [
            ChatMessage(role="system", content=settings.system_prompt)
        ]
        self._protocol_state = "awaiting_start"
        self._requested_close_code = 1000
        self._abort_outcome = TurnOutcome.INTERRUPTED
        self._abort_reason = "session_ended"
        self._abort_requested = asyncio.Event()
        self._delivery_waiters: dict[str, asyncio.Event] = {}
        self._terminal_delivery: dict[int, asyncio.Event] = {}
        self._closed = asyncio.Event()
        self._run_task: asyncio.Task[None] | None = None

    @property
    def turns(self) -> dict[int, TurnState]:
        return dict(self._turns)

    @property
    def retiring_turn_count(self) -> int:
        return len(self._retiring_turns)

    async def request_shutdown(self) -> None:
        if self._closed.is_set():
            return
        await self._put_inbound(ShutdownRequested(), apply_backpressure=True)

    async def wait_closed(self) -> None:
        await self._closed.wait()

    def force_cancel(self) -> None:
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()

    async def run(self) -> None:
        self._run_task = asyncio.current_task()
        prefix = f"voice-session-{self.session_id}:"
        receiver = asyncio.create_task(
            self._receive_loop(), name=f"{prefix}receiver"
        )
        writer = asyncio.create_task(self._write_loop(), name=f"{prefix}writer")
        reaper = asyncio.create_task(
            self._abort_requested.wait(), name=f"{prefix}abort-waiter"
        )
        orchestrator = asyncio.create_task(
            self._orchestrate(), name=f"{prefix}orchestrator"
        )
        close_code = self._requested_close_code
        try:
            await self._wait_for_session_end(receiver, writer, orchestrator, reaper)
        except asyncio.CancelledError:
            close_code = 1001
            raise
        except Exception:
            close_code = 1011
            raise
        finally:
            if close_code == 1000:
                close_code = self._requested_close_code
            await self._cancel_silence_timer()
            owned = [receiver, writer, reaper, orchestrator]
            owned.extend(self._supervisors)
            owned.extend(self._retire_monitors)
            for task in owned:
                if not task.done():
                    task.cancel()
            await self._outbound.close()
            await asyncio.gather(*owned, return_exceptions=True)
            self._finalize_retained_turns()
            await self._transport.close(close_code)
            self._closed.set()
            logger.info(
                "session_closed session_id=%s close_code=%s",
                self.session_id,
                close_code,
            )

    async def _wait_for_session_end(
        self,
        receiver: asyncio.Task[None],
        writer: asyncio.Task[None],
        orchestrator: asyncio.Task[None],
        abort_waiter: asyncio.Task[bool],
    ) -> None:
        watched = {receiver, writer, orchestrator, abort_waiter}
        while watched:
            done, _ = await asyncio.wait(
                watched, return_when=asyncio.FIRST_COMPLETED
            )
            if orchestrator in done:
                await orchestrator
                return
            if abort_waiter in done:
                return
            if writer in done:
                await writer
                raise RuntimeError("outbound writer stopped unexpectedly")
            if receiver in done:
                await receiver
                watched.remove(receiver)

    async def _receive_loop(self) -> None:
        while True:
            try:
                event = await self._transport.receive()
            except ProtocolError as exc:
                issue = ProtocolIssue(
                    code=exc.code,
                    message=exc.safe_message,
                    fatal=exc.fatal,
                    close_code=exc.close_code,
                    emit_error=exc.emit_error,
                )
                accepted = await self._put_inbound(issue, apply_backpressure=True)
                if not accepted:
                    return
                if exc.fatal:
                    return
                continue
            except TransportDisconnected:
                self._request_abort(1001)
                return
            envelope = InboundEnvelope(
                event=event,
                received_ns=self._clock.monotonic_ns(),
            )
            accepted = await self._put_inbound(envelope, apply_backpressure=True)
            if not accepted:
                return
            if isinstance(event, SessionStop):
                return

    async def _put_inbound(
        self, message: InboundMessage, *, apply_backpressure: bool
    ) -> bool:
        if not apply_backpressure:
            self._inbound.put_nowait(message)
            self._session_metrics.observe_queue("inbound", self._inbound.qsize())
            return True
        try:
            async with asyncio.timeout(
                self._settings.backpressure_timeout_ms / 1_000
            ):
                await self._inbound.put(message)
                self._session_metrics.observe_queue("inbound", self._inbound.qsize())
            return True
        except TimeoutError:
            self._session_metrics.overload_count += 1
            self._request_abort(
                1013,
                outcome=TurnOutcome.FAILED,
                reason="session_aborted",
            )
            return False

    async def _orchestrate(self) -> None:
        while True:
            message = await self._inbound.get()
            if isinstance(message, ShutdownRequested):
                self._requested_close_code = message.close_code
                await self._end_current_turn("session_ended")
                return
            if isinstance(message, ProtocolIssue):
                await self._handle_protocol_issue(message)
                if message.fatal:
                    return
                continue
            if isinstance(message, SilenceExpired):
                final = self._stt.finalize(message.timer_generation)
                if final is not None:
                    await self._start_turn(final)
                continue

            event = message.event
            if event.session_id != self.session_id:
                await self._fatal_policy(
                    "session_id_mismatch",
                    "event session_id does not match the connection",
                )
                return
            if event.type == "ping":
                if not await self._put_control(
                    Pong(
                        event_id=self._event_id(),
                        session_id=self.session_id,
                        timestamp=self._clock.utc_now(),
                    )
                ):
                    return
                continue
            if isinstance(event, SessionStart):
                if self._protocol_state != "awaiting_start":
                    await self._fatal_policy(
                        "duplicate_start",
                        "session.start is only valid once",
                    )
                    return
                self._protocol_state = "active"
                if not await self._put_control(
                    SessionReady(
                        event_id=self._event_id(),
                        session_id=self.session_id,
                        timestamp=self._clock.utc_now(),
                    )
                ):
                    return
                logger.info("session_started session_id=%s", self.session_id)
                continue
            if isinstance(event, SessionStop):
                self._protocol_state = "closing"
                await self._end_current_turn("session_ended")
                return
            if self._protocol_state != "active":
                await self._fatal_policy(
                    "invalid_state",
                    "audio events require an active session",
                )
                return
            if isinstance(event, AudioFrame):
                await self._handle_audio_frame(event, message.received_ns)
                continue
            if isinstance(event, AudioEnd):
                await self._cancel_silence_timer()
                final = self._stt.finalize()
                if final is not None:
                    await self._start_turn(final, baseline_ns=message.received_ns)
                continue

    async def _handle_protocol_issue(self, issue: ProtocolIssue) -> None:
        if issue.emit_error:
            event = ErrorEvent(
                event_id=self._event_id(),
                session_id=self.session_id,
                timestamp=self._clock.utc_now(),
                code=issue.code,
                message=issue.message,
                fatal=issue.fatal,
            )
            if issue.fatal:
                self._delivery_waiters[event.event_id] = issue.delivered
            if not await self._put_control(event):
                return
            if issue.fatal:
                try:
                    async with asyncio.timeout(
                        self._settings.backpressure_timeout_ms / 1_000
                    ):
                        await issue.delivered.wait()
                except TimeoutError:
                    pass
                finally:
                    self._delivery_waiters.pop(event.event_id, None)
        if issue.fatal:
            self._protocol_state = "closing"
            self._requested_close_code = issue.close_code or 1008
            await self._fail_current_turn("session_aborted")

    async def _fatal_policy(self, code: str, message: str) -> None:
        await self._handle_protocol_issue(
            ProtocolIssue(
                code=code,
                message=message,
                fatal=True,
                close_code=1008,
                emit_error=True,
            )
        )

    async def _handle_audio_frame(
        self, event: AudioFrame, received_ns: int
    ) -> None:
        self._interrupt_for_barge_in()
        partial = self._stt.accept_frame(event.text, received_ns)
        await self._cancel_silence_timer()
        self._silence_timer = asyncio.create_task(
            self._silence_after(partial.generation),
            name=(
                f"voice-session-{self.session_id}:silence-"
                f"{partial.generation}"
            ),
        )
        await asyncio.sleep(self._settings.stt_partial_delay_ms / 1_000)
        if not await self._put_control(
            TranscriptPartial(
                event_id=self._event_id(),
                session_id=self.session_id,
                timestamp=self._clock.utc_now(),
                utterance_id=partial.utterance_id,
                text=partial.text,
            )
        ):
            return

    async def _silence_after(self, timer_generation: int) -> None:
        await asyncio.sleep(self._settings.silence_timeout_ms / 1_000)
        await self._put_inbound(
            SilenceExpired(timer_generation), apply_backpressure=True
        )

    async def _cancel_silence_timer(self) -> None:
        timer = self._silence_timer
        self._silence_timer = None
        if timer is None or timer.done():
            if timer is not None:
                await asyncio.gather(timer, return_exceptions=True)
            return
        timer.cancel()
        await asyncio.gather(timer, return_exceptions=True)

    async def _start_turn(
        self, final: FinalResult, baseline_ns: int | None = None
    ) -> None:
        if not await self._put_control(
            TranscriptFinal(
                event_id=self._event_id(),
                session_id=self.session_id,
                timestamp=self._clock.utc_now(),
                utterance_id=final.utterance_id,
                text=final.text,
            )
        ):
            return
        turn_id = self._next_turn_id
        self._next_turn_id += 1
        turn = TurnState(
            turn_id=turn_id,
            metrics=TurnMetrics(
                turn_id=turn_id,
                baseline_ns=final.last_frame_ns if baseline_ns is None else baseline_ns,
                terminal_delivered=False,
            ),
            user_text=final.text,
        )
        try:
            turn.lifecycle_reservation = await self._outbound.reserve_turn(
                turn_id,
                self._settings.backpressure_timeout_ms / 1_000,
            )
        except TimeoutError:
            turn.fail("session_aborted")
            turn.metrics.finalize(
                TurnOutcome.FAILED, turn.reason, self._clock.monotonic_ns()
            )
            self._session_metrics.record_turn(turn.metrics)
            self._session_metrics.overload_count += 1
            self._request_abort(
                1013,
                outcome=TurnOutcome.FAILED,
                reason="session_aborted",
            )
            return
        self._session_metrics.observe_queue(
            "outbound_lifecycle", self._outbound.reserved_count
        )
        self._turns[turn_id] = turn
        self._session_metrics.mark_started(turn_id)
        runner = ActiveTurnRunner(
            provider=self._provider,
            outbound=self._outbound,
            settings=self._settings,
            session_id=self.session_id,
            clock=self._clock,
            observe_queue=self._session_metrics.observe_queue,
        )
        messages = tuple(
            [*self._history, ChatMessage(role="user", content=final.text)]
        )
        supervisor = asyncio.create_task(
            runner.run(turn, messages),
            name=f"voice-session-{self.session_id}:turn-{turn_id}",
        )
        self._active_turn_task = supervisor
        self._supervisors.add(supervisor)
        supervisor.add_done_callback(
            lambda task, owned_turn=turn: self._supervisor_finished(
                task, owned_turn
            )
        )
        self._current_turn_id = turn_id
        logger.info(
            "turn_started session_id=%s turn_id=%s",
            self.session_id,
            turn_id,
        )

    def _interrupt_for_barge_in(self) -> None:
        self._interrupt_current("barge_in")

    def _interrupt_current(self, reason: str) -> TurnState | None:
        return self._claim_current(TurnOutcome.INTERRUPTED, reason)

    def _claim_current(
        self, outcome: TurnOutcome, reason: str
    ) -> TurnState | None:
        turn_id = self._current_turn_id
        if turn_id is None:
            return None
        turn = self._turns[turn_id]
        if not (turn.generation_open or turn.playback_open):
            return None
        claimed = (
            turn.interrupt(reason)
            if outcome is TurnOutcome.INTERRUPTED
            else turn.fail(reason)
        )
        if not claimed:
            return None

        task = self._active_turn_task
        if task is not None and not task.done():
            task.cancel()
            self._retiring_turns.add(task)
            monitor = asyncio.create_task(
                self._monitor_retiring(task),
                name=f"voice-session-{self.session_id}:retire-{turn.turn_id}",
            )
            self._retire_monitors.add(monitor)
            monitor.add_done_callback(self._retire_monitors.discard)
            if len(self._retiring_turns) > self._settings.outbound_lifecycle_capacity:
                self._session_metrics.overload_count += 1
                self._request_abort(
                    1013,
                    outcome=TurnOutcome.FAILED,
                    reason="session_aborted",
                )
        reservation = turn.lifecycle_reservation
        if reservation is None:
            raise RuntimeError("active turn has no lifecycle reservation")
        self._terminal_delivery[turn.turn_id] = asyncio.Event()
        self._outbound.put_lifecycle_nowait(
            reservation,
            TerminalCommand(
                turn_id=turn.turn_id,
                outcome=outcome,
                reason=reason,
            ),
        )
        self._session_metrics.observe_queue(
            "outbound_lifecycle", self._outbound.lifecycle_size
        )
        return turn

    async def _end_current_turn(self, reason: str) -> None:
        turn = self._claim_current(TurnOutcome.INTERRUPTED, reason)
        await self._wait_terminal_delivery(turn)

    async def _fail_current_turn(self, reason: str) -> None:
        turn = self._claim_current(TurnOutcome.FAILED, reason)
        await self._wait_terminal_delivery(turn)

    async def _wait_terminal_delivery(self, turn: TurnState | None) -> None:
        if turn is None:
            return
        delivered = self._terminal_delivery[turn.turn_id]
        try:
            async with asyncio.timeout(
                self._settings.backpressure_timeout_ms / 1_000
            ):
                await delivered.wait()
        except TimeoutError:
            pass

    async def _put_control(self, event: ServerEvent) -> bool:
        try:
            async with asyncio.timeout(
                self._settings.backpressure_timeout_ms / 1_000
            ):
                await self._outbound.put_data(event)
            self._session_metrics.observe_queue(
                "outbound_data", self._outbound.data_size
            )
            return True
        except TimeoutError:
            self._session_metrics.overload_count += 1
            self._request_abort(
                1013,
                outcome=TurnOutcome.FAILED,
                reason="session_aborted",
            )
            return False

    async def _write_loop(self) -> None:
        while True:
            try:
                action = await self._outbound.next_dispatch(self._is_output_valid)
            except OutboundClosedError:
                return
            if isinstance(action, DropAction):
                self._session_metrics.stale_event_drop_count += 1
                continue
            if isinstance(action, OutputCompleteAction):
                await self._write_completion(action)
                continue
            await self._write_send_action(action)

    async def _write_send_action(self, action: SendAction) -> None:
        item = action.item
        if isinstance(item, TerminalCommand):
            turn = self._turns[item.turn_id]
            turn.metrics.finalize(
                item.outcome,
                item.reason,
                self._clock.monotonic_ns(),
            )
            turn.metrics.terminal_delivered = True
            event = self._terminal_event(turn)
            try:
                await self._transport.send(event)
            except BaseException:
                turn.metrics.terminal_delivered = False
                self._session_metrics.record_turn(turn.metrics)
                raise
            self._session_metrics.record_turn(turn.metrics)
            turn.playback_open = False
            delivered = self._terminal_delivery.get(turn.turn_id)
            if delivered is not None:
                delivered.set()
            logger.info(
                "turn_terminal session_id=%s turn_id=%s outcome=%s",
                self.session_id,
                turn.turn_id,
                turn.outcome.value,
            )
            await self._outbound.ack(action)
            return

        await self._transport.send(item)
        waiter = self._delivery_waiters.get(item.event_id)
        if waiter is not None:
            waiter.set()
        if isinstance(item, LLMToken):
            self._turns[item.turn_id].metrics.record_token_sent(
                self._clock.monotonic_ns()
            )
        elif isinstance(item, AudioChunk):
            self._turns[item.turn_id].metrics.record_audio_sent(
                self._clock.monotonic_ns()
            )

    async def _write_completion(self, action: OutputCompleteAction) -> None:
        turn = self._turns[action.turn_id]
        if not turn.complete_from_writer():
            return
        turn.metrics.finalize(
            TurnOutcome.COMPLETED,
            None,
            self._clock.monotonic_ns(),
        )
        turn.metrics.terminal_delivered = True
        event = TurnCompleted(
            event_id=self._event_id(),
            session_id=self.session_id,
            timestamp=self._clock.utc_now(),
            turn_id=turn.turn_id,
            metrics=turn.metrics.model_dump(mode="json"),
        )
        try:
            await self._transport.send(event)
        except BaseException:
            turn.metrics.terminal_delivered = False
            self._session_metrics.record_turn(turn.metrics)
            raise
        self._session_metrics.record_turn(turn.metrics)
        self._commit_history(turn)
        logger.info(
            "turn_terminal session_id=%s turn_id=%s outcome=%s",
            self.session_id,
            turn.turn_id,
            turn.outcome.value,
        )
        if self._current_turn_id == turn.turn_id:
            self._current_turn_id = None
        reservation = turn.lifecycle_reservation
        if reservation is not None:
            await self._outbound.ack(
                SendAction(action.sequence, event, reservation)
            )

    def _terminal_event(self, turn: TurnState) -> ServerEvent:
        common = {
            "event_id": self._event_id(),
            "session_id": self.session_id,
            "timestamp": self._clock.utc_now(),
            "turn_id": turn.turn_id,
            "metrics": turn.metrics.model_dump(mode="json"),
        }
        if turn.outcome is TurnOutcome.INTERRUPTED:
            return TurnInterrupted(reason=turn.reason or "interrupted", **common)
        if turn.outcome is TurnOutcome.FAILED:
            return TurnFailed(reason=turn.reason or "failed", **common)
        return TurnCompleted(**common)

    def _is_output_valid(self, turn_id: int) -> bool:
        turn = self._turns.get(turn_id)
        return turn is not None and turn.output_valid

    async def _monitor_retiring(self, task: asyncio.Task[None]) -> None:
        try:
            try:
                async with asyncio.timeout(
                    self._settings.cancellation_timeout_ms / 1_000
                ):
                    await asyncio.shield(task)
            except TimeoutError:
                self._session_metrics.timeout_count += 1
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except BaseException:
                pass
        finally:
            if task.done():
                self._retiring_turns.discard(task)

    def _supervisor_finished(
        self, task: asyncio.Task[None], turn: TurnState
    ) -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except BaseException:
            pass
        if turn.outcome is TurnOutcome.FAILED and turn.reason in {
            "backpressure_timeout",
            "lifecycle_capacity_timeout",
        }:
            self._session_metrics.overload_count += 1
            self._request_abort(
                1013,
                outcome=TurnOutcome.FAILED,
                reason="session_aborted",
            )

    def _request_abort(
        self,
        close_code: int,
        *,
        outcome: TurnOutcome = TurnOutcome.INTERRUPTED,
        reason: str = "session_ended",
    ) -> None:
        if self._abort_requested.is_set():
            return
        self._requested_close_code = close_code
        self._abort_outcome = outcome
        self._abort_reason = reason
        self._abort_requested.set()

    def _finalize_retained_turns(self) -> None:
        now_ns = self._clock.monotonic_ns()
        for turn in self._turns.values():
            if turn.outcome is TurnOutcome.OPEN:
                if self._abort_outcome is TurnOutcome.FAILED:
                    turn.fail(self._abort_reason)
                else:
                    turn.interrupt(self._abort_reason)
            if turn.metrics.total_latency_ms is None:
                turn.metrics.finalize(turn.outcome, turn.reason, now_ns)
            self._session_metrics.record_turn(turn.metrics)

    def _commit_history(self, turn: TurnState) -> None:
        if turn.history_committed:
            return
        assistant_text = " ".join(turn.response_tokens).strip()
        self._history.extend(
            (
                ChatMessage(role="user", content=turn.user_text),
                ChatMessage(role="assistant", content=assistant_text),
            )
        )
        turn.history_committed = True

    @staticmethod
    def _event_id() -> str:
        return uuid.uuid4().hex
