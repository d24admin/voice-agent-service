from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from app.clock import Clock
from app.config import Settings
from app.events import (
    AudioChunk,
    AudioEnd,
    AudioFrame,
    ClientEvent,
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
from app.outbound import (
    DropAction,
    OutboundBuffer,
    OutboundClosedError,
    OutputCompleteAction,
    SendAction,
    TerminalCommand,
)
from app.providers import ChatMessage, LLMProvider
from app.stt import FinalResult, SimulatedSTT
from app.transport import EventTransport
from app.turn import ActiveTurnRunner, TurnState


@dataclass(frozen=True, slots=True)
class InboundEnvelope:
    event: ClientEvent
    received_ns: int


@dataclass(frozen=True, slots=True)
class SilenceExpired:
    timer_generation: int


InboundMessage = InboundEnvelope | SilenceExpired


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
        self._turns: dict[int, TurnState] = {}
        self._next_turn_id = 1

    @property
    def turns(self) -> dict[int, TurnState]:
        return dict(self._turns)

    async def run(self) -> None:
        prefix = f"voice-session-{self.session_id}:"
        receiver = asyncio.create_task(
            self._receive_loop(), name=f"{prefix}receiver"
        )
        writer = asyncio.create_task(self._write_loop(), name=f"{prefix}writer")
        reaper = asyncio.create_task(
            self._turn_reaper(), name=f"{prefix}turn-reaper"
        )
        orchestrator = asyncio.create_task(
            self._orchestrate(), name=f"{prefix}orchestrator"
        )
        close_code = 1000
        try:
            await orchestrator
        except asyncio.CancelledError:
            close_code = 1001
            raise
        except Exception:
            close_code = 1011
            raise
        finally:
            await self._cancel_silence_timer()
            owned = [receiver, writer, reaper, orchestrator]
            if self._active_turn_task is not None:
                owned.append(self._active_turn_task)
            for task in owned:
                if not task.done():
                    task.cancel()
            await self._outbound.close()
            await asyncio.gather(*owned, return_exceptions=True)
            await self._transport.close(close_code)

    async def _receive_loop(self) -> None:
        while True:
            event = await self._transport.receive()
            envelope = InboundEnvelope(
                event=event,
                received_ns=self._clock.monotonic_ns(),
            )
            await self._inbound.put(envelope)
            if isinstance(event, SessionStop):
                return

    async def _orchestrate(self) -> None:
        while True:
            message = await self._inbound.get()
            if isinstance(message, SilenceExpired):
                final = self._stt.finalize(message.timer_generation)
                if final is not None:
                    await self._start_turn(final)
                continue

            event = message.event
            if isinstance(event, SessionStop):
                return
            if isinstance(event, SessionStart):
                await self._outbound.put_data(
                    SessionReady(
                        event_id=self._event_id(),
                        session_id=self.session_id,
                        timestamp=self._clock.utc_now(),
                    )
                )
                continue
            if isinstance(event, AudioFrame):
                await self._handle_audio_frame(event, message.received_ns)
                continue
            if isinstance(event, AudioEnd):
                await self._cancel_silence_timer()
                final = self._stt.finalize()
                if final is not None:
                    await self._start_turn(final)
                continue
            if event.type == "ping":
                await self._outbound.put_data(
                    Pong(
                        event_id=self._event_id(),
                        session_id=self.session_id,
                        timestamp=self._clock.utc_now(),
                    )
                )

    async def _handle_audio_frame(
        self, event: AudioFrame, received_ns: int
    ) -> None:
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
        await self._outbound.put_data(
            TranscriptPartial(
                event_id=self._event_id(),
                session_id=self.session_id,
                timestamp=self._clock.utc_now(),
                utterance_id=partial.utterance_id,
                text=partial.text,
            )
        )

    async def _silence_after(self, timer_generation: int) -> None:
        await asyncio.sleep(self._settings.silence_timeout_ms / 1_000)
        await self._inbound.put(SilenceExpired(timer_generation))

    async def _cancel_silence_timer(self) -> None:
        timer = self._silence_timer
        self._silence_timer = None
        if timer is None or timer.done():
            if timer is not None:
                await asyncio.gather(timer, return_exceptions=True)
            return
        timer.cancel()
        await asyncio.gather(timer, return_exceptions=True)

    async def _start_turn(self, final: FinalResult) -> None:
        await self._outbound.put_data(
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
            metrics=TurnMetrics(turn_id=turn_id, baseline_ns=final.last_frame_ns),
        )
        self._turns[turn_id] = turn
        self._session_metrics.mark_started(turn_id)
        runner = ActiveTurnRunner(
            provider=self._provider,
            outbound=self._outbound,
            settings=self._settings,
            session_id=self.session_id,
            clock=self._clock,
        )
        self._active_turn_task = asyncio.create_task(
            runner.run(turn, (ChatMessage(role="user", content=final.text),)),
            name=f"voice-session-{self.session_id}:turn-{turn_id}",
        )

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
            event = self._terminal_event(turn)
            await self._transport.send(event)
            self._session_metrics.record_turn(turn.metrics)
            await self._outbound.ack(action)
            return

        await self._transport.send(item)
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
        event = TurnCompleted(
            event_id=self._event_id(),
            session_id=self.session_id,
            timestamp=self._clock.utc_now(),
            turn_id=turn.turn_id,
            metrics=turn.metrics.model_dump(mode="json"),
        )
        await self._transport.send(event)
        self._session_metrics.record_turn(turn.metrics)
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

    async def _turn_reaper(self) -> None:
        await asyncio.Event().wait()

    @staticmethod
    def _event_id() -> str:
        return uuid.uuid4().hex
