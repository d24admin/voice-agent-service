from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TextIO
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx
import websockets

from app.metrics import TurnMetrics, TurnOutcome
from app.scenario import InterviewScenario


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    turn_id: int
    caller_transcript: str
    tokens: tuple[str, ...]
    spoken_chunks: tuple[str, ...]
    outcome: str
    reason: str | None
    metrics: dict[str, Any]

    @property
    def generated_text(self) -> str:
        return " ".join(self.tokens)

    @property
    def partial(self) -> bool:
        return self.outcome == "interrupted"


@dataclass(frozen=True, slots=True)
class CallResult:
    session_id: str
    outcome: str
    turn_metrics: tuple[TurnMetrics, ...]
    interrupted_turns: int
    errors: tuple[str, ...]
    conversation: tuple[ConversationTurn, ...] = ()


@dataclass(slots=True)
class _ObservedTurn:
    turn_id: int
    caller_transcript: str = ""
    tokens: list[str] = field(default_factory=list)
    spoken_chunks: list[str] = field(default_factory=list)
    outcome: str = "open"
    reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def freeze(self) -> ConversationTurn:
        return ConversationTurn(
            turn_id=self.turn_id,
            caller_transcript=self.caller_transcript,
            tokens=tuple(self.tokens),
            spoken_chunks=tuple(self.spoken_chunks),
            outcome=self.outcome,
            reason=self.reason,
            metrics=dict(self.metrics),
        )


EventSink = Callable[[dict[str, Any]], None]
Sleep = Callable[[float], Awaitable[None]]
TERMINAL_TYPES = {"turn.completed", "turn.interrupted", "turn.failed"}
STREAM_TYPES = {"llm.token", "audio.chunk"}


class _CallObserver:
    def __init__(self, sink: EventSink | None) -> None:
        self._sink = sink
        self._pending_transcripts: deque[str] = deque()
        self._turns: dict[int, _ObservedTurn] = {}
        self.interrupted_turn_ids: set[int] = set()
        self.errors: list[str] = []

    def record(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        turn_id_value = event.get("turn_id")
        turn_id = int(turn_id_value) if turn_id_value is not None else None

        if event_type in STREAM_TYPES and turn_id in self.interrupted_turn_ids:
            stream_name = "token" if event_type == "llm.token" else "audio"
            raise RuntimeError(
                f"stale {stream_name} observed after turn.interrupted"
            )
        if event_type == "transcript.final":
            self._pending_transcripts.append(str(event.get("text", "")))
        if event_type == "error":
            self.errors.append(str(event.get("message", "unknown server error")))

        if turn_id is not None:
            turn = self._turns.get(turn_id)
            if turn is None:
                transcript = self._pending_transcripts.popleft() if self._pending_transcripts else ""
                turn = _ObservedTurn(turn_id, caller_transcript=transcript)
                self._turns[turn_id] = turn
            if event_type == "llm.token":
                turn.tokens.append(str(event.get("token", "")))
            elif event_type == "audio.chunk":
                turn.spoken_chunks.append(str(event.get("text", "")))
            elif event_type in TERMINAL_TYPES:
                turn.outcome = event_type.removeprefix("turn.")
                reason = event.get("reason")
                turn.reason = None if reason is None else str(reason)
                turn.metrics = dict(event.get("metrics", {}))
                if event_type == "turn.interrupted":
                    self.interrupted_turn_ids.add(turn_id)

        if self._sink is not None:
            self._sink(event)

    def conversation(self) -> tuple[ConversationTurn, ...]:
        return tuple(turn.freeze() for turn in self._turns.values())


def _open_websocket(url: str) -> Any:
    return websockets.connect(url)


async def _fetch_metrics(server_url: str, session_id: str) -> dict[str, Any]:
    parsed = urlsplit(server_url)
    http_scheme = "https" if parsed.scheme == "wss" else "http"
    metrics_url = urlunsplit(
        (http_scheme, parsed.netloc, "/metrics", f"session_id={session_id}", "")
    )
    async with httpx.AsyncClient(timeout=5) as http:
        response = await http.get(metrics_url)
        response.raise_for_status()
        return response.json()


async def run_call(
    server_url: str,
    scenario: InterviewScenario,
    session_id: str | None = None,
    on_event: EventSink | None = None,
    *,
    sleep: Sleep = asyncio.sleep,
) -> CallResult:
    selected_session_id = session_id or f"call-{uuid.uuid4().hex[:12]}"
    query = urlencode({"scenario_id": scenario.id})
    websocket_url = (
        f"{server_url.rstrip('/')}/ws/{selected_session_id}?{query}"
    )
    observer = _CallObserver(on_event)
    response_ids: set[int] = set()
    final_outcome = "unknown"

    async with _open_websocket(websocket_url) as websocket:
        await _send(websocket, "session.start", selected_session_id)
        await _receive_until_type(websocket, "session.ready", observer)

        for index, caller_turn in enumerate(scenario.caller.turns):
            if index > 0:
                delay_ms = caller_turn.barge_in_after_ms
                if delay_ms is None:
                    previous = await _receive_new_terminal(
                        websocket, observer, response_ids
                    )
                    response_ids.add(int(previous["turn_id"]))
                else:
                    first_audio = await _receive_new_audio(
                        websocket, observer, response_ids
                    )
                    old_turn_id = int(first_audio["turn_id"])
                    response_ids.add(old_turn_id)
                    await _wait_barge_in_delay(
                        websocket,
                        observer,
                        old_turn_id,
                        delay_ms,
                        sleep,
                    )
                    observer.record(
                        {
                            "type": "client.barge_in",
                            "turn_id": old_turn_id,
                            "delay_ms": delay_ms,
                        }
                    )

            for frame in caller_turn.frames:
                await _send(
                    websocket,
                    "audio.frame",
                    selected_session_id,
                    text=frame,
                )
            if caller_turn.end == "explicit":
                await _send(websocket, "audio.end", selected_session_id)

        final_terminal = await _receive_new_terminal(
            websocket, observer, response_ids
        )
        final_outcome = str(final_terminal["type"]).removeprefix("turn.")
        await _send(websocket, "session.stop", selected_session_id)

    summary = await _fetch_metrics(server_url, selected_session_id)
    turn_metrics = tuple(
        TurnMetrics.model_validate({"baseline_ns": 0, **turn})
        for turn in summary.get("turns", [])
    )
    if any(metrics.outcome is TurnOutcome.FAILED for metrics in turn_metrics):
        final_outcome = "failed"
    interrupted = int(
        summary.get("counts", {}).get(
            "interrupted", len(observer.interrupted_turn_ids)
        )
    )
    return CallResult(
        session_id=selected_session_id,
        outcome=final_outcome,
        turn_metrics=turn_metrics,
        interrupted_turns=interrupted,
        errors=tuple(observer.errors),
        conversation=observer.conversation(),
    )


async def _wait_barge_in_delay(
    websocket: Any,
    observer: _CallObserver,
    old_turn_id: int,
    delay_ms: int,
    sleep: Sleep,
) -> None:
    timer = asyncio.create_task(sleep(delay_ms / 1_000))
    try:
        while not timer.done():
            receiver = asyncio.create_task(_receive(websocket, observer))
            done, _ = await asyncio.wait(
                {timer, receiver}, return_when=asyncio.FIRST_COMPLETED
            )
            if receiver in done:
                event = receiver.result()
                if (
                    event.get("type") in TERMINAL_TYPES
                    and int(event.get("turn_id", -1)) == old_turn_id
                ):
                    raise RuntimeError(
                        "scenario did not barge in: the previous turn completed "
                        "before the configured delay"
                    )
                continue
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
    finally:
        if not timer.done():
            timer.cancel()
        await asyncio.gather(timer, return_exceptions=True)


async def _receive_new_audio(
    websocket: Any,
    observer: _CallObserver,
    excluded_turn_ids: set[int],
) -> dict[str, Any]:
    while True:
        event = await _receive(websocket, observer)
        if event.get("type") == "audio.chunk" and int(event["turn_id"]) not in excluded_turn_ids:
            return event
        if event.get("type") in TERMINAL_TYPES and int(event["turn_id"]) not in excluded_turn_ids:
            raise RuntimeError(
                "scenario did not barge in: the previous turn completed "
                "before producing audio"
            )


async def _receive_new_terminal(
    websocket: Any,
    observer: _CallObserver,
    excluded_turn_ids: set[int],
) -> dict[str, Any]:
    while True:
        event = await _receive(websocket, observer)
        if (
            event.get("type") in TERMINAL_TYPES
            and int(event["turn_id"]) not in excluded_turn_ids
        ):
            return event


async def _send(
    websocket: Any,
    event_type: str,
    session_id: str,
    **extra: Any,
) -> None:
    await websocket.send(
        json.dumps(
            {
                "type": event_type,
                "event_id": uuid.uuid4().hex,
                "session_id": session_id,
                "timestamp": datetime.now(UTC).isoformat(),
                **extra,
            }
        )
    )


async def _receive(
    websocket: Any,
    observer: _CallObserver,
) -> dict[str, Any]:
    event = json.loads(await websocket.recv())
    observer.record(event)
    return event


async def _receive_until_type(
    websocket: Any,
    event_type: str,
    observer: _CallObserver,
) -> dict[str, Any]:
    while True:
        event = await _receive(websocket, observer)
        if event.get("type") == event_type:
            return event


class ReadableTraceRenderer:
    _RESET = "\033[0m"
    _COLORS = {
        "heading": "\033[1;36m",
        "partial": "\033[2m",
        "final": "\033[1;32m",
        "tts": "\033[1;34m",
        "warning": "\033[1;33m",
        "success": "\033[1;32m",
        "failure": "\033[1;31m",
    }

    def __init__(
        self,
        *,
        scenario_id: str,
        bot_name: str,
        stream: TextIO = sys.stdout,
        color: bool | None = None,
    ) -> None:
        self._scenario_id = scenario_id
        self._bot_name = bot_name
        self._stream = stream
        self._color = (
            bool(getattr(stream, "isatty", lambda: False)())
            and "NO_COLOR" not in os.environ
            if color is None
            else color
        )
        self._llm_line_open = False
        self._seen_turns: set[int] = set()
        self._header_rendered = False

    def render_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if event_type == "llm.token":
            self._ensure_turn(_event_turn_id(event))
            if not self._llm_line_open:
                self._write(self._paint("LLM     ", "heading"))
                self._llm_line_open = True
            self._write(f"{event.get('token', '')} ")
            self._flush()
            return

        self.close_open_line()
        if event_type == "session.ready":
            self._render_header(str(event.get("session_id", "unknown")))
        elif event_type in {"transcript.partial", "transcript.final"}:
            self._ensure_turn(_event_turn_id(event))
            if event_type == "transcript.partial":
                label = self._paint("STT …   ", "partial")
            else:
                label = self._paint("STT ✓   ", "final")
            self._line(f"{label}{event.get('text', '')}")
        elif event_type == "audio.chunk":
            self._ensure_turn(_event_turn_id(event))
            label = self._paint("TTS ▶   ", "tts")
            self._line(f"{label}“{event.get('text', '')}”")
        elif event_type == "client.barge_in":
            old_turn = int(event.get("turn_id", 0))
            delay_ms = event.get("delay_ms", 0)
            label = self._paint("⚡ BARGE-IN", "warning")
            self._line(
                f"\n{label} · turn {old_turn + 1} started "
                f"{delay_ms} ms after first audio"
            )
        elif event_type in TERMINAL_TYPES:
            self._ensure_turn(_event_turn_id(event))
            outcome = event_type.removeprefix("turn.")
            reason = event.get("reason")
            style = "success" if outcome == "completed" else "warning"
            if outcome == "failed":
                style = "failure"
            suffix = "" if reason is None else f" · {reason}"
            self._line(
                f"{self._paint('RESULT  ', style)}{outcome.upper()}{suffix}"
            )
            self._line(_latency_line(dict(event.get("metrics", {}))))
        elif event_type == "error":
            label = self._paint("ERROR   ", "failure")
            self._line(f"{label}{event.get('message', 'unknown server error')}")

    def render_summary(self, result: CallResult) -> None:
        self.close_open_line()
        self._line(f"\n{self._paint('CONVERSATION', 'heading')} · session {result.session_id}")
        for turn in result.conversation:
            reason = "" if turn.reason is None else f" · {turn.reason}"
            self._line(f"\nTurn {turn.turn_id} · {turn.outcome.upper()}{reason}")
            self._line(f"  Caller       {turn.caller_transcript or '(no final transcript)'}")
            bot_label = "Bot partial" if turn.partial else "Bot"
            self._line(
                f"  {bot_label:<12} {turn.generated_text or '(no tokens observed)'}"
            )
            self._line("  Spoken")
            if turn.spoken_chunks:
                for index, chunk in enumerate(turn.spoken_chunks, start=1):
                    self._line(f"    {index}. {chunk}")
            else:
                self._line("    (no speech observed)")
            self._line(f"  {_latency_line(turn.metrics)}")

        counts = {
            outcome: sum(
                turn.outcome == outcome for turn in result.conversation
            )
            for outcome in ("completed", "interrupted", "failed")
        }
        self._line(
            f"\n{self._paint('SUMMARY', 'heading')}  "
            f"{len(result.conversation)} turns · "
            f"{counts['completed']} completed · "
            f"{counts['interrupted']} interrupted · "
            f"{counts['failed']} failed"
        )

    def close_open_line(self) -> None:
        if self._llm_line_open:
            self._write("\n")
            self._flush()
            self._llm_line_open = False

    def _render_header(self, session_id: str) -> None:
        if self._header_rendered:
            return
        title = f"{self._bot_name.upper()} · {self._scenario_id} · session {session_id}"
        self._line(self._paint(title, "heading"))
        self._line("─" * min(72, max(48, len(title))))
        self._header_rendered = True

    def _ensure_turn(self, turn_id: int | None) -> None:
        if turn_id is None or turn_id in self._seen_turns:
            return
        self.close_open_line()
        self._line(f"\n{self._paint(f'TURN {turn_id}', 'heading')}")
        self._seen_turns.add(turn_id)

    def _paint(self, text: str, style: str) -> str:
        if not self._color:
            return text
        return f"{self._COLORS[style]}{text}{self._RESET}"

    def _line(self, text: str = "") -> None:
        self._write(f"{text}\n")
        self._flush()

    def _write(self, text: str) -> None:
        self._stream.write(text)

    def _flush(self) -> None:
        self._stream.flush()


def render_event(event: dict[str, Any], *, verbose: bool) -> None:
    if verbose:
        print(json.dumps(event, sort_keys=True))
        return
    ReadableTraceRenderer(
        scenario_id="scenario",
        bot_name="Voice Agent",
        color=False,
    ).render_event(event)


def render_summary(result: CallResult) -> None:
    ReadableTraceRenderer(
        scenario_id="scenario",
        bot_name="Voice Agent",
        color=False,
    ).render_summary(result)


def _event_turn_id(event: dict[str, Any]) -> int | None:
    raw = event.get("turn_id", event.get("utterance_id"))
    return None if raw is None else int(raw)


def _latency_line(metrics: dict[str, Any]) -> str:
    return (
        f"LATENCY TTFT {_metric(metrics.get('ttft_ms'))} · "
        f"TTFA {_metric(metrics.get('ttfa_ms'))} · "
        f"total {_metric(metrics.get('total_latency_ms'))}"
    )


def _metric(value: object) -> str:
    if value is None:
        return "n/a"
    milliseconds = float(value)
    if milliseconds >= 1_000:
        return f"{milliseconds / 1_000:.2f} s"
    rendered = f"{milliseconds:.1f}".rstrip("0").rstrip(".")
    return f"{rendered} ms"
