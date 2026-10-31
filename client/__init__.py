from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import websockets

from app.metrics import TurnMetrics


@dataclass(frozen=True, slots=True)
class DefaultScript:
    explicit_frames: tuple[str, ...]
    silence_frames: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BargeInScript:
    initial_frames: tuple[str, ...]
    interruption_frames: tuple[str, ...]
    trigger: str = "first_audio_chunk"


DEFAULT_SCRIPT = DefaultScript(
    explicit_frames=("Hello", "I need", "some help"),
    silence_frames=("Anything else",),
)
BARGE_IN_SCRIPT = BargeInScript(
    initial_frames=("Please explain my options",),
    interruption_frames=("I need to interrupt",),
)


@dataclass(frozen=True, slots=True)
class CallResult:
    session_id: str
    outcome: str
    turn_metrics: tuple[TurnMetrics, ...]
    interrupted_turns: int
    errors: tuple[str, ...]


EventSink = Callable[[dict[str, Any]], None]


def _open_websocket(url: str) -> Any:
    return websockets.connect(url)


async def _fetch_metrics(server_url: str, session_id: str) -> dict[str, Any]:
    parsed = urlsplit(server_url)
    http_scheme = "https" if parsed.scheme == "wss" else "http"
    metrics_url = urlunsplit(
        (http_scheme, parsed.netloc, "/metrics", f"session_id={session_id}", "")
    )
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(metrics_url)
        response.raise_for_status()
        return response.json()


async def run_call(
    server_url: str,
    scenario: str,
    session_id: str | None = None,
    on_event: EventSink | None = None,
) -> CallResult:
    if scenario not in {"default", "barge_in"}:
        raise ValueError(f"unknown scenario: {scenario}")
    selected_session_id = session_id or f"call-{uuid.uuid4().hex[:12]}"
    websocket_url = f"{server_url.rstrip('/')}/ws/{selected_session_id}"
    errors: list[str] = []
    terminal_outcome = "unknown"
    interrupted = 0

    async with _open_websocket(websocket_url) as websocket:
        await _send(websocket, "session.start", selected_session_id)
        await _receive_until(websocket, "session.ready", on_event, errors)

        if scenario == "default":
            for fragment in DEFAULT_SCRIPT.explicit_frames:
                await _send(
                    websocket,
                    "audio.frame",
                    selected_session_id,
                    text=fragment,
                )
            await _send(websocket, "audio.end", selected_session_id)
            first_terminal = await _receive_terminal(
                websocket, on_event, errors
            )
            terminal_outcome = _outcome(first_terminal)

            for fragment in DEFAULT_SCRIPT.silence_frames:
                await _send(
                    websocket,
                    "audio.frame",
                    selected_session_id,
                    text=fragment,
                )
            second_terminal = await _receive_terminal(
                websocket, on_event, errors
            )
            terminal_outcome = _outcome(second_terminal)
        else:
            for fragment in BARGE_IN_SCRIPT.initial_frames:
                await _send(
                    websocket,
                    "audio.frame",
                    selected_session_id,
                    text=fragment,
                )
            await _send(websocket, "audio.end", selected_session_id)
            first_audio = await _receive_until(
                websocket, "audio.chunk", on_event, errors
            )
            old_turn_id = int(first_audio["turn_id"])

            for fragment in BARGE_IN_SCRIPT.interruption_frames:
                await _send(
                    websocket,
                    "audio.frame",
                    selected_session_id,
                    text=fragment,
                )
            await _send(websocket, "audio.end", selected_session_id)

            interruption_seen = False
            while True:
                event = await _receive(websocket, on_event, errors)
                if (
                    interruption_seen
                    and event.get("type") == "audio.chunk"
                    and event.get("turn_id") == old_turn_id
                ):
                    raise RuntimeError(
                        "stale audio observed after turn.interrupted"
                    )
                if (
                    event.get("type") == "turn.interrupted"
                    and event.get("turn_id") == old_turn_id
                ):
                    interruption_seen = True
                    interrupted += 1
                if (
                    event.get("type") == "turn.completed"
                    and event.get("turn_id") != old_turn_id
                ):
                    terminal_outcome = "completed"
                    break

        await _send(websocket, "session.stop", selected_session_id)

    summary = await _fetch_metrics(server_url, selected_session_id)
    turn_metrics = tuple(
        TurnMetrics.model_validate({"baseline_ns": 0, **turn})
        for turn in summary.get("turns", [])
    )
    if scenario == "barge_in":
        interrupted = int(summary.get("counts", {}).get("interrupted", interrupted))
    return CallResult(
        session_id=selected_session_id,
        outcome=terminal_outcome,
        turn_metrics=turn_metrics,
        interrupted_turns=interrupted,
        errors=tuple(errors),
    )


async def _send(
    websocket: Any,
    event_type: str,
    session_id: str,
    **extra: Any,
) -> None:
    payload = {
        "type": event_type,
        "event_id": uuid.uuid4().hex,
        "session_id": session_id,
        "timestamp": datetime.now(UTC).isoformat(),
        **extra,
    }
    await websocket.send(json.dumps(payload))


async def _receive(
    websocket: Any,
    on_event: EventSink | None,
    errors: list[str],
) -> dict[str, Any]:
    event = json.loads(await websocket.recv())
    if on_event is not None:
        on_event(event)
    if event.get("type") == "error":
        errors.append(str(event.get("message", "unknown server error")))
    return event


async def _receive_until(
    websocket: Any,
    event_type: str,
    on_event: EventSink | None,
    errors: list[str],
) -> dict[str, Any]:
    while True:
        event = await _receive(websocket, on_event, errors)
        if event.get("type") == event_type:
            return event


async def _receive_terminal(
    websocket: Any,
    on_event: EventSink | None,
    errors: list[str],
) -> dict[str, Any]:
    while True:
        event = await _receive(websocket, on_event, errors)
        if event.get("type") in {
            "turn.completed",
            "turn.interrupted",
            "turn.failed",
        }:
            return event


def _outcome(event: dict[str, Any]) -> str:
    return str(event["type"]).removeprefix("turn.")


def render_event(event: dict[str, Any], *, verbose: bool) -> None:
    if verbose:
        print(json.dumps(event, sort_keys=True))
        return
    event_type = event.get("type")
    if event_type == "transcript.partial":
        print(f"caller (partial): {event.get('text')}")
    elif event_type == "transcript.final":
        print(f"caller (final): {event.get('text')}")
    elif event_type == "llm.token":
        print(f"LLM token: {event.get('token')}")
    elif event_type == "audio.chunk":
        print(f"bot audio [{event.get('sequence')}]: {event.get('text')}")
    elif isinstance(event_type, str) and event_type.startswith("turn."):
        metrics = event.get("metrics", {})
        print(
            f"turn {event.get('turn_id')} {event_type.removeprefix('turn.')} | "
            f"TTFT={_metric(metrics.get('ttft_ms'))} "
            f"TTFA={_metric(metrics.get('ttfa_ms'))} "
            f"total={_metric(metrics.get('total_latency_ms'))}"
        )
    elif event_type == "error":
        print(f"server error: {event.get('message')}")


def render_summary(result: CallResult) -> None:
    print(
        f"server summary: session={result.session_id} outcome={result.outcome} "
        f"turns={len(result.turn_metrics)} interrupted={result.interrupted_turns}"
    )
    for metrics in result.turn_metrics:
        print(
            f"  turn {metrics.turn_id} {metrics.outcome.value}: "
            f"TTFT={_metric(metrics.ttft_ms)} "
            f"TTFA={_metric(metrics.ttfa_ms)} "
            f"total={_metric(metrics.total_latency_ms)}"
        )


def _metric(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.1f}ms"
