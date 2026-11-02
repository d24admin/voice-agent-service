from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any

import pytest

import client
from app.metrics import TurnMetrics, TurnOutcome
from app.scenario import load_scenario
from client import CallResult, ConversationTurn, ReadableTraceRenderer, run_call
from client.__main__ import main


FIXTURE = Path(__file__).parents[1] / "scenarios" / "ecommerce.json"


def event(event_type: str, **extra: Any) -> dict[str, Any]:
    return {
        "type": event_type,
        "event_id": f"server-{event_type}",
        "session_id": "demo-1",
        "timestamp": "2026-11-02T12:00:00+00:00",
        **extra,
    }


def terminal(turn_id: int, outcome: str = "completed") -> dict[str, Any]:
    reason = "barge_in" if outcome == "interrupted" else None
    return event(
        f"turn.{outcome}",
        turn_id=turn_id,
        reason=reason,
        metrics={
            "turn_id": turn_id,
            "outcome": outcome,
            "reason": reason,
            "ttft_ms": 20.0,
            "ttfa_ms": 40.0,
            "total_latency_ms": 80.0,
            "token_count": 2,
            "audio_chunk_count": 1,
        },
    )


class FakeSocket:
    def __init__(
        self,
        incoming: list[dict[str, Any]],
        *,
        hold_interruption_until_second_turn: bool = False,
    ) -> None:
        self.incoming = list(incoming)
        self.sent: list[dict[str, Any]] = []
        self.hold_interruption = hold_interruption_until_second_turn
        self.second_turn_sent = asyncio.Event()

    async def send(self, raw: str) -> None:
        payload = json.loads(raw)
        self.sent.append(payload)
        if payload.get("text", "").startswith("Actually"):
            self.second_turn_sent.set()

    async def recv(self) -> str:
        if not self.incoming:
            raise RuntimeError("fake socket exhausted")
        item = self.incoming[0]
        if (
            self.hold_interruption
            and item["type"] == "turn.interrupted"
            and not self.second_turn_sent.is_set()
        ):
            await self.second_turn_sent.wait()
        return json.dumps(self.incoming.pop(0))


class FakeConnection:
    def __init__(self, socket: FakeSocket) -> None:
        self.socket = socket

    async def __aenter__(self) -> FakeSocket:
        return self.socket

    async def __aexit__(self, *_args: object) -> None:
        return None


def metrics_summary(*turns: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": "demo-1",
        "counts": {
            "started": len(turns),
            "completed": sum(turn["outcome"] == "completed" for turn in turns),
            "interrupted": sum(turn["outcome"] == "interrupted" for turn in turns),
            "failed": sum(turn["outcome"] == "failed" for turn in turns),
        },
        "ttft": {"sample_count": 1, "p50": 20.0, "p95": 20.0},
        "ttfa": {"sample_count": 1, "p50": 40.0, "p95": 40.0},
        "total_latency": {"sample_count": 1, "p50": 80.0, "p95": 80.0},
        "turns": list(turns),
    }


async def immediate_sleep(_delay: float) -> None:
    await asyncio.sleep(0)


async def never_sleep(_delay: float) -> None:
    await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_scenario_runner_executes_configured_barge_in_and_builds_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_scenario(FIXTURE)
    interruption = terminal(1, "interrupted")
    completion = terminal(2)
    incoming = [
        event("session.ready"),
        event("transcript.partial", utterance_id=1, text="damaged order"),
        event("transcript.final", utterance_id=1, text="damaged order"),
        event("llm.token", turn_id=1, token="Sorry"),
        event("audio.chunk", turn_id=1, sequence=0, text="I am sorry."),
        event("llm.token", turn_id=1, token="More"),
        interruption,
        event("transcript.partial", utterance_id=2, text="refund"),
        event("transcript.final", utterance_id=2, text="refund instead"),
        event("llm.token", turn_id=2, token="Understood"),
        event("audio.chunk", turn_id=2, sequence=0, text="Understood."),
        completion,
    ]
    socket = FakeSocket(incoming, hold_interruption_until_second_turn=True)
    observed: list[dict[str, Any]] = []
    opened_urls: list[str] = []

    def open_websocket(url: str) -> FakeConnection:
        opened_urls.append(url)
        return FakeConnection(socket)

    monkeypatch.setattr(client, "_open_websocket", open_websocket)
    monkeypatch.setattr(
        client,
        "_fetch_metrics",
        lambda *_args: async_value(
            metrics_summary(interruption["metrics"], completion["metrics"])
        ),
    )

    result = await run_call(
        "ws://localhost:8000",
        scenario,
        "demo-1",
        on_event=observed.append,
        sleep=immediate_sleep,
    )

    assert result.outcome == "completed"
    assert result.interrupted_turns == 1
    assert [turn.outcome for turn in result.conversation] == [
        "interrupted",
        "completed",
    ]
    assert result.conversation[0].caller_transcript == "damaged order"
    assert result.conversation[0].generated_text == "Sorry More"
    assert result.conversation[0].partial is True
    assert result.conversation[1].spoken_chunks == ("Understood.",)
    assert any(item["type"] == "client.barge_in" for item in observed)
    assert opened_urls == [
        "ws://localhost:8000/ws/demo-1?scenario_id=ecommerce"
    ]
    sent_text = [message.get("text") for message in socket.sent]
    assert sent_text.index("Actually, I no longer want a replacement.") > 0
    assert [message["type"] for message in socket.sent].count("audio.end") == 2


@pytest.mark.asyncio
async def test_configured_barge_in_fails_if_old_turn_finishes_before_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_scenario(FIXTURE)
    socket = FakeSocket(
        [
            event("session.ready"),
            event("transcript.final", utterance_id=1, text="damaged order"),
            event("audio.chunk", turn_id=1, sequence=0, text="First sentence."),
            terminal(1),
        ]
    )
    monkeypatch.setattr(client, "_open_websocket", lambda _url: FakeConnection(socket))

    with pytest.raises(RuntimeError, match="scenario did not barge in"):
        await run_call(
            "ws://localhost:8000",
            scenario,
            "demo-1",
            sleep=never_sleep,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_type", ["llm.token", "audio.chunk"])
async def test_runner_rejects_stale_stream_output_after_interruption(
    monkeypatch: pytest.MonkeyPatch,
    stale_type: str,
) -> None:
    scenario = load_scenario(FIXTURE)
    stale_payload = {"token": "stale"} if stale_type == "llm.token" else {
        "sequence": 1,
        "text": "Stale audio.",
    }
    socket = FakeSocket(
        [
            event("session.ready"),
            event("transcript.final", utterance_id=1, text="damaged order"),
            event("audio.chunk", turn_id=1, sequence=0, text="First sentence."),
            terminal(1, "interrupted"),
            event(stale_type, turn_id=1, **stale_payload),
        ],
        hold_interruption_until_second_turn=True,
    )
    monkeypatch.setattr(client, "_open_websocket", lambda _url: FakeConnection(socket))

    with pytest.raises(RuntimeError, match="stale .* after turn.interrupted"):
        await run_call(
            "ws://localhost:8000",
            scenario,
            "demo-1",
            sleep=immediate_sleep,
        )


def test_readable_renderer_preserves_interleaving_without_one_line_per_token() -> None:
    stream = io.StringIO()
    renderer = ReadableTraceRenderer(
        scenario_id="ecommerce",
        bot_name="ShopHelp",
        stream=stream,
        color=False,
    )

    renderer.render_event(event("session.ready"))
    renderer.render_event(
        event("transcript.partial", utterance_id=1, text="refund")
    )
    renderer.render_event(
        event("transcript.final", utterance_id=1, text="refund please")
    )
    renderer.render_event(event("llm.token", turn_id=1, token="I"))
    renderer.render_event(event("llm.token", turn_id=1, token="can"))
    renderer.render_event(
        event("audio.chunk", turn_id=1, sequence=0, text="I can help.")
    )
    renderer.render_event(event("llm.token", turn_id=1, token="help"))
    renderer.render_event(
        {"type": "client.barge_in", "turn_id": 1, "delay_ms": 250}
    )
    renderer.render_event(terminal(1, "interrupted"))

    output = stream.getvalue()
    assert "SHOPHELP · ecommerce · session demo-1" in output
    assert "TURN 1" in output
    assert "STT …   refund" in output
    assert "STT ✓   refund please" in output
    assert "LLM     I can \nTTS ▶   “I can help.”\nLLM     help \n" in output
    assert "LLM token:" not in output
    assert "⚡ BARGE-IN · turn 2 started 250 ms after first audio" in output
    assert "RESULT  INTERRUPTED · barge_in" in output
    assert "LATENCY TTFT 20 ms · TTFA 40 ms · total 80 ms" in output


def test_readable_summary_labels_partial_output_and_counts_outcomes() -> None:
    stream = io.StringIO()
    renderer = ReadableTraceRenderer(
        scenario_id="ecommerce",
        bot_name="ShopHelp",
        stream=stream,
        color=False,
    )
    metrics = TurnMetrics(turn_id=1, baseline_ns=0)
    metrics.ttft_ms = 20
    metrics.ttfa_ms = 40
    metrics.finalize(TurnOutcome.INTERRUPTED, "barge_in", 80_000_000)
    renderer.render_summary(
        CallResult(
            session_id="demo-1",
            outcome="interrupted",
            turn_metrics=(metrics,),
            interrupted_turns=1,
            errors=(),
            conversation=(
                ConversationTurn(
                    turn_id=1,
                    caller_transcript="refund please",
                    tokens=("I", "can", "help"),
                    spoken_chunks=("I can help.",),
                    outcome="interrupted",
                    reason="barge_in",
                    metrics=metrics.model_dump(mode="json"),
                ),
                ConversationTurn(
                    turn_id=2,
                    caller_transcript="book instead",
                    tokens=("Done.",),
                    spoken_chunks=("Done.",),
                    outcome="completed",
                    reason=None,
                    metrics={
                        "ttft_ms": 20,
                        "ttfa_ms": 40,
                        "total_latency_ms": 1515.9,
                    },
                ),
            ),
        )
    )

    output = stream.getvalue()
    for expected in (
        "CONVERSATION · session demo-1",
        "Turn 1 · INTERRUPTED · barge_in",
        "Caller       refund please",
        "Bot partial  I can help",
        "1. I can help.",
        "total 1.52 s",
        "SUMMARY  2 turns · 1 completed · 1 interrupted · 0 failed",
    ):
        assert expected in output


def test_readable_renderer_uses_color_only_for_tty_without_no_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TTYBuffer(io.StringIO):
        def isatty(self) -> bool:
            return True

    colored = TTYBuffer()
    monkeypatch.delenv("NO_COLOR", raising=False)
    ReadableTraceRenderer(
        scenario_id="travel", bot_name="TripAssist", stream=colored
    ).render_event(event("session.ready"))
    assert "\x1b[" in colored.getvalue()

    plain = TTYBuffer()
    monkeypatch.setenv("NO_COLOR", "1")
    ReadableTraceRenderer(
        scenario_id="travel", bot_name="TripAssist", stream=plain
    ).render_event(event("session.ready"))
    assert "\x1b[" not in plain.getvalue()


def test_cli_reports_scenario_errors_without_traceback(
    tmp_path: Path,
    capsys: Any,
) -> None:
    missing = tmp_path / "missing.json"
    assert main(["--scenario-file", str(missing)]) == 2
    error = capsys.readouterr().err
    assert "scenario file not found" in error
    assert "Traceback" not in error


async def async_value(value: Any) -> Any:
    return value
