from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.api.events import (
    AudioEnd,
    AudioFrame,
    LLMToken,
    Ping,
    SessionStart,
    SessionStop,
    parse_client_event,
)


def event_payload(event_type: str, **payload: object) -> dict[str, object]:
    return {
        "type": event_type,
        "event_id": f"event-{event_type}",
        "session_id": "session-1",
        "timestamp": "2026-10-31T10:00:00Z",
        **payload,
    }


def test_settings_defaults_are_assessment_safe() -> None:
    settings = Settings.from_environment({})

    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.stt_partial_delay_ms == 20
    assert settings.silence_timeout_ms == 350
    assert settings.llm_token_delay_ms == 40
    assert settings.tts_sentence_delay_ms == 100
    assert settings.first_token_timeout_ms == 500
    assert settings.generation_timeout_ms == 5000
    assert settings.cancellation_timeout_ms == 250
    assert settings.backpressure_timeout_ms == 250
    assert settings.shutdown_grace_ms == 1000
    assert settings.max_message_bytes == 65_536
    assert settings.inbound_queue_size == 64
    assert settings.token_queue_size == 32
    assert settings.outbound_data_capacity == 128
    assert settings.outbound_lifecycle_capacity == 16
    assert settings.metrics_session_capacity == 200
    assert settings.metrics_turns_per_session == 200


def test_environment_overrides_defaults_with_typed_values() -> None:
    settings = Settings.from_environment(
        {
            "VOICE_HOST": "0.0.0.0",
            "VOICE_PORT": "9000",
            "VOICE_TOKEN_QUEUE_SIZE": "7",
            "UNRELATED": "ignored",
        }
    )

    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.token_queue_size == 7


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("VOICE_PORT", "0"),
        ("VOICE_SILENCE_TIMEOUT_MS", "-1"),
        ("VOICE_INBOUND_QUEUE_SIZE", "0"),
        ("VOICE_MAX_MESSAGE_BYTES", "0"),
    ],
)
def test_settings_reject_non_positive_bounds(name: str, value: str) -> None:
    with pytest.raises(ValidationError):
        Settings.from_environment({name: value})


def test_settings_reject_unknown_voice_variable() -> None:
    with pytest.raises(ValueError, match="VOICE_NOT_A_SETTING"):
        Settings.from_environment({"VOICE_NOT_A_SETTING": "1"})


@pytest.mark.parametrize(
    ("event_type", "extra", "expected_type"),
    [
        ("session.start", {}, SessionStart),
        ("audio.frame", {"text": "hello"}, AudioFrame),
        ("audio.end", {}, AudioEnd),
        ("session.stop", {}, SessionStop),
        ("ping", {}, Ping),
    ],
)
def test_parse_client_event_discriminates_variants(
    event_type: str,
    extra: dict[str, object],
    expected_type: type[object],
) -> None:
    event = parse_client_event(event_payload(event_type, **extra))

    assert isinstance(event, expected_type)
    assert event.event_id == f"event-{event_type}"
    assert event.session_id == "session-1"
    assert event.timestamp == datetime(2026, 10, 31, 10, 0, tzinfo=UTC)


@pytest.mark.parametrize("text", ["", "   "])
def test_audio_frame_rejects_empty_fragment(text: str) -> None:
    with pytest.raises(ValidationError):
        parse_client_event(event_payload("audio.frame", text=text))


def test_parse_client_event_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        parse_client_event(event_payload("audio.unknown"))


def test_event_rejects_naive_timestamp() -> None:
    payload = event_payload("ping")
    payload["timestamp"] = "2026-10-31T10:00:00"

    with pytest.raises(ValidationError):
        parse_client_event(payload)


def test_event_normalizes_aware_timestamp_to_utc() -> None:
    payload = event_payload("ping")
    payload["timestamp"] = datetime(
        2026, 10, 31, 15, 30, tzinfo=timezone(timedelta(hours=5, minutes=30))
    )

    event = parse_client_event(payload)

    assert event.timestamp == datetime(2026, 10, 31, 10, 0, tzinfo=UTC)
    assert event.timestamp.tzinfo is UTC


def test_turn_scoped_server_event_requires_integer_turn_id() -> None:
    token = LLMToken(
        type="llm.token",
        event_id="server-event-1",
        session_id="session-1",
        timestamp=datetime.now(UTC),
        turn_id=3,
        token="hello",
    )

    assert token.turn_id == 3
    with pytest.raises(ValidationError):
        LLMToken(
            type="llm.token",
            event_id="server-event-2",
            session_id="session-1",
            timestamp=datetime.now(UTC),
            turn_id="3",
            token="hello",
        )
