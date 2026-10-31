from datetime import UTC, datetime

import pytest

from app.events import AudioChunk, LLMToken, Ping, TranscriptFinal
from app.metrics import TurnOutcome
from app.outbound import (
    DropAction,
    OutboundBuffer,
    SendAction,
    TerminalCommand,
)


NOW = datetime(2026, 10, 31, tzinfo=UTC)


def transcript() -> TranscriptFinal:
    return TranscriptFinal(
        event_id="transcript-1",
        session_id="session-1",
        timestamp=NOW,
        utterance_id=1,
        text="hello",
    )


def token() -> LLMToken:
    return LLMToken(
        event_id="token-1",
        session_id="session-1",
        timestamp=NOW,
        turn_id=1,
        token="old",
    )


def audio() -> AudioChunk:
    return AudioChunk(
        event_id="audio-1",
        session_id="session-1",
        timestamp=NOW,
        turn_id=1,
        text="old sentence.",
        sequence=0,
    )


def ping() -> Ping:
    return Ping(
        event_id="ping-1",
        session_id="session-1",
        timestamp=NOW,
    )


@pytest.mark.asyncio
async def test_sequence_barrier_preserves_control_order_and_drops_stale_stream() -> None:
    buffer = OutboundBuffer(data_capacity=8, lifecycle_capacity=2)
    reservation = await buffer.reserve_turn(1, timeout_s=0.1)
    await buffer.put_data(transcript())
    await buffer.put_data(token(), turn_id=1, fenced=True)
    await buffer.put_data(audio(), turn_id=1, fenced=True)
    buffer.put_lifecycle_nowait(
        reservation,
        TerminalCommand(1, TurnOutcome.INTERRUPTED, "barge_in"),
    )
    await buffer.put_data(ping())

    first = await buffer.next_dispatch(lambda turn_id: turn_id != 1)
    second = await buffer.next_dispatch(lambda turn_id: turn_id != 1)
    third = await buffer.next_dispatch(lambda turn_id: turn_id != 1)
    fourth = await buffer.next_dispatch(lambda turn_id: turn_id != 1)
    fifth = await buffer.next_dispatch(lambda turn_id: turn_id != 1)

    assert isinstance(first, SendAction)
    assert first.item == transcript()
    assert isinstance(second, DropAction)
    assert second.item == token()
    assert isinstance(third, DropAction)
    assert third.item == audio()
    assert isinstance(fourth, SendAction)
    assert fourth.item == TerminalCommand(1, TurnOutcome.INTERRUPTED, "barge_in")
    assert isinstance(fifth, SendAction)
    assert fifth.item == ping()
    await buffer.ack(fourth)
    assert reservation.released is True


@pytest.mark.asyncio
async def test_full_data_lane_does_not_consume_reserved_terminal_capacity() -> None:
    buffer = OutboundBuffer(data_capacity=1, lifecycle_capacity=1)
    reservation = await buffer.reserve_turn(1, timeout_s=0.1)
    await buffer.put_data(token(), turn_id=1, fenced=True)

    buffer.put_lifecycle_nowait(
        reservation,
        TerminalCommand(1, TurnOutcome.INTERRUPTED, "barge_in"),
    )

    assert buffer.data_size == 1
    assert buffer.lifecycle_size == 1


@pytest.mark.asyncio
async def test_reservation_times_out_at_capacity_and_releases_after_ack() -> None:
    buffer = OutboundBuffer(data_capacity=2, lifecycle_capacity=1)
    first_reservation = await buffer.reserve_turn(1, timeout_s=0.1)

    with pytest.raises(TimeoutError):
        await buffer.reserve_turn(2, timeout_s=0.01)

    buffer.put_lifecycle_nowait(
        first_reservation,
        TerminalCommand(1, TurnOutcome.FAILED, "provider_error"),
    )
    action = await buffer.next_dispatch(lambda _: False)
    assert isinstance(action, SendAction)
    await buffer.ack(action)

    second_reservation = await buffer.reserve_turn(2, timeout_s=0.1)
    assert second_reservation.turn_id == 2


@pytest.mark.asyncio
async def test_close_releases_outstanding_reservations_and_clears_memory() -> None:
    buffer = OutboundBuffer(data_capacity=2, lifecycle_capacity=2)
    first = await buffer.reserve_turn(1, timeout_s=0.1)
    second = await buffer.reserve_turn(2, timeout_s=0.1)
    await buffer.put_data(token(), turn_id=1, fenced=True)

    await buffer.close()

    assert first.released is True
    assert second.released is True
    assert buffer.reserved_count == 0
    assert buffer.data_size == 0
    assert buffer.lifecycle_size == 0
