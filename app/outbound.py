from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from app.events import ServerEvent
from app.metrics import TurnOutcome


class OutboundClosedError(RuntimeError):
    pass


@dataclass(slots=True)
class LifecycleReservation:
    reservation_id: int
    turn_id: int
    used: bool = False
    released: bool = False


@dataclass(frozen=True, slots=True)
class TerminalCommand:
    turn_id: int
    outcome: TurnOutcome
    reason: str | None


@dataclass(frozen=True, slots=True)
class OutputCompleteMarker:
    turn_id: int


@dataclass(frozen=True, slots=True)
class _DataEnvelope:
    sequence: int
    item: ServerEvent | OutputCompleteMarker
    turn_id: int | None
    fenced: bool


@dataclass(frozen=True, slots=True)
class _LifecycleEnvelope:
    sequence: int
    command: TerminalCommand
    reservation: LifecycleReservation


@dataclass(frozen=True, slots=True)
class SendAction:
    sequence: int
    item: ServerEvent | TerminalCommand
    reservation: LifecycleReservation | None = None


@dataclass(frozen=True, slots=True)
class DropAction:
    sequence: int
    item: ServerEvent
    turn_id: int


@dataclass(frozen=True, slots=True)
class OutputCompleteAction:
    sequence: int
    turn_id: int


DispatchAction = SendAction | DropAction | OutputCompleteAction


class OutboundBuffer:
    def __init__(self, data_capacity: int, lifecycle_capacity: int) -> None:
        if data_capacity <= 0 or lifecycle_capacity <= 0:
            raise ValueError("outbound capacities must be positive")
        self._data_capacity = data_capacity
        self._lifecycle_capacity = lifecycle_capacity
        self._data: deque[_DataEnvelope] = deque()
        self._lifecycle: deque[_LifecycleEnvelope] = deque()
        self._reservations: dict[int, LifecycleReservation] = {}
        self._next_sequence = 1
        self._next_reservation_id = 1
        self._available = asyncio.Event()
        self._space_changed = asyncio.Condition()
        self._closed = False

    @property
    def data_size(self) -> int:
        return len(self._data)

    @property
    def lifecycle_size(self) -> int:
        return len(self._lifecycle)

    @property
    def reserved_count(self) -> int:
        return len(self._reservations)

    async def reserve_turn(
        self, turn_id: int, timeout_s: float
    ) -> LifecycleReservation:
        async with asyncio.timeout(timeout_s):
            async with self._space_changed:
                await self._space_changed.wait_for(
                    lambda: self._closed
                    or len(self._reservations) < self._lifecycle_capacity
                )
                if self._closed:
                    raise OutboundClosedError("outbound buffer is closed")
                reservation = LifecycleReservation(
                    reservation_id=self._next_reservation_id,
                    turn_id=turn_id,
                )
                self._next_reservation_id += 1
                self._reservations[reservation.reservation_id] = reservation
                return reservation

    async def put_data(
        self,
        event: ServerEvent,
        turn_id: int | None = None,
        fenced: bool = False,
    ) -> None:
        if fenced and turn_id is None:
            raise ValueError("fenced events require a turn_id")
        await self._put_data_item(event, turn_id=turn_id, fenced=fenced)

    async def put_output_complete(self, turn_id: int) -> None:
        await self._put_data_item(
            OutputCompleteMarker(turn_id), turn_id=turn_id, fenced=False
        )

    async def _put_data_item(
        self,
        item: ServerEvent | OutputCompleteMarker,
        *,
        turn_id: int | None,
        fenced: bool,
    ) -> None:
        async with self._space_changed:
            await self._space_changed.wait_for(
                lambda: self._closed or len(self._data) < self._data_capacity
            )
            if self._closed:
                raise OutboundClosedError("outbound buffer is closed")
            self._data.append(
                _DataEnvelope(
                    sequence=self._take_sequence(),
                    item=item,
                    turn_id=turn_id,
                    fenced=fenced,
                )
            )
            self._available.set()

    def put_lifecycle_nowait(
        self, reservation: LifecycleReservation, command: TerminalCommand
    ) -> None:
        if self._closed:
            raise OutboundClosedError("outbound buffer is closed")
        owned = self._reservations.get(reservation.reservation_id)
        if owned is not reservation or reservation.released:
            raise ValueError("lifecycle reservation is not active")
        if reservation.used:
            raise ValueError("lifecycle reservation has already been used")
        if reservation.turn_id != command.turn_id:
            raise ValueError("terminal command does not own this reservation")

        reservation.used = True
        self._lifecycle.append(
            _LifecycleEnvelope(
                sequence=self._take_sequence(),
                command=command,
                reservation=reservation,
            )
        )
        self._available.set()

    async def next_dispatch(
        self, is_valid: Callable[[int], bool]
    ) -> DispatchAction:
        while True:
            envelope = self._take_next_envelope()
            if envelope is not None:
                break
            if self._closed:
                raise OutboundClosedError("outbound buffer is closed")
            self._available.clear()
            if self._data or self._lifecycle:
                self._available.set()
                continue
            await self._available.wait()

        if isinstance(envelope, _LifecycleEnvelope):
            return SendAction(
                sequence=envelope.sequence,
                item=envelope.command,
                reservation=envelope.reservation,
            )

        await self._notify_space()
        item = envelope.item
        if isinstance(item, OutputCompleteMarker):
            return OutputCompleteAction(envelope.sequence, item.turn_id)
        if envelope.fenced and envelope.turn_id is not None and not is_valid(
            envelope.turn_id
        ):
            return DropAction(envelope.sequence, item, envelope.turn_id)
        return SendAction(envelope.sequence, item)

    async def ack(self, action: SendAction) -> None:
        if action.reservation is None:
            return
        self._release(action.reservation)
        await self._notify_space()

    async def close(self) -> None:
        self._closed = True
        self._data.clear()
        self._lifecycle.clear()
        for reservation in tuple(self._reservations.values()):
            self._release(reservation)
        self._available.set()
        await self._notify_space()

    def _take_sequence(self) -> int:
        sequence = self._next_sequence
        self._next_sequence += 1
        return sequence

    def _take_next_envelope(
        self,
    ) -> _DataEnvelope | _LifecycleEnvelope | None:
        if self._data and self._lifecycle:
            if self._data[0].sequence < self._lifecycle[0].sequence:
                return self._data.popleft()
            return self._lifecycle.popleft()
        if self._data:
            return self._data.popleft()
        if self._lifecycle:
            return self._lifecycle.popleft()
        return None

    def _release(self, reservation: LifecycleReservation) -> None:
        if reservation.released:
            return
        reservation.released = True
        self._reservations.pop(reservation.reservation_id, None)

    async def _notify_space(self) -> None:
        async with self._space_changed:
            self._space_changed.notify_all()
