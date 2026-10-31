from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    TypeAdapter,
    field_validator,
)


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str
    event_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return value


class SessionStart(Event):
    type: Literal["session.start"] = "session.start"


class AudioFrame(Event):
    type: Literal["audio.frame"] = "audio.frame"
    text: str

    @field_validator("text")
    @classmethod
    def require_audio_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("audio frame text must not be empty")
        return value


class AudioEnd(Event):
    type: Literal["audio.end"] = "audio.end"


class SessionStop(Event):
    type: Literal["session.stop"] = "session.stop"


class Ping(Event):
    type: Literal["ping"] = "ping"


ClientEvent: TypeAlias = Annotated[
    SessionStart | AudioFrame | AudioEnd | SessionStop | Ping,
    Field(discriminator="type"),
]
_CLIENT_EVENT_ADAPTER = TypeAdapter(ClientEvent)


def parse_client_event(payload: object) -> ClientEvent:
    return _CLIENT_EVENT_ADAPTER.validate_python(payload)


class TurnEvent(Event):
    turn_id: StrictInt = Field(ge=1)


class SessionReady(Event):
    type: Literal["session.ready"] = "session.ready"


class TranscriptPartial(Event):
    type: Literal["transcript.partial"] = "transcript.partial"
    utterance_id: StrictInt = Field(ge=1)
    text: str


class TranscriptFinal(Event):
    type: Literal["transcript.final"] = "transcript.final"
    utterance_id: StrictInt = Field(ge=1)
    text: str


class LLMToken(TurnEvent):
    type: Literal["llm.token"] = "llm.token"
    token: str


class AudioChunk(TurnEvent):
    type: Literal["audio.chunk"] = "audio.chunk"
    text: str
    sequence: StrictInt = Field(ge=0)


class TurnInterrupted(TurnEvent):
    type: Literal["turn.interrupted"] = "turn.interrupted"
    reason: str
    metrics: dict[str, Any]


class TurnCompleted(TurnEvent):
    type: Literal["turn.completed"] = "turn.completed"
    metrics: dict[str, Any]


class TurnFailed(TurnEvent):
    type: Literal["turn.failed"] = "turn.failed"
    reason: str
    metrics: dict[str, Any]


class ErrorEvent(Event):
    type: Literal["error"] = "error"
    code: str
    message: str
    fatal: bool = False


class Pong(Event):
    type: Literal["pong"] = "pong"


ServerEvent: TypeAlias = (
    SessionReady
    | TranscriptPartial
    | TranscriptFinal
    | LLMToken
    | AudioChunk
    | TurnInterrupted
    | TurnCompleted
    | TurnFailed
    | ErrorEvent
    | Pong
)
