from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt


class Settings(BaseModel):
    """Validated service settings loaded from explicit VOICE_* variables."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    system_prompt: str = "You are a concise, helpful voice assistant."
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65_535)
    stt_partial_delay_ms: NonNegativeInt = 20
    silence_timeout_ms: PositiveInt = 350
    llm_token_delay_ms: NonNegativeInt = 40
    tts_sentence_delay_ms: NonNegativeInt = 100
    first_token_timeout_ms: PositiveInt = 500
    generation_timeout_ms: PositiveInt = 5_000
    shutdown_grace_ms: PositiveInt = 1_000
    max_message_bytes: PositiveInt = 65_536
    token_queue_size: PositiveInt = 32
    output_queue_size: PositiveInt = 128

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> Settings:
        source = os.environ if environ is None else environ
        supported = {
            f"VOICE_{field_name.upper()}": field_name
            for field_name in cls.model_fields
        }
        unknown = sorted(
            key for key in source if key.startswith("VOICE_") and key not in supported
        )
        if unknown:
            raise ValueError(f"Unsupported voice setting: {', '.join(unknown)}")

        values: dict[str, Any] = {
            field_name: source[environment_name]
            for environment_name, field_name in supported.items()
            if environment_name in source
        }
        return cls.model_validate(values)
