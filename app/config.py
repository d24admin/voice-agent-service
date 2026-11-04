from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    SecretStr,
    field_validator,
    model_validator,
)


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
    llm_provider: Literal["mock", "gemini"] = "mock"
    llm_api_key: SecretStr | None = None
    llm_model: str = "gemini-3.5-flash-lite"
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    llm_max_retries: int = Field(default=1, ge=0, le=3)
    llm_retry_delay_ms: NonNegativeInt = 250
    llm_failure_message: str = (
        "I’m having trouble connecting right now. "
        "Please try again in a moment."
    )

    @field_validator("llm_model", "llm_failure_message")
    @classmethod
    def _strip_non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator("llm_base_url")
    @classmethod
    def _validate_llm_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("LLM base URL must be an HTTP(S) URL without a query")
        return normalized

    @model_validator(mode="after")
    def _require_gemini_api_key(self) -> Settings:
        if self.llm_provider != "gemini":
            return self
        if self.llm_api_key is None or not self.llm_api_key.get_secret_value().strip():
            raise ValueError(
                "VOICE_LLM_API_KEY is required when VOICE_LLM_PROVIDER=gemini"
            )
        return self

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
