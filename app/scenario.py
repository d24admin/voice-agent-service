from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)


NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BotScenario(_StrictModel):
    name: StrictStr
    system_prompt: StrictStr

    @field_validator("name", "system_prompt")
    @classmethod
    def require_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized


class MockLLMScenario(_StrictModel):
    responses: tuple[StrictStr, ...] = Field(min_length=1)
    token_delay_ms: NonNegativeInt

    @field_validator("responses")
    @classmethod
    def require_responses(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("responses must not contain empty text")
        return normalized


class CallerTurnScenario(_StrictModel):
    frames: tuple[StrictStr, ...] = Field(min_length=1)
    end: Literal["explicit", "silence"] = "explicit"
    barge_in_after_ms: NonNegativeInt | None = None

    @field_validator("frames")
    @classmethod
    def require_frames(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("frames must not contain empty text")
        return normalized


class CallerScenario(_StrictModel):
    turns: tuple[CallerTurnScenario, ...] = Field(min_length=1)


class SimulationScenario(_StrictModel):
    stt_partial_delay_ms: NonNegativeInt
    silence_timeout_ms: PositiveInt
    tts_chunk_delay_ms: NonNegativeInt


class InterviewScenario(_StrictModel):
    id: StrictStr
    bot: BotScenario
    mock_llm: MockLLMScenario
    caller: CallerScenario
    simulation: SimulationScenario

    @field_validator("id")
    @classmethod
    def require_slug(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
            raise ValueError("id must be a lowercase slug")
        return value

    @model_validator(mode="after")
    def reject_first_turn_barge_in(self) -> InterviewScenario:
        if self.caller.turns[0].barge_in_after_ms is not None:
            raise ValueError("first caller turn cannot configure barge-in")
        return self


@dataclass(frozen=True, slots=True)
class ScenarioCatalog:
    scenarios: tuple[InterviewScenario, ...]
    default_id: str

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(scenario.id for scenario in self.scenarios)

    def select(self, scenario_id: str | None) -> InterviewScenario:
        selected_id = self.default_id if scenario_id is None else scenario_id
        for scenario in self.scenarios:
            if scenario.id == selected_id:
                return scenario
        raise KeyError(selected_id)


def load_scenario(path: str | Path) -> InterviewScenario:
    scenario_path = Path(path)
    try:
        raw = scenario_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"scenario file not found: {scenario_path}") from exc
    except OSError as exc:
        raise ValueError(f"could not read scenario file: {scenario_path}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid scenario JSON: {scenario_path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("scenario root must be a JSON object")
    return InterviewScenario.model_validate(payload)


def load_scenario_catalog(
    directory: str | Path,
    *,
    default_id: str = "ecommerce",
) -> ScenarioCatalog:
    scenario_dir = Path(directory)
    if not scenario_dir.is_dir():
        raise ValueError(f"scenario directory not found: {scenario_dir}")
    files = sorted(scenario_dir.glob("*.json"))
    if not files:
        raise ValueError(f"scenario directory contains no JSON scenarios: {scenario_dir}")

    scenarios_by_id: dict[str, InterviewScenario] = {}
    for path in files:
        scenario = load_scenario(path)
        if scenario.id in scenarios_by_id:
            raise ValueError(f"duplicate scenario id: {scenario.id}")
        scenarios_by_id[scenario.id] = scenario
    if default_id not in scenarios_by_id:
        raise ValueError(f"default scenario id not found: {default_id}")
    return ScenarioCatalog(
        scenarios=tuple(scenarios_by_id[key] for key in sorted(scenarios_by_id)),
        default_id=default_id,
    )
