from __future__ import annotations

from pathlib import Path

import pytest

from app.__main__ import (
    build_catalog_runtime,
    build_scenario_runtime,
    build_single_scenario_runtime,
)
from app.providers.llm import ChatMessage


FIXTURE = Path(__file__).parents[1] / "scenarios" / "ecommerce.json"
SCENARIOS = FIXTURE.parent


@pytest.mark.asyncio
async def test_scenario_runtime_overlays_prompt_and_simulation_timings() -> None:
    settings, provider_factory = build_scenario_runtime(
        FIXTURE,
        environ={"VOICE_PORT": "9000", "VOICE_TOKEN_QUEUE_SIZE": "7"},
    )

    assert settings.port == 9000
    assert settings.token_queue_size == 7
    assert settings.system_prompt.startswith("You are a concise ecommerce")
    assert settings.llm_token_delay_ms == 60
    assert settings.stt_partial_delay_ms == 20
    assert settings.silence_timeout_ms == 350
    assert settings.tts_sentence_delay_ms == 100

    first = provider_factory("session-a")
    second = provider_factory("session-b")
    messages = (ChatMessage(role="user", content="hello"),)

    first_session_response = " ".join(
        [token async for token in first.stream(messages, 1, "session-a")]
    )
    first_session_next_response = " ".join(
        [token async for token in first.stream(messages, 2, "session-a")]
    )
    second_session_response = " ".join(
        [token async for token in second.stream(messages, 1, "session-b")]
    )

    assert first_session_response.startswith("I am sorry")
    assert first_session_next_response.startswith("Understood")
    assert second_session_response == first_session_response


@pytest.mark.asyncio
async def test_catalog_runtime_selects_scenario_and_creates_fresh_providers() -> None:
    settings, runtime_factory = build_catalog_runtime(
        SCENARIOS,
        default_id="ecommerce",
        environ={"VOICE_PORT": "9001", "VOICE_OUTPUT_QUEUE_SIZE": "9"},
    )

    travel_a = runtime_factory("travel-a", "travel")
    travel_b = runtime_factory("travel-b", "travel")
    default = runtime_factory("default", None)

    assert settings.port == 9001
    assert settings.output_queue_size == 9
    assert travel_a.settings.system_prompt.startswith("You are a concise flight-change")
    assert travel_a.settings.llm_token_delay_ms == 50
    assert travel_a.settings.tts_sentence_delay_ms == 100
    assert default.settings.system_prompt.startswith("You are a concise ecommerce")

    messages = (ChatMessage(role="user", content="hello"),)
    first = " ".join(
        [token async for token in travel_a.provider.stream(messages, 1, "travel-a")]
    )
    second = " ".join(
        [token async for token in travel_a.provider.stream(messages, 2, "travel-a")]
    )
    independent = " ".join(
        [token async for token in travel_b.provider.stream(messages, 1, "travel-b")]
    )

    assert first.startswith("I can help you review")
    assert second.startswith("Thanks for clarifying")
    assert independent == first


def test_single_scenario_runtime_accepts_only_matching_id() -> None:
    _, runtime_factory = build_single_scenario_runtime(FIXTURE)

    assert runtime_factory("one", None).settings.llm_token_delay_ms == 60
    assert runtime_factory("two", "ecommerce").settings.llm_token_delay_ms == 60
    with pytest.raises(KeyError, match="travel"):
        runtime_factory("three", "travel")
