from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.scenario import (
    InterviewScenario,
    ScenarioCatalog,
    load_scenario,
    load_scenario_catalog,
)


FIXTURE = Path(__file__).parents[1] / "scenarios" / "ecommerce.json"


def valid_payload() -> dict[str, object]:
    return {
        "id": "ecommerce",
        "bot": {
            "name": "ShopHelp",
            "system_prompt": "Help customers with orders and refunds.",
        },
        "mock_llm": {
            "responses": ["First response.", "Second response."],
            "token_delay_ms": 60,
        },
        "caller": {
            "turns": [
                {"frames": ["My order is damaged."]},
                {
                    "frames": ["I want a refund."],
                    "end": "silence",
                    "barge_in_after_ms": 250,
                },
            ]
        },
        "simulation": {
            "stt_partial_delay_ms": 20,
            "silence_timeout_ms": 350,
            "tts_chunk_delay_ms": 100,
        },
    }


def test_ecommerce_fixture_loads_with_interview_defaults() -> None:
    scenario = load_scenario(FIXTURE)

    assert scenario.id == "ecommerce"
    assert scenario.mock_llm.token_delay_ms == 60
    assert scenario.caller.turns[0].end == "explicit"
    assert scenario.caller.turns[1].barge_in_after_ms == 250
    assert len(scenario.mock_llm.responses[0].split(".")) >= 4


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update({"unexpected": True}), "unexpected"),
        (
            lambda data: data["mock_llm"].update({"token_delay_ms": "60"}),
            "token_delay_ms",
        ),
        (
            lambda data: data["caller"]["turns"][0].update({"frames": [""]}),
            "frames",
        ),
        (
            lambda data: data["mock_llm"].update({"responses": []}),
            "responses",
        ),
        (
            lambda data: data["simulation"].update({"silence_timeout_ms": 0}),
            "silence_timeout_ms",
        ),
        (
            lambda data: data["caller"]["turns"][0].update({"end": "later"}),
            "end",
        ),
        (
            lambda data: data["caller"]["turns"][0].update(
                {"barge_in_after_ms": 10}
            ),
            "first caller turn",
        ),
    ],
)
def test_scenario_validation_rejects_invalid_input(
    mutate: object,
    message: str,
) -> None:
    payload = valid_payload()
    mutate(payload)  # type: ignore[operator]

    with pytest.raises(ValidationError, match=message):
        InterviewScenario.model_validate(payload)


def test_loader_reports_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scenario file not found"):
        load_scenario(tmp_path / "missing.json")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid scenario JSON"):
        load_scenario(malformed)


def test_loader_rejects_non_object_json(tmp_path: Path) -> None:
    scenario_file = tmp_path / "scenario.json"
    scenario_file.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(ValueError, match="scenario root must be a JSON object"):
        load_scenario(scenario_file)


def test_catalog_loads_all_scenarios_in_stable_id_order() -> None:
    catalog = load_scenario_catalog(FIXTURE.parent, default_id="ecommerce")

    assert isinstance(catalog, ScenarioCatalog)
    assert catalog.default_id == "ecommerce"
    assert catalog.ids == ("appointments", "ecommerce", "telecom", "travel")
    assert catalog.select(None).id == "ecommerce"
    assert catalog.select("travel").bot.name == "TripAssist"


def test_catalog_rejects_missing_and_empty_directories(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scenario directory not found"):
        load_scenario_catalog(tmp_path / "missing")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="contains no JSON scenarios"):
        load_scenario_catalog(empty)


def test_catalog_rejects_duplicate_ids_and_missing_default(tmp_path: Path) -> None:
    duplicate_dir = tmp_path / "duplicate"
    duplicate_dir.mkdir()
    payload = valid_payload()
    for name in ("one.json", "two.json"):
        (duplicate_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate scenario id: ecommerce"):
        load_scenario_catalog(duplicate_dir)

    one_dir = tmp_path / "one"
    one_dir.mkdir()
    (one_dir / "ecommerce.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="default scenario id not found: travel"):
        load_scenario_catalog(one_dir, default_id="travel")


def test_catalog_identifies_the_malformed_file(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match=r"broken\.json"):
        load_scenario_catalog(tmp_path)
