from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.metrics import TurnMetrics, TurnOutcome
from app.scenario import InterviewScenario, load_scenario, load_scenario_catalog
from client import CallResult
from load_test import OutcomeCounts, PercentileSummary, render_report, run_load
from load_test.__main__ import main


SCENARIOS = Path(__file__).parents[1] / "scenarios"
FIXTURE = SCENARIOS / "ecommerce.json"


def metrics(turn_id: int, outcome: TurnOutcome, value: float) -> TurnMetrics:
    item = TurnMetrics(turn_id=turn_id, baseline_ns=0)
    item.ttft_ms = value
    item.ttfa_ms = value + 100
    item.total_latency_ms = value + 200
    item.outcome = outcome
    item.reason = "barge_in" if outcome is TurnOutcome.INTERRUPTED else None
    return item


def successful_result(
    session_id: str,
    value: float = 10,
    *,
    interrupted: bool = False,
) -> CallResult:
    turns = [metrics(1, TurnOutcome.COMPLETED, value)]
    if interrupted:
        turns.insert(0, metrics(1, TurnOutcome.INTERRUPTED, value - 1))
        turns[1].turn_id = 2
    return CallResult(
        session_id=session_id,
        outcome="completed",
        turn_metrics=tuple(turns),
        interrupted_turns=int(interrupted),
        errors=(),
    )


@pytest.mark.asyncio
async def test_seeded_assignments_are_reproducible_and_can_change() -> None:
    scenarios = load_scenario_catalog(SCENARIOS).scenarios

    async def fake_run_call(
        server_url: str,
        scenario: InterviewScenario,
        session_id: str,
    ) -> CallResult:
        del server_url, scenario
        return successful_result(session_id)

    first = await run_load(
        concurrency=20,
        scenarios=scenarios,
        seed=42,
        call_runner=fake_run_call,
    )
    repeated = await run_load(
        concurrency=20,
        scenarios=reversed(scenarios),
        seed=42,
        call_runner=fake_run_call,
    )
    changed = await run_load(
        concurrency=20,
        scenarios=scenarios,
        seed=7,
        call_runner=fake_run_call,
    )

    assert [item.scenario_id for item in first.assignments] == [
        item.scenario_id for item in repeated.assignments
    ]
    assert [item.scenario_id for item in changed.assignments] != [
        item.scenario_id for item in first.assignments
    ]
    assert sum(agent.requested for agent in first.agents) == 20


@pytest.mark.asyncio
async def test_twenty_calls_report_unique_sessions_peak_concurrency_and_isolated_failure() -> None:
    scenarios = load_scenario_catalog(SCENARIOS).scenarios
    started = 0
    release = asyncio.Event()

    async def blocking_call(
        server_url: str,
        scenario: InterviewScenario,
        session_id: str,
    ) -> CallResult:
        nonlocal started
        del server_url
        assert scenario.id in session_id
        started += 1
        if started == 20:
            release.set()
        await release.wait()
        if session_id.endswith("-7"):
            raise OSError("one simulated call failed")
        return successful_result(session_id, interrupted=True)

    report = await run_load(
        concurrency=20,
        scenarios=scenarios,
        seed=42,
        call_runner=blocking_call,
    )

    assert report.requested == 20
    assert report.succeeded == 19
    assert report.failed == 1
    assert report.unique_sessions == 20
    assert report.peak_concurrency == 20
    assert report.turns == OutcomeCounts(completed=19, interrupted=19, failed=0)
    assert len({item.session_id for item in report.assignments}) == 20
    assert len(report.errors) == 1
    assert "one simulated call failed" in report.errors[0]
    assert any(agent.failed == 1 for agent in report.agents)
    assert sum(agent.succeeded for agent in report.agents) == 19
    assert sum(agent.failed for agent in report.agents) == 1


@pytest.mark.asyncio
async def test_latencies_use_completed_turns_from_successful_calls_only() -> None:
    ecommerce = load_scenario(FIXTURE)
    calls = 0

    async def mixed_call(
        server_url: str,
        scenario: InterviewScenario,
        session_id: str,
    ) -> CallResult:
        nonlocal calls
        del server_url, scenario
        calls += 1
        if calls == 1:
            return successful_result(session_id, value=10)
        return CallResult(
            session_id=session_id,
            outcome="failed",
            turn_metrics=(
                metrics(1, TurnOutcome.COMPLETED, 999),
                metrics(2, TurnOutcome.FAILED, 1),
            ),
            interrupted_turns=0,
            errors=("provider failed",),
        )

    report = await run_load(
        concurrency=2,
        scenarios=(ecommerce,),
        call_runner=mixed_call,
    )

    assert report.succeeded == 1
    assert report.failed == 1
    assert report.turns == OutcomeCounts(completed=2, interrupted=0, failed=1)
    assert report.ttft == PercentileSummary(sample_count=1, p50=10, p95=10)
    assert report.ttfa == PercentileSummary(sample_count=1, p50=110, p95=110)
    assert report.total_latency == PercentileSummary(
        sample_count=1, p50=210, p95=210
    )
    assert report.agents[0].ttft == report.ttft
    assert "ecommerce/" in report.errors[0]
    assert "provider failed" in report.errors[0]


@pytest.mark.asyncio
async def test_single_scenario_mode_assigns_every_call_to_that_agent() -> None:
    observed: list[str] = []

    async def fake_run_call(
        server_url: str,
        scenario: InterviewScenario,
        session_id: str,
    ) -> CallResult:
        del server_url
        observed.append(scenario.id)
        return successful_result(session_id)

    report = await run_load(
        concurrency=5,
        scenario=load_scenario(FIXTURE),
        seed=99,
        call_runner=fake_run_call,
    )

    assert observed == ["ecommerce"] * 5
    assert [agent.scenario_id for agent in report.agents] == ["ecommerce"]
    assert report.agents[0].requested == 5


@pytest.mark.asyncio
async def test_load_session_ids_stay_within_the_server_limit() -> None:
    long_id = "a" * 120
    scenario = load_scenario(FIXTURE).model_copy(update={"id": long_id})
    observed: list[str] = []

    async def fake_run_call(
        server_url: str,
        selected: InterviewScenario,
        session_id: str,
    ) -> CallResult:
        del server_url
        assert selected.id == long_id
        observed.append(session_id)
        return successful_result(session_id)

    await run_load(
        concurrency=2,
        scenarios=(scenario,),
        call_runner=fake_run_call,
    )

    assert len(set(observed)) == 2
    assert all(len(session_id) <= 128 for session_id in observed)
    assert all(long_id[:80] in session_id for session_id in observed)


@pytest.mark.asyncio
async def test_empty_samples_and_per_agent_report_are_readable(capsys: object) -> None:
    scenarios = load_scenario_catalog(SCENARIOS).scenarios

    async def empty_call(
        server_url: str,
        scenario: InterviewScenario,
        session_id: str,
    ) -> CallResult:
        del server_url, scenario
        return CallResult(session_id, "completed", (), 0, (), ())

    report = await run_load(
        concurrency=4,
        scenarios=scenarios,
        seed=42,
        call_runner=empty_call,
    )
    render_report(report)
    output = capsys.readouterr().out  # type: ignore[attr-defined]

    assert report.ttft == PercentileSummary(0, None, None)
    assert "LOAD TEST · seed 42" in output
    assert "CALLS     requested 4 · succeeded 4 · failed 0" in output
    assert "SESSIONS  unique 4 · peak concurrency 1" in output
    assert "COMPLETED-TURN LATENCY · OVERALL" in output
    assert "AGENTS" in output
    assert "samples 0 · p50 n/a · p95 n/a" in output
    positions = [output.index(agent.scenario_id) for agent in report.agents]
    assert positions == sorted(positions)


@pytest.mark.asyncio
async def test_invalid_load_inputs_are_rejected() -> None:
    ecommerce = load_scenario(FIXTURE)
    with pytest.raises(ValueError, match="positive"):
        await run_load(concurrency=0, scenarios=(ecommerce,))
    with pytest.raises(ValueError, match="at least one scenario"):
        await run_load(concurrency=1, scenarios=())
    with pytest.raises(ValueError, match="mutually exclusive"):
        await run_load(concurrency=1, scenarios=(ecommerce,), scenario=ecommerce)


def test_cli_supports_catalog_and_single_file_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[tuple[list[str], int]] = []

    async def fake_run_load(**kwargs: object) -> object:
        selected = kwargs["scenarios"]
        captured.append(
            ([scenario.id for scenario in selected], int(kwargs["seed"]))  # type: ignore[union-attr]
        )
        return await run_load(
            concurrency=1,
            scenarios=selected,  # type: ignore[arg-type]
            call_runner=lambda _url, _scenario, session_id: asyncio.sleep(
                0, result=successful_result(session_id)
            ),
        )

    monkeypatch.setattr("load_test.__main__.run_load", fake_run_load)

    assert main(["--scenario-dir", str(SCENARIOS), "--concurrency", "1"]) == 0
    assert main(
        [
            "--scenario-file",
            str(FIXTURE),
            "--concurrency",
            "1",
            "--seed",
            "7",
        ]
    ) == 0
    gemini_fixture = SCENARIOS / "gemini" / "load-test.json"
    assert main(
        [
            "--scenario-file",
            str(gemini_fixture),
            "--concurrency",
            "1",
            "--seed",
            "11",
        ]
    ) == 0
    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "travel.json").write_text(
        (SCENARIOS / "travel.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    assert main(
        [
            "--scenario-dir",
            str(custom),
            "--default-scenario-id",
            "travel",
            "--concurrency",
            "1",
        ]
    ) == 0
    assert captured == [
        (["appointments", "ecommerce", "telecom", "travel"], 42),
        (["ecommerce"], 7),
        (["ecommerce-gemini-load"], 11),
        (["travel"], 42),
    ]
