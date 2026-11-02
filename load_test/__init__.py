from __future__ import annotations

import asyncio
import random
import uuid
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.metrics import TurnMetrics, TurnOutcome, nearest_rank
from app.scenario import InterviewScenario, load_scenario_catalog
from client import CallResult, run_call


CallRunner = Callable[[str, InterviewScenario, str], Awaitable[CallResult]]
MAX_SESSION_ID_LENGTH = 128


@dataclass(frozen=True, slots=True)
class PercentileSummary:
    sample_count: int
    p50: float | None
    p95: float | None


@dataclass(frozen=True, slots=True)
class OutcomeCounts:
    completed: int = 0
    interrupted: int = 0
    failed: int = 0


@dataclass(frozen=True, slots=True)
class CallAssignment:
    scenario_id: str
    session_id: str


@dataclass(frozen=True, slots=True)
class AgentReport:
    scenario_id: str
    requested: int
    succeeded: int
    failed: int
    turns: OutcomeCounts
    ttft: PercentileSummary
    ttfa: PercentileSummary
    total_latency: PercentileSummary


@dataclass(frozen=True, slots=True)
class LoadReport:
    requested: int
    succeeded: int
    failed: int
    seed: int
    unique_sessions: int
    peak_concurrency: int
    turns: OutcomeCounts
    ttft: PercentileSummary
    ttfa: PercentileSummary
    total_latency: PercentileSummary
    agents: tuple[AgentReport, ...]
    assignments: tuple[CallAssignment, ...]
    errors: tuple[str, ...]

    @property
    def interrupted(self) -> int:
        return self.turns.interrupted


@dataclass(frozen=True, slots=True)
class _CallRecord:
    assignment: CallAssignment
    scenario: InterviewScenario
    result: CallResult | None
    error: BaseException | None

    @property
    def succeeded(self) -> bool:
        if self.result is None or self.error is not None:
            return False
        return not (
            self.result.errors
            or self.result.outcome == "failed"
            or any(
                turn.outcome is TurnOutcome.FAILED
                for turn in self.result.turn_metrics
            )
        )


async def run_load(
    concurrency: int = 20,
    scenarios: Iterable[InterviewScenario] | None = None,
    server_url: str = "ws://127.0.0.1:8000",
    call_runner: CallRunner = run_call,
    *,
    scenario: InterviewScenario | None = None,
    seed: int = 42,
) -> LoadReport:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if scenarios is not None and scenario is not None:
        raise ValueError("scenarios and scenario are mutually exclusive")
    if scenario is not None:
        selected_scenarios = (scenario,)
    elif scenarios is None:
        selected_scenarios = load_scenario_catalog(
            Path("scenarios"), default_id="ecommerce"
        ).scenarios
    else:
        selected_scenarios = tuple(scenarios)
    if not selected_scenarios:
        raise ValueError("at least one scenario is required")
    selected_scenarios = tuple(
        sorted(selected_scenarios, key=lambda item: item.id)
    )
    scenario_ids = [item.id for item in selected_scenarios]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenario IDs must be unique")

    generator = random.Random(seed)
    batch_id = uuid.uuid4().hex[:8]
    selected_calls = [
        generator.choice(selected_scenarios) for _ in range(concurrency)
    ]
    assignments = tuple(
        CallAssignment(
            scenario_id=selected.id,
            session_id=_load_session_id(batch_id, selected.id, index),
        )
        for index, selected in enumerate(selected_calls)
    )

    active = 0
    peak_concurrency = 0

    async def execute(
        assignment: CallAssignment,
        selected: InterviewScenario,
    ) -> CallResult:
        nonlocal active, peak_concurrency
        active += 1
        peak_concurrency = max(peak_concurrency, active)
        try:
            return await call_runner(
                server_url,
                selected,
                assignment.session_id,
            )
        finally:
            active -= 1

    raw_results = await asyncio.gather(
        *(
            execute(assignment, selected)
            for assignment, selected in zip(assignments, selected_calls)
        ),
        return_exceptions=True,
    )
    records = tuple(
        _CallRecord(
            assignment=assignment,
            scenario=selected,
            result=raw if isinstance(raw, CallResult) else None,
            error=raw if isinstance(raw, BaseException) else None,
        )
        for assignment, selected, raw in zip(
            assignments, selected_calls, raw_results
        )
    )
    successful = tuple(record for record in records if record.succeeded)
    all_turns = tuple(
        turn
        for record in records
        if record.result is not None
        for turn in record.result.turn_metrics
    )
    completed_success_turns = tuple(
        turn
        for record in successful
        if record.result is not None
        for turn in record.result.turn_metrics
        if turn.outcome is TurnOutcome.COMPLETED
    )

    agents = tuple(
        _agent_report(
            scenario_id=item.id,
            records=records,
        )
        for item in selected_scenarios
    )
    ttft, ttfa, total_latency = _latency_summaries(completed_success_turns)
    return LoadReport(
        requested=concurrency,
        succeeded=len(successful),
        failed=concurrency - len(successful),
        seed=seed,
        unique_sessions=len({item.session_id for item in assignments}),
        peak_concurrency=peak_concurrency,
        turns=_outcome_counts(all_turns),
        ttft=ttft,
        ttfa=ttfa,
        total_latency=total_latency,
        agents=agents,
        assignments=assignments,
        errors=tuple(
            error
            for record in records
            for error in _record_errors(record)
        ),
    )


def _agent_report(
    *,
    scenario_id: str,
    records: Sequence[_CallRecord],
) -> AgentReport:
    selected = tuple(
        record
        for record in records
        if record.assignment.scenario_id == scenario_id
    )
    successful = tuple(record for record in selected if record.succeeded)
    turns = tuple(
        turn
        for record in selected
        if record.result is not None
        for turn in record.result.turn_metrics
    )
    completed = tuple(
        turn
        for record in successful
        if record.result is not None
        for turn in record.result.turn_metrics
        if turn.outcome is TurnOutcome.COMPLETED
    )
    ttft, ttfa, total_latency = _latency_summaries(completed)
    return AgentReport(
        scenario_id=scenario_id,
        requested=len(selected),
        succeeded=len(successful),
        failed=len(selected) - len(successful),
        turns=_outcome_counts(turns),
        ttft=ttft,
        ttfa=ttfa,
        total_latency=total_latency,
    )


def _load_session_id(batch_id: str, scenario_id: str, index: int) -> str:
    prefix = f"load-{batch_id}-"
    suffix = f"-{index}"
    available = MAX_SESSION_ID_LENGTH - len(prefix) - len(suffix)
    return f"{prefix}{scenario_id[:available]}{suffix}"


def _outcome_counts(turns: Iterable[TurnMetrics]) -> OutcomeCounts:
    values = tuple(turns)
    return OutcomeCounts(
        completed=sum(turn.outcome is TurnOutcome.COMPLETED for turn in values),
        interrupted=sum(
            turn.outcome is TurnOutcome.INTERRUPTED for turn in values
        ),
        failed=sum(turn.outcome is TurnOutcome.FAILED for turn in values),
    )


def _latency_summaries(
    turns: Iterable[TurnMetrics],
) -> tuple[PercentileSummary, PercentileSummary, PercentileSummary]:
    values = tuple(turns)
    return (
        _summary([turn.ttft_ms for turn in values if turn.ttft_ms is not None]),
        _summary([turn.ttfa_ms for turn in values if turn.ttfa_ms is not None]),
        _summary(
            [
                turn.total_latency_ms
                for turn in values
                if turn.total_latency_ms is not None
            ]
        ),
    )


def _summary(values: Sequence[float]) -> PercentileSummary:
    return PercentileSummary(
        sample_count=len(values),
        p50=nearest_rank(values, 0.50),
        p95=nearest_rank(values, 0.95),
    )


def _record_errors(record: _CallRecord) -> tuple[str, ...]:
    prefix = f"{record.assignment.scenario_id}/{record.assignment.session_id}"
    if record.error is not None:
        return (f"{prefix}: {_safe_error(record.error)}",)
    if record.succeeded or record.result is None:
        return ()
    messages = record.result.errors or ("call reported failed outcome",)
    return tuple(f"{prefix}: {message}" for message in messages)


def _safe_error(error: BaseException) -> str:
    message = str(error).strip()
    return message or type(error).__name__


def render_report(report: LoadReport) -> None:
    print(f"LOAD TEST · seed {report.seed}")
    print("─" * 48)
    print(
        f"CALLS     requested {report.requested} · "
        f"succeeded {report.succeeded} · failed {report.failed}"
    )
    print(
        f"SESSIONS  unique {report.unique_sessions} · "
        f"peak concurrency {report.peak_concurrency}"
    )
    print(f"TURNS     {_turn_counts(report.turns)}")
    print("\nCOMPLETED-TURN LATENCY · OVERALL")
    print("Metric  Samples  p50       p95")
    for name, summary in _summary_rows(report):
        print(
            f"{name:<7} {summary.sample_count:<8} "
            f"{_metric(summary.p50):<9} {_metric(summary.p95)}"
        )

    print("\nAGENTS")
    for agent in report.agents:
        print(agent.scenario_id)
        print(
            f"  Calls  requested {agent.requested} · "
            f"succeeded {agent.succeeded} · failed {agent.failed}"
        )
        print(f"  Turns  {_turn_counts(agent.turns)}")
        for name, summary in _summary_rows(agent):
            print(
                f"  {name:<5} samples {summary.sample_count} · "
                f"p50 {_metric(summary.p50)} · p95 {_metric(summary.p95)}"
            )
    for error in report.errors:
        print(f"\nERROR  {error}")


def _summary_rows(
    report: LoadReport | AgentReport,
) -> tuple[tuple[str, PercentileSummary], ...]:
    return (
        ("TTFT", report.ttft),
        ("TTFA", report.ttfa),
        ("Total", report.total_latency),
    )


def _turn_counts(counts: OutcomeCounts) -> str:
    return (
        f"completed {counts.completed} · interrupted {counts.interrupted} · "
        f"failed {counts.failed}"
    )


def _metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1_000:
        return f"{value / 1_000:.2f} s"
    rendered = f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{rendered} ms"
