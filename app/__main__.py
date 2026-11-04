from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from app.config import Settings
from app.api.server import RuntimeFactory, SessionRuntime, create_app
from app.providers.llm import GeminiLLMProvider, LLMProvider, MockLLMProvider
from app.scenario import (
    InterviewScenario,
    ScenarioCatalog,
    load_scenario,
    load_scenario_catalog,
)


ProviderFactory = Callable[[str], LLMProvider]
DEFAULT_SCENARIO_DIR = Path("scenarios")
DEFAULT_SCENARIO_ID = "ecommerce"


def _scenario_settings(base: Settings, scenario: InterviewScenario) -> Settings:
    return Settings.model_validate(
        {
            **base.model_dump(),
            "system_prompt": scenario.bot.system_prompt,
            "llm_token_delay_ms": scenario.mock_llm.token_delay_ms,
            "stt_partial_delay_ms": scenario.simulation.stt_partial_delay_ms,
            "silence_timeout_ms": scenario.simulation.silence_timeout_ms,
            "tts_sentence_delay_ms": scenario.simulation.tts_chunk_delay_ms,
        }
    )


def _scenario_provider(
    settings: Settings, scenario: InterviewScenario
) -> LLMProvider:
    if settings.llm_provider == "mock":
        return MockLLMProvider(
            responses=scenario.mock_llm.responses,
            token_delay_s=scenario.mock_llm.token_delay_ms / 1_000,
        )
    if settings.llm_api_key is None:
        raise ValueError("Gemini provider requires an API key")
    return GeminiLLMProvider(
        api_key=settings.llm_api_key.get_secret_value(),
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        max_retries=settings.llm_max_retries,
        retry_delay_s=settings.llm_retry_delay_ms / 1_000,
    )


def _runtime_factory(
    catalog: ScenarioCatalog,
    base: Settings,
) -> RuntimeFactory:
    def create_runtime(
        _session_id: str,
        scenario_id: str | None,
    ) -> SessionRuntime:
        scenario = catalog.select(scenario_id)
        settings = _scenario_settings(base, scenario)
        return SessionRuntime(
            settings=settings,
            provider=_scenario_provider(settings, scenario),
        )

    return create_runtime


def build_scenario_runtime(
    scenario_file: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[Settings, ProviderFactory]:
    scenario = load_scenario(scenario_file)
    base = Settings.from_environment(environ)
    settings = _scenario_settings(base, scenario)

    def provider_factory(_session_id: str) -> LLMProvider:
        return _scenario_provider(settings, scenario)

    return settings, provider_factory


def build_catalog_runtime(
    scenario_dir: str | Path,
    *,
    default_id: str = DEFAULT_SCENARIO_ID,
    environ: Mapping[str, str] | None = None,
) -> tuple[Settings, RuntimeFactory]:
    catalog = load_scenario_catalog(scenario_dir, default_id=default_id)
    base = Settings.from_environment(environ)
    return base, _runtime_factory(catalog, base)


def build_single_scenario_runtime(
    scenario_file: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[Settings, RuntimeFactory]:
    scenario = load_scenario(scenario_file)
    base = Settings.from_environment(environ)
    catalog = ScenarioCatalog((scenario,), scenario.id)
    return base, _runtime_factory(catalog, base)


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv(".env", override=False)
    parser = argparse.ArgumentParser(description="Run the simulated voice service")
    scenarios = parser.add_mutually_exclusive_group()
    scenarios.add_argument(
        "--scenario-dir",
        type=Path,
        help="directory containing validated interview scenarios",
    )
    scenarios.add_argument(
        "--scenario-file",
        type=Path,
        help="single validated interview scenario JSON",
    )
    parser.add_argument(
        "--default-scenario-id",
        default=DEFAULT_SCENARIO_ID,
        help="fallback scenario ID in catalog mode",
    )
    args = parser.parse_args(argv)
    if args.scenario_file is not None:
        settings, runtime_factory = build_single_scenario_runtime(
            args.scenario_file
        )
    else:
        settings, runtime_factory = build_catalog_runtime(
            args.scenario_dir or DEFAULT_SCENARIO_DIR,
            default_id=args.default_scenario_id,
        )
    uvicorn.run(
        create_app(settings, runtime_factory=runtime_factory),
        host=settings.host,
        port=settings.port,
    )


if __name__ == "__main__":
    main()
