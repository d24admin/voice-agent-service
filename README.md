# Voice Agent Service

A locally runnable orchestration layer for a streaming voice bot. Text stands
in for audio while the service demonstrates partial/final STT, streaming LLM
tokens, sentence TTS, caller interruption, bounded queues, cancellation, event
ordering, and latency measurement.

The default provider is deterministic mock and needs no credentials. Optional
Gemini integration and fixtures live outside the default four-agent catalog.
Ecommerce is the default agent and mock barge-in demo.

## Review this project in five minutes

Python 3.11 or newer is required:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start the server, then run the scripted caller from another terminal:

```bash
python -m app --scenario-dir scenarios
python -m client --scenario-file scenarios/ecommerce.json
```

- With no Gemini provider configured, the server uses mock responses.
- The caller streams a response and barges in 50 ms after the first audio
  chunk.
- Run a deterministic 20-call load check and the test suite with:

```bash
python -m load_test --scenario-dir scenarios --concurrency 20 --seed 42
pytest -q
```

- The load runner connects to the running server; it does not select the
  provider. Gemini load testing is documented separately below.

To follow the main execution path, read [app/api/server.py](app/api/server.py),
[app/conversation/call_session.py](app/conversation/call_session.py),
[app/conversation/turn_pipeline.py](app/conversation/turn_pipeline.py), and
[tests/test_barge_in.py](tests/test_barge_in.py).

- The mock service needs no microphone or audio device; caller and bot audio
  are represented as text frames.
- Keep the server running while the client or load runner connects from
  another terminal.

## What successful output looks like

- The CLI shows transcripts, streamed tokens, spoken sentences, turn outcomes,
  and latency.
- Barge-in appears as `INTERRUPTED · barge_in`; the next turn continues the
  conversation and interrupted bot text is labelled partial.
- `--verbose` emits JSON Lines for machine processing.

For example, a call trace contains:

```text
STT ✓   My order arrived damaged.
LLM     I can help with that request.
TTS ▶   “I can help with that request.”
⚡ BARGE-IN · turn 2 started 50 ms after first audio
RESULT  INTERRUPTED · barge_in
```

- The load report includes requested and successful calls, unique sessions,
  peak concurrency, turn outcomes, and per-agent totals.
- Completed-turn p50/p95 latency is reported overall and per agent; only
  successful calls contribute samples.
- The client fails if stale tokens or audio arrive after `turn.interrupted`.
- The renderer keeps tokens on a flowing line and marks TTS and terminal
  events, showing their order without dumping protocol JSON.

## Runtime architecture

```text
Scenario catalog
  ├── server selects one agent with ?scenario_id=...
  ├── scenario configures prompt, responses, and timings
  └── client runs that scenario's caller turns and barge-in timing

JSON WebSocket
      │
      ▼
CallSession controller ──► simulated STT ──► final transcript
      │                                           │
      │                                           ▼
      │                              per-turn asyncio.TaskGroup
      │                              ┌─────────────────────────┐
      │                              │ LLM producer            │
      │                              │       │ bounded tokens  │
      │                              │       ▼                 │
      │                              │ sentence TTS consumer   │
      │                              └────────────┬────────────┘
      │                                           │
      └────────────► bounded FIFO output queue ◄──┘
                                │
                                ▼
                         one socket writer
```

- Each `CallSession` owns its connection and tasks; the controller handles
  input, shutdown, writer failures, and silence deadlines.
- A per-turn `TaskGroup` connects the LLM producer and TTS consumer through a
  bounded queue, so TTS can speak completed sentences while generation
  continues.
- One bounded event queue and writer preserve WebSocket ordering.
- Process-level metrics stay separate from per-session conversation state.
  The local demo needs no database, broker, or persistent store.
- Provider factories create a fresh mock or Gemini provider per session, and
  the session closes its Gemini HTTP client.

## Turn flow and barge-in

- A turn receives caller frames and finalizes a transcript through explicit
  or silence endpointing.
- It streams LLM tokens, emits sentence-sized `audio.chunk` events, and ends
  with a terminal event.

- An incoming caller frame interrupts the active bot turn before entering
  STT.
- The session marks the turn interrupted, invalidates its output, cancels the
  turn task, queues `turn.interrupted`, then processes the new frame.
- Input does not wait for provider cleanup. The pipeline and writer drop
  invalid old-turn output, and terminal state can be claimed only once.

- For explicit endpointing, the client sends `audio.end`; for silence
  endpointing, the controller finalizes STT after the configured quiet period.
- Configured barge-in starts after the first `audio.chunk`; after the delay,
  the client sends the next turn.
- If the prior turn finishes before that delay, the client reports that the
  expected interruption did not occur.

- Interruption currently cancels a turn immediately. This text-audio demo
  cannot cut off speech mid-sentence because each `audio.chunk` contains a full
  sentence.
- For production audio, the TTS adapter should emit smaller audio chunks and
  interrupt playback at chunk boundaries, stopping or discarding buffered
  chunks when caller speech arrives.

## Design choices and tradeoffs

| Choice | Benefit | Tradeoff |
|---|---|---|
| One `CallSession` per WebSocket | Co-locates state and task ownership | State is in-memory and connection-scoped |
| Structured task groups and bounded queues | Cleanup and backpressure are explicit | Slow consumers can delay producers |
| One ordered writer with turn-ID fencing | Prevents concurrent sends and stale queued output | In-flight bytes cannot be recalled |
| Fresh provider per session | Isolates mock cursors and Gemini HTTP clients | Each Gemini call has its own connection pool |
| Retry only before first token | Recovers transient errors without duplicate output | Later provider errors fail the turn |
| Strict scenarios and seeded load assignment | Reproducible local demos and load distribution | Test routing is not production routing |
| In-memory metrics | Reliable monotonic durations without infrastructure | Metrics reset with the process |
| Sentence TTS overlap | Starts speech before full generation ends | The simulation does not model audio playback |
| Text-based caller fixtures | Repeatable endpointing and interruption checks | No microphone, VAD, or codec behavior is measured |
| Strict shared scenario catalog | Server and load runner validate the same agent IDs | A single-scenario server accepts only its fixture's ID |

- Reconnect continuity, persistent metrics, and real media are outside this
  assessment. Gemini demonstrates the external-provider boundary.

## Async task ownership

| Task | Owner and cleanup |
|---|---|
| WebSocket receive and shutdown waiter | Controller replaces, cancels, and gathers them |
| Socket writer | Serializes sends; stalled writes are cancelled after shutdown grace |
| Turn task and LLM/TTS children | Turn task is cancelled on interruption; `TaskGroup` cleans up siblings |
| Gemini HTTP client | `CallSession` closes it when the session ends |
| Client barge timer and load calls | Client gathers timer/receive tasks; load runner retrieves each result |
| Per-turn `TaskGroup` | Owns the LLM producer and TTS consumer; sibling failures cancel related work |
| Pending WebSocket receive | Controller replaces it after each event and gathers it during cleanup |

- Shutdown stops live sessions, waits for the configured grace period, then
  cancels remaining work and closes transports.
- The controller replaces its pending receive as events arrive. During
  barge-in, the session owns the cancelled turn task until cleanup finishes,
  so a slow provider cannot leave an untracked background task.

## Assessment requirements and evidence

| Requirement | Implementation | Primary evidence |
|---|---|---|
| WebSocket protocol and scripted client | `app/api`, `client` | [tests/test_api.py](tests/test_api.py), [tests/test_client.py](tests/test_client.py) |
| Partial/final STT and silence endpointing | `app/providers/stt.py`, `CallSession` | [tests/test_stt.py](tests/test_stt.py), [tests/test_session.py](tests/test_session.py) |
| Streaming LLM/TTS overlap | `ActiveTurnRunner`, [app/providers/tts.py](app/providers/tts.py) | [tests/test_turn_pipeline.py](tests/test_turn_pipeline.py) |
| Barge-in and stale-output fencing | `CallSession`, `TurnState`, writer | [tests/test_barge_in.py](tests/test_barge_in.py) |
| Gemini mapping, retries, and failures | [app/providers/llm.py](app/providers/llm.py) | [tests/test_providers_gemini.py](tests/test_providers_gemini.py) |
| Metrics and load reporting | [app/metrics.py](app/metrics.py), [load_test](load_test) | [tests/test_metrics_state.py](tests/test_metrics_state.py), [tests/test_load_test.py](tests/test_load_test.py) |
| Event ordering and transcript retention | Session and client | [tests/test_session.py](tests/test_session.py), [tests/test_client.py](tests/test_client.py) |

- These checks cover successful turns, interruption, provider failure,
  cancellation, bounded output, duplicate sessions, and shutdown.
- Load tests also cover seeded assignment, per-agent aggregation, and
  isolation between concurrent calls.

## Metrics and latency budget

Per-turn metrics use a monotonic clock:

- **TTFT:** endpoint to first written `llm.token`.
- **TTFA:** endpoint to first written `audio.chunk`.
- **Total latency:** endpoint to terminal turn event.

- `GET /metrics` aggregates retained sessions; add `?session_id=...` to select
  one session.
- Percentiles use nearest rank over completed turns; the load runner reports
  completed turns from successful calls.
- TTFT and TTFA end at the first successfully written token or audio chunk;
  total latency ends at the terminal turn event.
- Failed and interrupted turns remain in outcome counts but not completed-turn
  latency percentiles.
- Durations use a monotonic clock. Aggregates include only sessions retained
  by the running process.

- The ~800 ms responsiveness target is speech-end to first audible response
  (TTFA), not full-answer completion. One possible budget is:

| Stage | Target |
|---|---:|
| Endpointing | 150 ms |
| Network/orchestration | 50 ms |
| LLM first token | 200 ms |
| First speakable sentence | 150 ms |
| TTS first chunk | 150 ms |
| Transport and contingency | 100 ms |
| **Total** | **≤800 ms** |

- The mock barge-in fixture uses explicit endpointing; its 350 ms silence
  timeout is for predictable local behavior, not production VAD.
- The budget is a design target, not a guarantee; provider and network times
  vary.

## Interfaces, state, and failure behavior

HTTP endpoints are `GET /health`, `GET /metrics`, and
`GET /metrics?session_id=...`. The WebSocket endpoint is
`WS /ws/{session_id}`; catalog clients select an agent with
`?scenario_id=travel`, while omission selects ecommerce.

- Client events: `session.start`, `audio.frame`, `audio.end`, `session.stop`,
  and `ping`. Server events: transcripts, `llm.token`, `audio.chunk`, terminal
  turn events, `error`, and `pong`.
- Recoverable schema errors emit `error`; fatal protocol errors close with
  `1008`, oversized input with `1009`, and unexpected failures with `1011`.
  First-token and generation deadlines produce one `turn.failed` event.
- Every event carries an event ID, session ID, and timezone-aware timestamp.
- Unknown catalog scenario IDs are rejected before session metrics or
  provider state are created.
- Conversation history, STT state, active turns, and queues belong to one
  `CallSession`; metrics are process-wide. There is no database, cache, broker,
  or scheduled job.

Check a running server with:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/metrics
```

## Configurable scenarios

The default catalog contains four deterministic mock scenarios:

| ID | Agent | Demonstration |
|---|---|---|
| `ecommerce` | ShopHelp | Explicit endpointing and refund barge-in |
| `travel` | TripAssist | Explicit and silence endpointing |
| `appointments` | ScheduleDesk | Morning-to-afternoon barge-in |
| `telecom` | NetHelp | Silence endpointing and troubleshooting |

- Each scenario defines the bot prompt, mock responses and token delay, caller
  turns, endpointing, optional barge-in delay, and simulated STT/silence/TTS
  timings.
- The strict Pydantic model rejects unknown fields, coercion, empty
  frames/responses, invalid delays, and first-turn barge-in settings.
- Scenario timing overlays prompt and simulation settings at startup. Host,
  port, queue sizes, message limits, and shutdown grace remain environment
  settings; JSON cannot override infrastructure limits.

- For turn N, `barge_in_after_ms` starts after the first `audio.chunk` from
  turn N-1. If that turn ends before the delay fires, the scripted call reports
  the missing interruption.
- `tts_chunk_delay_ms` maps to `tts_sentence_delay_ms`; each session receives
  a fresh mock provider.

Start the default catalog, choose another default agent, or start a single
scenario with:

```bash
python -m app --scenario-dir scenarios
python -m app --scenario-dir scenarios --default-scenario-id travel
python -m app --scenario-file scenarios/ecommerce.json
```

- Catalog mode scans only JSON files directly inside `scenarios/`, excluding
  nested Gemini fixtures from the default catalog and mock load test. Select a
  nested fixture explicitly with `--scenario-file`.
- The load runner accepts `--default-scenario-id` for a custom catalog whose
  default is not ecommerce. Tests and alternate provider factories can use
  `create_app()` directly.

- Host, port, queue sizes, message limits, shutdown grace, and safety timeouts
  are environment settings.
- Scenario files change prompts and simulation timings, not infrastructure
  limits. `Settings.from_environment()` rejects unknown `VOICE_*` names.

## Optional Gemini free-tier provider

- Copy `.env.example` to `.env`; it selects `gemini-3.5-flash-lite` and the
  Gemini provider. Add `VOICE_LLM_API_KEY` from
  [Google AI Studio](https://aistudio.google.com/app/apikey).
- For mock runs, leave Gemini unconfigured. If using the template, remove its
  Gemini provider and key settings.
- Startup checks the provider, nonblank model and API key, and base URL; Gemini
  validates the key and model on request.
- `.env` is Git-ignored. Exported variables take precedence; rotate any
  exposed key and keep credentials out of tracked files.

- **Two-turn barge-in demo:** the fixture interrupts 100 ms after the first
  sentence. Run the matching server and caller in separate terminals:

```bash
VOICE_LLM_PROVIDER=gemini python -m app --scenario-file scenarios/gemini/ecommerce.json
python -m client --scenario-file scenarios/gemini/ecommerce.json
```

- **Gemini load test:** start the server with the one-turn fixture, then run
  the load test in another terminal. It has no timed barge-in, so fast answers
  cannot finish before an interruption timer:

```bash
VOICE_LLM_PROVIDER=gemini python -m app --scenario-file scenarios/gemini/load-test.json
python -m load_test --scenario-file scenarios/gemini/load-test.json --concurrency 3 --seed 42
```

- Start with low concurrency and increase gradually; rate limits and latency
  can cause provider errors or timeouts.
- Deadlines are 500 ms to first token and 5 s for generation. Override them
  with `VOICE_FIRST_TOKEN_TIMEOUT_MS` and `VOICE_GENERATION_TIMEOUT_MS`.
- Slow responses fail with `first_token_timeout` or `generation_timeout`.
- Missing configuration (blank key/model or malformed base URL) fails at
  startup; an invalid/expired key or unavailable model fails on the Gemini
  request and is not retried.
- Transient failures retry before the first token (one retry after 250 ms by
  default). Configure zero to three retries with `VOICE_LLM_MAX_RETRIES` and
  the pause with `VOICE_LLM_RETRY_DELAY_MS`. Authentication and model errors
  are not retried; no retry happens after output starts.
- When retries fail, the service emits the configured
  `VOICE_LLM_FAILURE_MESSAGE` and `turn.failed`. A caller interruption cancels
  generation with `turn.interrupted` and reason `barge_in`.
- A timed barge-in test can fail if the previous response completes before
  its configured delay; use the separate one-turn fixture for load testing.
- The default base URL is
  `https://generativelanguage.googleapis.com/v1beta`; override it with
  `VOICE_LLM_BASE_URL`. The Gemini HTTP client has no separate timeout, so the
  turn deadlines bound how long the service waits.
- Free-tier limits and model availability can change. Prompts and responses
  may be used to improve Google products; avoid customer data, confidential
  information, and secrets. Review current
  [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing) and
  [API terms](https://ai.google.dev/gemini-api/terms-archive/terms_02_05_25)
  before sending sensitive information.

## Repository tour

| Path | Responsibility |
|---|---|
| [app/__main__.py](app/__main__.py) | Scenario startup and per-session runtime |
| [app/api](app/api) | HTTP/WebSocket routes and validation |
| [app/conversation](app/conversation) | Session state, turn lifecycle, cancellation, writer |
| [app/providers](app/providers) | Simulated STT, mock/Gemini LLMs, sentence TTS |
| [app/scenario.py](app/scenario.py) | Scenario models and catalog loader |
| [app/metrics.py](app/metrics.py) | Turn metrics and percentiles |
| [client](client) | Scripted calls and readable traces |
| [load_test](load_test) | Seeded concurrent calls and reports |
| [scenarios](scenarios) | Mock and explicit Gemini fixtures |
| [tests](tests) | Protocol, concurrency, provider, and load tests |

## Production readiness and safety boundaries

The service is adapter-friendly but intentionally single-process, in-memory,
and text-based:

| Current simulation | Production direction |
|---|---|
| Text caller frames | PCM/Opus or telephony media |
| Simulated STT | Streaming STT with VAD and transcript stabilization |
| Mock/Gemini LLM | Hosted or local streaming provider |
| Sentence text in `audio.chunk` | Streaming TTS audio |
| Scripted caller | Browser/WebRTC or telephony gateway |

- Session ownership, endpointing, cancellation, stale-output fencing, ordered
  writes, and latency measurement remain orchestration responsibilities.
- Replacing text fixtures with real media requires coordinating playback,
  VAD, transport buffering, and cancellation across those adapters.

### Scale and reliability

A production service would add:

- OpenTelemetry/Prometheus metrics, centralized logs, and trace propagation.
- Authentication, tenant-aware quotas, rate limits, and horizontal scaling.
- Provider circuit breakers, deadlines, load shedding, and backpressure.
- Managed credential storage and rotation.
- Explicit session retention, reconnect state, and external coordination where
  needed.
- Production audio, streaming STT/TTS, and measured voice activity detection.

- Horizontal workers need connection affinity for live WebSockets and
  external coordination if reconnects must resume calls.
- Preserve retry-before-first-output behavior while adding provider circuit
  breakers and load shedding for quotas and queue pressure.

### Safety and guardrails

The assessment already uses strict event/scenario schemas, message limits,
bounded queues, provider deadlines, and logs that exclude transcripts and
secrets. Prompts do not claim actions the service cannot perform. Production
use would also require:

- treating transcripts, retrieved content, and model output as untrusted,
  including prompt-injection and instruction-conflict testing;
- minimizing collection, redacting logs/traces, encrypting data in transit and
  at rest, and applying tenant-specific retention/deletion policies;
- moderating user input and generated output before delivery or tool use;
- enforcing authenticated tenant boundaries around sessions, metrics, and
  provider credentials;
- allowlisting tools and validating and authorizing each action's parameters;
- applying per-user and per-tenant abuse controls, quotas, and rate limits; and
- requiring confirmation, idempotency, and audit records for consequential
  actions such as issuing refunds or changing bookings.

This assessment does not execute external tools or consequential actions.

## Use of AI tools

- Used OpenAI Codex for architecture discussion, concurrency analysis, test
  refinement, and documentation review.
- Manually reviewed the changed implementation and traced cancellation,
  provider, metrics, and configuration paths.
- Ran the relevant focused tests and the full local test suite.
- Verified the documented startup, client, metrics, load-test, and Gemini
  smoke-test workflows locally.
- Kept credentials out of tracked documentation and source files.

## Verification

- Run `pytest -q` for the full suite. It covers scenario validation, protocol
  policy, endpointing and barge-in, LLM/TTS overlap, Gemini request mapping and
  retries, cancellation cleanup, metrics, catalog selection, reporting, seeded
  load assignment, duplicate sessions, and shutdown behavior.

- Focus scenario, configuration, and load checks with
  `pytest -q tests/test_scenario.py tests/test_load_test.py tests/test_config_events.py`.
- Gemini provider unit tests use mocked HTTP transport; the separate Gemini
  load command above sends live requests through the configured server.

- Mock startup, the scripted caller, metrics endpoints, and 20-call load
  command are shown above. Repeat the seeded load command to confirm agent
  assignment; latency percentiles vary with machine and event-loop scheduling.
