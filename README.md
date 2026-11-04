# Voice Agent Service

A locally runnable orchestration layer for a streaming voice bot. Text chunks
stand in for audio, but the difficult parts are real: partial and final STT,
streaming LLM tokens, sentence-level TTS pipelining, caller interruption,
bounded queues, task cancellation, event ordering, and latency measurement.

The default mock provider requires no credentials, microphone, audio device,
or external service. An optional Gemini adapter exercises the same streaming
and interruption path with a real hosted model. The default catalog contains
four deterministic mock agents; the separate Gemini fixture is excluded from
catalog startup and load testing. Ecommerce remains the default agent.

## Review this project in five minutes

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start the service:

```bash
python -m app --scenario-dir scenarios
```

In another terminal, run the scripted caller:

```bash
source .venv/bin/activate
python -m client --scenario-file scenarios/ecommerce.json
```

Without `VOICE_LLM_PROVIDER=gemini` configured in the environment or `.env`,
the service defaults to the deterministic mock provider. The trace demonstrates
partial/final transcripts, word-by-word LLM output,
TTS starting before generation finishes, and a guaranteed barge-in. The first
bot answer is interrupted 50 ms after its first audio chunk; the second caller
turn changes the request from replacement to refund. Under the default mock
configuration, these startup, client, and load-test commands use deterministic
responses and make no provider requests.

Then run the required 20-call load check and the test suite:

```bash
python -m load_test --scenario-dir scenarios --concurrency 20 --seed 42
pytest -q
```

The load command distributes calls reproducibly across the four mock agents. It
reports requested/successful calls, unique sessions, actual peak concurrency,
turn outcomes, and completed-turn p50/p95 latency overall and per agent.
`load_test` connects to the already-running server and does not select its LLM
provider. If `.env` was copied from `.env.example`, it selects Gemini; change
`VOICE_LLM_PROVIDER` in `.env` to `mock` before starting the server for a
deterministic load run. Leave it set to `gemini` only when intentionally
load-testing the live provider, where provider latency, rate limits, and
timeouts can cause calls or scripted barge-ins to fail.

The shortest useful code-reading path is:

1. [app/api/server.py](app/api/server.py) — HTTP/WebSocket entry points and
   per-session dependency construction.
2. [app/conversation/call_session.py](app/conversation/call_session.py) — the
   session controller, endpointing, barge-in, output queue, and writer.
3. [app/conversation/turn_pipeline.py](app/conversation/turn_pipeline.py) — the
   concurrent LLM producer and sentence-TTS consumer.
4. [app/conversation/turn_state.py](app/conversation/turn_state.py) — minimal,
   single-assignment turn state.
5. [tests/test_barge_in.py](tests/test_barge_in.py) — the concurrency invariants
   expressed as executable tests.

## What successful output looks like

The normal CLI is intentionally readable rather than raw JSON:

```text
SHOPHELP · ecommerce · session call-...
────────────────────────────────────────────────

TURN 1
STT …   Hi, my order arrived damaged.
STT ✓   Hi, my order arrived damaged. I would like replacement options.
LLM     I am sorry your order arrived damaged.
TTS ▶   “I am sorry your order arrived damaged.”
LLM     I can help you review…

⚡ BARGE-IN · turn 2 started 50 ms after first audio
RESULT  INTERRUPTED · barge_in
LATENCY TTFT ... · TTFA ... · total ...
```

Consecutive LLM tokens stay on a flowing line; TTS and terminal events break
the line at their true arrival position. The final report shows caller text,
generated text, numbered spoken chunks, outcome, reason, and latency per turn.
Interrupted output is labelled partial. Interactive terminals use restrained
color; redirected output and `NO_COLOR=1` stay plain. `--verbose` switches to
machine-readable JSON Lines.

The load report explicitly separates call health from turn behavior and uses
only completed turns from successful calls for latency percentiles:

```text
LOAD TEST · seed 42
CALLS     requested 20 · succeeded 20 · failed 0
SESSIONS  unique 20 · peak concurrency 20
TURNS     completed ... · interrupted ... · failed ...

COMPLETED-TURN LATENCY · OVERALL
Metric  Samples  p50       p95
...

AGENTS
appointments
  Calls  requested ... · succeeded ... · failed ...
  Turns  completed ... · interrupted ... · failed ...
  ...
```

The important correctness assertion is stronger than the display: after
`turn.interrupted`, any old-turn `llm.token` or `audio.chunk` fails the client
run.

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

One `CallSession` owns one connection and all tasks created for it. The
controller waits for the next WebSocket event, application shutdown, writer
failure, or the current silence deadline. There is no receiver-loop /
orchestrator split and no separate silence-timer task.

When STT produces a final transcript, the session starts an
`ActiveTurnRunner`. Its `TaskGroup` contains two children connected by a
bounded token queue:

- the LLM producer emits each token immediately and continues generating;
- the TTS consumer builds complete sentences and speaks each sentence after
  its simulated synthesis delay.

This is genuine overlap: while TTS waits on an earlier sentence, the LLM can
continue producing later tokens.

All server events use one bounded FIFO queue and one writer. The writer is the
only task that sends on the WebSocket, which makes event ordering and latency
measurement easy to reason about.

## Turn flow and barge-in

A normal explicit-endpointed turn looks like this:

```text
session.start
  → session.ready
audio.frame(s)
  → transcript.partial(s)
audio.end
  → transcript.final
  → llm.token(s)
  → audio.chunk(s) as sentences become ready
  → turn.completed
session.stop
```

Silence endpointing follows the same path, except the controller finalizes STT
when its current silence deadline expires.

An incoming caller frame interrupts an active bot turn before that frame is
added to STT:

```text
new audio.frame
  → claim the old turn as interrupted
  → invalidate old-turn output
  → cancel the old turn task
  → enqueue turn.interrupted
  → process the new caller frame
```

The input path never waits for provider cleanup. Cancellation is cooperative,
so correctness does not rely on it happening immediately. The implementation
uses two output fences:

1. the turn pipeline stops enqueueing after the turn becomes invalid;
2. the writer drops already queued tokens/audio whose turn is invalid.

Terminal state is single-assignment, so a completion racing with an interrupt
cannot produce two terminal events. The session still owns cancelled turn tasks
and gathers remaining work during shutdown.

The session cancels a turn as soon as caller speech arrives, but interruption
is currently sentence-level in this text-audio simulation: each `audio.chunk`
represents a complete sentence, so the demo cannot model cutting off audio
mid-sentence. When the service moves to real audio, the TTS adapter should
split synthesized audio into smaller chunks and stop playback or discard
buffered chunks when caller speech arrives. That will support interruption at
audio-chunk boundaries.

## Design choices and tradeoffs

| Choice | Why | Tradeoff |
|---|---|---|
| One `CallSession` per WebSocket | Keeps protocol state, STT, turn state, history, and task ownership together | Session state is in-memory and connection-scoped |
| One controller | Serializes caller input and turn transitions without locks | Slow work must be moved into owned tasks |
| `TaskGroup` per turn | Gives the LLM/TTS pair structured cancellation and cleanup | A turn is the smallest independent pipeline unit |
| Bounded token and output queues | Prevents unbounded memory growth and provides natural backpressure | A slow consumer can delay producers |
| One writer | Prevents concurrent socket writes and provides one ordering/metrics boundary | All outbound events share the same FIFO |
| Cancellation plus turn-ID fencing | Prevents stale output even when a provider cleans up slowly | In-flight bytes already being sent cannot be recalled |
| Fresh mock provider per session | Gives concurrent calls independent response cursors | Mock conversation state is not shared across reconnects |
| Fresh Gemini provider per session | Reuses one HTTP client across a call and closes it with the session | Each call owns a separate connection pool |
| Retry only before first output | Recovers one transient provider failure without duplicating caller-visible output | Failures after partial output terminate the turn without retry |
| Strict shared scenario catalog | Server, client, and load runner share validated agent IDs | Server and runner must load matching catalogs |
| Seeded random load assignment | Repeated runs exercise the same agent distribution | It is deterministic test randomness, not production routing |
| Monotonic in-memory metrics | Produces reliable durations without infrastructure | Metrics reset when the process restarts |

The assessment intentionally does not implement reconnect continuity,
persistent metrics, or real media. Gemini demonstrates the external-provider
boundary; the remaining items are deployment concerns rather than
prerequisites for demonstrating the core streaming and interruption behavior.

## Async task ownership

Every spawned task has an owner and an explicit cleanup path.

| Task | Reason it exists | Cancellation and cleanup |
|---|---|---|
| Pending WebSocket receive | Lets the controller race input against silence and shutdown | Replaced after each event; cancelled and gathered during cleanup |
| Session shutdown waiter | Wakes the controller when the application is stopping | Cancelled and gathered with the receive task |
| Socket writer | Serializes every transport send | Queue drains normally; a stalled writer is cancelled after shutdown grace |
| Turn task | Owns one bot response independently of caller input | Cancelled on barge-in/session end and tracked until done |
| LLM producer | Streams provider tokens while TTS handles earlier text | Owned by the per-turn `TaskGroup` |
| TTS consumer | Segments tokens and emits sentence audio | Owned by the same `TaskGroup`; sibling failure cancels it |
| Gemini HTTP client | Reuses connections across turns in one call | Owned and closed by the `CallSession` provider lifecycle |
| Client barge timer and receive | Keeps consuming server events during the configured delay | Losing task is cancelled; both are gathered in `finally` |
| Load-call coroutines | Run isolated sessions concurrently | `gather(return_exceptions=True)` retrieves every result so one failure does not cancel siblings |

Application shutdown requests every live session to stop, waits up to the
configured grace period, then force-cancels remaining session work. The session
records its final internal metrics and closes the transport even when the
writer was stalled.

## Assessment requirements and evidence

| Requirement | Implementation | Primary evidence |
|---|---|---|
| FastAPI WebSocket and scripted CLI | [app/api/server.py](app/api/server.py), [app/api/events.py](app/api/events.py), [client](client) | [tests/test_api.py](tests/test_api.py), [tests/test_client.py](tests/test_client.py) |
| Partial/final STT and silence endpointing | [app/providers/stt.py](app/providers/stt.py), `CallSession` | [tests/test_stt.py](tests/test_stt.py), [tests/test_session.py](tests/test_session.py) |
| Streaming provider interface and mock LLM | [app/providers/llm.py](app/providers/llm.py), `ActiveTurnRunner` | [tests/test_turn_pipeline.py](tests/test_turn_pipeline.py) |
| Sentence TTS before generation finishes | [app/providers/tts.py](app/providers/tts.py), `ActiveTurnRunner` | overlap test in [tests/test_turn_pipeline.py](tests/test_turn_pipeline.py) |
| Correct barge-in and no stale output | `CallSession`, `TurnState`, writer fence | [tests/test_barge_in.py](tests/test_barge_in.py) |
| TTFT, TTFA, total latency, p50/p95 | [app/metrics.py](app/metrics.py), `GET /metrics` | [tests/test_metrics_state.py](tests/test_metrics_state.py) |
| 20 concurrent simulated calls | [load_test](load_test) | [tests/test_load_test.py](tests/test_load_test.py) |
| Event ordering and final-transcript retention | Session and client tests | [tests/test_session.py](tests/test_session.py), [tests/test_client.py](tests/test_client.py) |

## Metrics and latency budget

Per-turn metrics use a monotonic clock:

- **TTFT** — endpoint to the first successfully written `llm.token`;
- **TTFA** — endpoint to the first successfully written `audio.chunk`;
- **total turn latency** — endpoint to the terminal outcome;
- successfully written token/audio counts and the terminal outcome.

`GET /metrics?session_id=...` returns one session, while `GET /metrics` combines
the sessions currently retained by the process. Percentiles use nearest-rank
p50/p95 over completed turns. The load runner independently aggregates the
completed turns from successful calls.

For conversational responsiveness, the meaningful ~800 ms target is speech-end
to first audible response (TTFA), not completion of an arbitrarily long bot
answer. A production budget could be allocated as follows:

| Stage | Target |
|---|---:|
| Endpoint detection / VAD | 150 ms |
| Network and orchestration | 50 ms |
| LLM first token | 200 ms |
| Tokens needed for the first speakable sentence | 150 ms |
| TTS first chunk | 150 ms |
| Output transport and playout | 50 ms |
| Contingency | 50 ms |
| **TTFA target** | **≤800 ms** |

The deterministic fixture uses explicit endpointing for its barge-in demo. Its
350 ms silence timeout favors predictable local behavior, not a production VAD
target. Optimization effort should go first to endpoint detection, then LLM
first-token latency, earlier clause/sentence segmentation, and TTS startup.
Serialization and queue overhead should be optimized only after measuring them.

## Interfaces, state, and failure behavior

HTTP endpoints:

- `GET /health` — process health;
- `GET /metrics` — aggregate in-memory metrics;
- `GET /metrics?session_id=...` — one session's metrics.

The WebSocket endpoint is `WS /ws/{session_id}`. Catalog clients select an
agent with the optional `?scenario_id=travel` query; omission uses ecommerce.
An unknown ID closes with `1008` before provider or metric state is created.
Every event contains a type, event ID, session ID, and timezone-aware timestamp.

| Direction | Event types |
|---|---|
| Client → server | `session.start`, `audio.frame`, `audio.end`, `session.stop`, `ping` |
| Server → client | `session.ready`, transcript partial/final, `llm.token`, `audio.chunk`, terminal turn events, `error`, `pong` |

Malformed JSON and schema errors produce recoverable `error` events. Fatal
protocol violations close with `1008`; oversized input uses `1009`; unexpected
internal failure uses `1011`; normal stop uses `1000`. First-token and total
generation safety timeouts produce one `turn.failed` outcome.

Conversation history, STT state, active turns, and queues belong to one
`CallSession`. Metrics live in the process-level registry. There is no database,
cache, broker, scheduled job, or persistent state.

Inspect a running service with:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/metrics
curl -fsS 'http://127.0.0.1:8000/metrics?session_id=<session-id>'
```

## Configurable interview scenarios

The default `scenarios/` catalog contains four deterministic mock scenarios:

| ID | Agent | Demonstrates |
|---|---|---|
| `ecommerce` | ShopHelp | Explicit endpointing and a guaranteed refund barge-in |
| `travel` | TripAssist | Explicit then silence endpointing for a flight-change request |
| `appointments` | ScheduleDesk | A guaranteed morning-to-afternoon barge-in |
| `telecom` | NetHelp | Silence endpointing followed by normal troubleshooting |

Each JSON scenario configures:

- bot name and system prompt;
- ordered mock responses and token delay;
- caller frames, explicit/silence endpointing, and optional barge-in delay;
- simulated STT, silence, and TTS timings.

The Pydantic model in [app/scenario.py](app/scenario.py) forbids unknown fields
and coercion, rejects empty frames/responses, validates delays, and rejects a
barge-in delay on the first turn.

For caller turn N, `barge_in_after_ms` means: wait for the first `audio.chunk`
from turn N-1, keep receiving events for the configured delay, then send turn N.
If the previous turn terminates before the timer fires, the run fails rather
than silently turning into a normal follow-up.

The scenario owns the prompt, mock responses, token delay, STT delay, silence
timeout, and TTS delay. `tts_chunk_delay_ms` maps to the runtime setting
`tts_sentence_delay_ms`. A fresh `MockLLMProvider` is built for every session,
so concurrent calls never share response cursors.

Catalog mode is the server default. Single-agent mode remains available and
accepts only an omitted or matching `scenario_id`:

```bash
python -m app --scenario-dir scenarios
python -m app --scenario-dir scenarios --default-scenario-id travel
python -m app --scenario-file scenarios/ecommerce.json
```

The load runner accepts the same `--default-scenario-id` when exercising a
custom catalog that does not contain ecommerce.

The Gemini live barge-in fixture is stored at
[`scenarios/gemini/ecommerce.json`](scenarios/gemini/ecommerce.json); the
single-turn concurrency fixture is
[`scenarios/gemini/load-test.json`](scenarios/gemini/load-test.json). Catalog
and load-test catalog mode scan only JSON files directly inside `scenarios/`,
so both fixtures are excluded from the default four-agent catalog and mock
load test. Select either fixture explicitly with `--scenario-file`.

Host, port, queue sizes, message limits, shutdown grace, and safety timeouts
remain environment-owned:

```bash
VOICE_PORT=9000 VOICE_OUTPUT_QUEUE_SIZE=256 \
python -m app --scenario-dir scenarios
```

`Settings.from_environment()` accepts only known `VOICE_*` names. Scenario
values overlay the prompt and simulation timing fields when the CLI starts the
server; infrastructure limits cannot be overridden by JSON. Programmatic
`create_app()` remains available for focused tests or alternate provider
factories without loading a scenario file.

## Optional Gemini free-tier provider

The default startup and load-test commands above use the mock provider unless
Gemini is configured in the environment or `.env`. For a live Gemini call,
create a key in
[Google AI Studio](https://aistudio.google.com/app/apikey), copy the tracked
template to the ignored local `.env`, and add the key there:

```bash
cp .env.example .env
# Edit .env and set VOICE_LLM_API_KEY to the newly generated key.
```

The template sets `VOICE_LLM_PROVIDER=gemini` and the default model
`gemini-3.5-flash-lite`. Local settings validation restricts the provider to
`mock` or `gemini`, requires a nonblank model and API key for Gemini, and
validates the base URL. Gemini confirms whether the key and model are accepted
when the smoke request or live call is made. The server loads `.env` before
settings are constructed. Existing exported process variables take
precedence. `.env` is ignored by Git; keep credentials there or in environment
variables, never in tracked files. Revoke and rotate a key immediately if it
is exposed, then update the local `.env` value.

Start the server in one terminal with the nested Gemini fixture:

```bash
source .venv/bin/activate
VOICE_LLM_PROVIDER=gemini python -m app --scenario-file scenarios/gemini/ecommerce.json
```

Run the matching two-turn live caller from another terminal:

```bash
source .venv/bin/activate
python -m client --scenario-file scenarios/gemini/ecommerce.json
```

For Gemini concurrency testing, keep the Gemini server running and use the
load-test fixture explicitly. It uses one caller turn without a timed
barge-in, so a fast response cannot make the call fail by completing before
the interruption timer. Start a server for this fixture in one terminal:

```bash
VOICE_LLM_PROVIDER=gemini python -m app --scenario-file scenarios/gemini/load-test.json
```

Then run the load test from another terminal. This does not add Gemini to the
default catalog or mock load test:

```bash
python -m load_test --scenario-file scenarios/gemini/load-test.json --concurrency 3 --seed 42
```

This sends live requests through the Gemini provider configured on the server.
Start with low concurrency and increase it gradually; higher concurrency can
hit provider rate limits or timeouts. Use
`scenarios/gemini/ecommerce.json` with `python -m client` when you want to
exercise its 100 ms Gemini barge-in behavior.

The live fixture keeps its barge-in delay at 100 ms to exercise interruption
while Gemini is generating. For a credential and streaming-endpoint smoke
test, run this after creating `.env`; it sends a generic prompt and does not
place the key in the command or tracked files:

```bash
set -a
. ./.env
set +a
VOICE_LLM_BASE_URL="${VOICE_LLM_BASE_URL:-https://generativelanguage.googleapis.com/v1beta}"
curl -N -fsS \
  -H "x-goog-api-key: ${VOICE_LLM_API_KEY}" \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  "${VOICE_LLM_BASE_URL}/models/${VOICE_LLM_MODEL}:streamGenerateContent?alt=sse" \
  -d '{"contents":[{"role":"user","parts":[{"text":"Reply with the word OK."}]}]}'
```

`VOICE_LLM_BASE_URL` can replace the default base URL for a compatible test
endpoint. The service applies a 500 ms first-token safety timeout and a 5 s
generation timeout; `VOICE_FIRST_TOKEN_TIMEOUT_MS` and
`VOICE_GENERATION_TIMEOUT_MS` change those values. The Gemini HTTP stream itself
has no separate client timeout, so the service-level turn deadlines govern
waiting for output.

`gemini-3.5-flash-lite` is the documented default because it is currently
available on the Gemini free tier and is optimized for cost-efficient work.
Free-tier quotas, availability, and model names can change; verify the current
[Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing) before a demo.
The base URL defaults to
`https://generativelanguage.googleapis.com/v1beta` and can be overridden with
`VOICE_LLM_BASE_URL` for a compatible test endpoint.

The adapter retries one transient failure before any token is emitted, waiting
250 ms by default. `VOICE_LLM_MAX_RETRIES` accepts `0` through `3`, and
`VOICE_LLM_RETRY_DELAY_MS` changes the delay. Authentication and invalid-model
errors are not retried. Once output starts, a provider error fails the turn
without retry so callers never receive duplicated text or speech.

An invalid/expired key or unavailable model produces an authentication or
model error from the provider; those errors are not retried. A missing or blank
key, blank model, or malformed base URL is rejected during local configuration
validation. Rate limits and transient provider/network errors use the configured
retry policy; if they persist, the turn emits the configured provider-failure
message. A slow response can end in `first_token_timeout` or
`generation_timeout`. A caller speaking during generation cancels the old
turn and emits `turn.interrupted` with `reason="barge_in"`; the nested fixture
deliberately tests this path at 100 ms.

If Gemini remains unavailable before producing output, the service streams and
speaks “I’m having trouble connecting right now. Please try again in a
moment.”, then emits `turn.failed` with `reason="provider_error"`. Override the
text with `VOICE_LLM_FAILURE_MESSAGE`. Because the turn failed, this message is
not committed to successful conversation history.

Gemini's free tier may use submitted prompts and generated responses to improve
Google products. Do not send real customer PII, confidential material, secrets,
or production conversations through this demo. Review the current
[Gemini API terms](https://ai.google.dev/gemini-api/terms-archive/terms_02_05_25)
before use.

## Repository tour

| Path | Responsibility |
|---|---|
| [app/__main__.py](app/__main__.py) | Catalog/single-file startup and per-session runtime factory |
| [app/api](app/api) | FastAPI routes, event schemas, WebSocket validation/transport |
| [app/conversation](app/conversation) | Session ownership, turn lifecycle, concurrency, barge-in, writer |
| [app/providers](app/providers) | Simulated STT, streaming mock and Gemini LLMs, sentence segmentation |
| [app/scenario.py](app/scenario.py) | Strict scenario model and validated catalog loader |
| [app/metrics.py](app/metrics.py) | Per-turn records and nearest-rank p50/p95 aggregation |
| [client](client) | Scenario execution, ordered event observation, trace and report |
| [load_test](load_test) | Seeded agent assignment, concurrent calls, and per-agent aggregation |
| [scenarios](scenarios) | Deterministic interview fixtures |
| [tests](tests) | Protocol, concurrency, failure, reporting, and load evidence |

## Production readiness and safety boundaries

The orchestration boundary is intentionally adapter-friendly:

| This project | Production equivalent |
|---|---|
| Text in `audio.frame` | PCM/Opus or telephony audio from WebRTC, SIP, or Twilio Media Streams |
| `SimulatedSTT` partial/final text | Deepgram streaming STT or Whisper streaming with VAD and transcript stabilization |
| `MockLLMProvider` tokens | `GeminiLLMProvider` today; other hosted or local streaming adapters can use the same interface |
| Text in `audio.chunk` | Streaming TTS audio, commonly PCM or mu-law for telephony |
| JSON scripted caller | Browser/WebRTC client or bidirectional VoIP media gateway |

Replacing adapters does not change the central responsibilities: session
ownership, endpointing decisions, turn IDs, structured cancellation,
stale-output fencing, one ordered writer, and latency measurement.

### Scale and reliability

This assessment is intentionally single-process and in-memory. A production
deployment would also need to:

- export bounded-cardinality service and turn metrics through OpenTelemetry
  collectors and a Prometheus scrape endpoint instead of relying on the
  in-process registry;
- authenticate callers and services, enforce tenant-aware rate limits and
  quotas, and reject unauthenticated WebSocket upgrades;
- scale horizontally behind a load balancer that keeps each live WebSocket on
  one worker, with external coordination when reconnects or cross-node
  continuity are required;
- retain the existing retry-before-first-output rule while adding provider
  circuit breakers, deadlines, load shedding, and queue-based backpressure;
- store conversation state in a bounded external store only when reconnects
  must resume a call, rather than assuming process-local `CallSession` state;
- send structured logs to a centralized system and propagate trace context
  across the media gateway, orchestration service, and external providers;
- load credentials from a managed secret store, rotate them, and keep them out
  of source control, client-visible errors, logs, and traces;
- replace text frames with PCM/Opus or telephony media, streaming STT/TTS, and
  a production voice-activity detector with measured endpointing behavior; and
- define session retention and eviction explicitly, using bounded TTL/LRU
  policies in memory or lifecycle policies in the external state store.

### Safety and guardrails

The current assessment already uses strict event and scenario schemas,
message-size limits, bounded queues, provider timeouts, and sanitized lifecycle
logging that excludes transcripts and provider secrets. Its scenario prompts
also avoid claiming that a refund, booking change, or appointment was completed
when the service cannot perform that action.

Those controls are not a complete production safety layer. A production
service would additionally need to:

- treat caller transcripts, retrieved content, and provider output as
  untrusted input; isolate system instructions and test prompt-injection and
  instruction-conflict cases;
- minimize PII collection, redact it from logs and traces, encrypt it in
  transit and at rest, and enforce tenant-specific retention and deletion
  policies;
- moderate caller input and generated text/audio before it is delivered or
  used to invoke another system;
- enforce authenticated tenant boundaries on session state, metrics, traces,
  provider credentials, and any future tool results;
- allowlist tools and actions, validate their parameters, and authorize each
  invocation against the caller's and tenant's scopes;
- detect abuse and apply per-identity and per-tenant rate limits, quotas, and
  escalation or blocking policies; and
- require explicit user confirmation, action-specific authorization,
  idempotency controls, and an audit record before consequential operations
  such as issuing a refund or changing a booking.

This assessment does not execute external tools or consequential actions; it
only explains the next step a production integration could take.

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

The full suite covers scenario validation, protocol policy, explicit/silence
endpointing, genuine LLM/TTS overlap, Gemini request mapping and retries,
spoken provider-failure handling, cancellation cleanup, bounded output,
catalog selection, readable reporting, seeded load assignment,
duplicate-session protection, stalled-writer shutdown, and load isolation.

The manually verified acceptance commands are:

```bash
python -m app --scenario-dir scenarios
python -m client --scenario-file scenarios/ecommerce.json
python -m load_test --scenario-dir scenarios --concurrency 20 --seed 42
```

Run the seeded load command twice to confirm the same per-agent assignment
counts. Exact latency values depend on the machine and event-loop scheduling,
so the runner reports measured p50/p95 rather than asserting fixed numbers.
