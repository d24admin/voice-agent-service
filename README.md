# Voice Agent Service

A locally runnable real-time voice orchestration assessment: simulated streaming
STT feeds a streaming mock LLM, tokens feed sentence-level simulated TTS before
generation finishes, and barge-in cancels obsolete work without leaking stale
audio. Audio is represented by text so the concurrency, latency, endpointing,
backpressure, and interruption behavior can run without credentials or media
hardware.

## Quick start

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start the service:

```bash
python -m app
```

In another terminal, run either deterministic call:

```bash
source .venv/bin/activate
python -m client --scenario default
python -m client --scenario barge_in
```

The default scenario uses both explicit `audio.end` and silence endpointing.
The barge-in scenario waits for the first bot `audio.chunk`, interrupts it, and
fails if any old-turn audio appears after `turn.interrupted`. Add `--verbose`
for machine-readable JSON Lines.

Run 20 isolated calls concurrently:

```bash
python -m load_test
```

Override load options when needed:

```bash
python -m load_test --concurrency 20 --scenario barge_in --server-url ws://127.0.0.1:8000
```

Run the complete test suite:

```bash
pytest -v
```

Inspect the service after a demo:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/metrics
curl -fsS 'http://127.0.0.1:8000/metrics?session_id=<session-id>'
```

Runtime settings use explicit `VOICE_` variables. For example:

```bash
VOICE_PORT=9000 VOICE_SILENCE_TIMEOUT_MS=250 python -m app
```

Unknown `VOICE_` settings and invalid bounds fail at startup instead of being
silently ignored.

## Architecture

```text
JSON WebSocket
      |
      v
+------------------------ CallSession -------------------------+
| receiver -> bounded inbound queue -> orchestrator            |
|                                      |                       |
|                              simulated STT + silence timer    |
|                                      | final transcript       |
|                                      v                       |
|                        +------ active turn ------+            |
|                        | LLM producer            |            |
|                        |       | bounded tokens  |            |
|                        |       v                 |            |
|                        | sentence TTS consumer   |            |
|                        +-------------|-----------+            |
|                                      v                       |
| ordered data + reserved lifecycle -> single writer -> socket |
| cancelled supervisors -------------> turn reaper             |
+--------------------------------------------------------------+
```

One `CallSession` owns every task and queue for a connection. STT state changes
are serialized by the orchestrator. Each turn has one supervisor with exactly
two `TaskGroup` children: the LLM producer and TTS consumer. Their bounded token
queue permits the producer to continue while TTS sleeps on an earlier sentence,
which is actual LLM/TTS overlap rather than merely reordered output.

The outbound buffer has a bounded ordered-data lane and a reserved lifecycle
lane. Every turn reserves terminal capacity before generation starts. All
outbound items receive one enqueue sequence, so a lifecycle event acts as an
ordering barrier: earlier non-stream events remain ordered, invalid old-turn
tokens/audio are dropped, and higher-sequence data cannot jump ahead.

### Example event flow

```text
client                         server
  |-- session.start ----------->|
  |<------------- session.ready |
  |-- audio.frame "hello" ---->|
  |<-------- transcript.partial |
  |-- audio.frame "there" ---->|
  |-- audio.end --------------->|
  |<---------- transcript.final |
  |<----------------- llm.token |  generation continues
  |<--------------- audio.chunk |  first sentence already speaks
  |<----------------- llm.token |
  |<--------------- audio.chunk |
  |<------------- turn.completed|
  |-- session.stop ------------>|
```

During barge-in, the orchestrator performs four non-awaiting operations in the
same event-loop turn: claim `interrupted`, invalidate the old output fence,
cancel and hand off the supervisor, and enqueue `turn.interrupted`. It then
accepts the triggering caller frame. The reaper waits for provider cleanup so
new STT work never blocks on a cancellation-resistant adapter.

## Task ownership and cancellation

Every `asyncio.create_task` has one owner and one terminal path.

| Task | Why it exists | Normal completion | Cancellation/cleanup |
|---|---|---|---|
| Receiver | Keep reading while orchestration and cleanup proceed | `session.stop` or peer disconnect | `CallSession.run()` cancels and awaits it |
| Orchestrator | Serialize protocol, STT, endpointing, and barge-in transitions | Stop, fatal policy, or shutdown request | Session cleanup cancels it and the silence timer |
| Writer | Be the only transport writer; apply sequence/fence policy; stamp observed metrics | Outbound closes | Session cleanup cancels and awaits it |
| Silence timer | Endpoint one versioned utterance after inactivity | Enqueues one `SilenceExpired` | Replaced by a frame, explicit end, or shutdown |
| Active-turn supervisor | Own one LLM/TTS `TaskGroup` | Both pipeline children finish | Invalidated and cancelled on barge-in/session end, then transferred to reaper |
| LLM producer | Stream provider tokens while TTS handles earlier text | Marks generation closed and queues internal end-of-stream | Supervisor/`TaskGroup` cancellation; provider iterator is closed in `finally` |
| TTS consumer | Segment live tokens and emit sentence chunks | Flushes trailing text and queues output-complete | Same `TaskGroup` cancellation; buffered obsolete text is discarded |
| Turn reaper | Await fenced cancelled supervisors outside caller-input path | Retired task releases resources | Session shutdown cancels it after all owned tasks are cancelled |
| Load-call coroutine | Keep each simulated call isolated | One `CallResult` | `gather(return_exceptions=True)` retrieves every result independently |

Cancellation is cooperative, but correctness does not depend on prompt provider
cancellation. A monotonically increasing turn ID and `output_valid` fence stop
late tokens from entering TTS; the writer independently drops already queued
old-turn tokens and audio. Terminal events are unfenced and emitted at most
once through the turn's reserved lifecycle slot.

## Protocol and metrics

The WebSocket endpoint is `WS /ws/{session_id}`. Every event contains `type`,
`event_id`, `session_id`, and a timezone-aware timestamp. Supported client
events are `session.start`, `audio.frame`, `audio.end`, `session.stop`, and
`ping`. Server events are `session.ready`, transcript partial/final, `llm.token`,
`audio.chunk`, terminal turn events, `error`, and `pong`.

Malformed JSON/schema errors are recoverable. Session-ID mismatches, audio
before start, and duplicate start are fatal policy violations. Normal stop uses
close code `1000`, fatal policy uses `1008`, oversized input uses `1009`, and
overload uses `1013`.

Per-turn metrics use a monotonic clock and include:

- TTFT: endpoint to the first successfully written `llm.token`.
- TTFA: endpoint to the first successfully written `audio.chunk`.
- Total latency: endpoint to terminal-event dispatch.
- Actual successfully written token/audio counts and terminal outcome.

The writer stamps TTFT/TTFA only after transport send succeeds. Missing values
on interrupted or failed turns serialize as `null`, never zero. `/metrics`
uses nearest-rank p50/p95 over completed-turn samples. Session records remain
available in a bounded in-memory LRU after disconnect.

## Failure behavior

| Failure | Behavior |
|---|---|
| Invalid configuration | Startup fails with a concise validation error |
| Malformed/unsupported event | Typed recoverable `error`; connection remains usable |
| Fatal protocol violation | Best-effort fatal `error`, terminalize open turn, close `1008` |
| Oversized input | Close `1009` without retaining the payload |
| Inbound/lifecycle backpressure | Bounded queues remain bounded; typed overload and close `1013` |
| First-token/generation timeout | Invalidate unsent stream output and emit one `turn.failed` |
| Provider error after visible output | Preserve written events, fence queued output, no replay, emit `turn.failed` |
| Caller barge-in | Atomically interrupt/fence old turn, accept the same caller frame, reap cleanup separately |
| Delayed provider cancellation | No late token reaches TTS/transport; new utterance endpointing continues |
| Stop/disconnect/shutdown | Assign `session_ended` where transport permits and await all owned tasks within policy |
| One load-call failure | Record the failure; other calls continue and remain isolated |

Logs include session/turn lifecycle IDs and sanitized outcome categories. They
exclude transcript, prompt, generated-token/chunk text, raw exception messages,
and credential-bearing provider URLs.

## From simulation to production

This assessment uses text chunks in place of audio, but the orchestration
boundaries map directly to a real voice stack. The session state, turn IDs,
cancellation, stale-output fencing, and metrics remain the same; only adapters
and transport-specific buffer clearing change.

| Simulation | Production equivalent |
|---|---|
| Text `audio.frame` events | PCM, Opus, or telephony audio frames |
| Simulated partial/final transcripts | Deepgram streaming STT or a local Whisper streaming wrapper |
| Streaming mock tokens | Hosted/local streaming LLM through `LLMProvider` |
| Text `audio.chunk` events | Hosted streaming TTS or local Piper returning PCM/mu-law |
| JSON WebSocket client | WebRTC media gateway or bidirectional SIP/Twilio stream |

Deepgram interim and endpointed transcripts map to the existing partial/final
events. A local Whisper deployment adds VAD, rolling audio windows, and
transcript-stability handling. A real TTS adapter consumes the same sentence
stream and forwards audio bytes as soon as the provider returns them.

For browser calls, WebRTC carries inbound/outbound media. For telephone calls,
SIP or Twilio Media Streams connects to the same orchestrator. Production
barge-in must also clear audio already buffered by the media transport; with
Twilio this maps to its `clear` message. Pipecat or LiveKit could later replace
the custom media/pipeline plumbing once demonstrating the underlying ownership
and concurrency is no longer the assessment objective.

Useful references:

- [Deepgram live STT](https://developers.deepgram.com/reference/speech-to-text/listen-streaming)
- [WhisperStreaming and SimulStreaming](https://github.com/ufal/whisper_streaming)
- [Deepgram streaming TTS](https://developers.deepgram.com/reference/text-to-speech/speak-streaming)
- [Piper local TTS](https://github.com/rhasspy/piper)
- [Twilio Media Streams messages](https://www.twilio.com/docs/voice/media-streams/websocket-messages)

### Latency budget

The approximately 800 ms target means end of caller speech to first audible bot
audio, not full-response completion. A reasonable p95 response-start budget is:

| Stage | Budget |
|---|---:|
| Inbound transport handling | 30 ms |
| Endpoint detection and final STT result | 180 ms |
| Orchestration and prompt preparation | 20 ms |
| LLM time to first token | 200 ms |
| First speakable phrase accumulation | 80 ms |
| TTS time to first audio byte | 150 ms |
| Outbound transport and playout | 70 ms |
| **Total** | **730 ms** |

This is a target, not a guarantee for every provider/network. STT runs while the
caller speaks; LLM generation, TTS, and playback overlap after the first phrase.
Optimize endpointing first, then LLM TTFT, phrase-level TTS startup, and finally
transport buffering/transcoding.

In production, boundary spans and histograms would move to OpenTelemetry and
Prometheus. Metrics labels must remain bounded (provider/model/outcome, not
session IDs or text); durable per-call analytics can be exported asynchronously
to a privacy-reviewed store. Alerts should separate endpointing, LLM, TTS, and
transport regressions instead of relying only on total latency.

## Known limitations

- Text stands in for audio; there is no codec, jitter buffer, acoustic VAD,
  echo cancellation, or actual media playout cancellation.
- Metrics and session history are process-local bounded memory, not durable or
  shared across workers.
- The mock provider has deterministic canned responses; provider retry/fallback
  is intentionally out of scope, and visible partial generations are never
  replayed.
- Authentication, authorization, quotas, TLS termination, multi-worker routing,
  distributed tracing exporters, and production rate limiting are not included.
- The load command demonstrates concurrency and isolation; it is not a capacity
  benchmark and does not model real audio bandwidth.

## Acceptance evidence

The automated suite plus manual commands prove the design acceptance rows:

| Acceptance | Evidence |
|---|---|
| AC-01 keyless run | `python -m app`; `python -m client --scenario default` |
| AC-02/03 STT endpointing | `tests/test_stt.py`, `tests/test_session.py`, default CLI |
| AC-04 true pipelining | `test_llm_production_overlaps_blocked_sentence_tts` |
| AC-05/06/07 barge-in and fencing | `tests/test_barge_in.py`, barge-in CLI |
| AC-08 delayed cancellation | `test_delayed_provider_cancellation_does_not_block_new_endpointing` |
| AC-09 task cleanup | session/barge-in task-leak assertions and full `pytest -v` |
| AC-10/18 protocol policy | `tests/test_api.py` recoverable/fatal/close-code cases |
| AC-11/12/13 metrics | `tests/test_metrics_state.py`, session/API tests, `/metrics` curls |
| AC-14 backpressure | outbound reservation tests and API overload close-`1013` test |
| AC-15 concurrent isolation | `tests/test_load_test.py`; `python -m load_test` |
| AC-16 closed-session metrics | registry/API tests and post-demo session query |
| AC-17 session termination | manager shutdown and disconnect-during-terminal-send tests |

## AI-tool disclosure

AI-assisted coding was used to turn the approved design into incremental
test-first implementation tasks, generate candidate code/tests, and perform a
final review. Each task was executed sequentially: its tests were observed
failing first, the smallest implementation was added, focused and full suites
were run, and the result was committed independently. Concurrency behavior was
validated with synchronization events and transport gates rather than accepted
from generated code by inspection alone. No external audio/LLM service, secret,
or private production data was used.
