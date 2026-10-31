from app.providers.stt import SimulatedSTT


def test_accept_frame_emits_cumulative_ordered_partials() -> None:
    stt = SimulatedSTT()

    first = stt.accept_frame(" hello ", 100)
    second = stt.accept_frame("world", 200)

    assert first.utterance_id == 1
    assert first.generation == 1
    assert first.text == "hello"
    assert first.last_frame_ns == 100
    assert second.utterance_id == 1
    assert second.generation == 2
    assert second.text == "hello world"
    assert second.last_frame_ns == 200
    assert stt.current_timer_generation() == 2


def test_explicit_finalize_preserves_last_frame_latency_baseline() -> None:
    stt = SimulatedSTT()
    stt.accept_frame("one", 100)
    stt.accept_frame("two", 450)

    final = stt.finalize()

    assert final is not None
    assert final.utterance_id == 1
    assert final.text == "one two"
    assert final.last_frame_ns == 450


def test_empty_utterance_does_not_finalize() -> None:
    assert SimulatedSTT().finalize() is None


def test_stale_silence_generation_cannot_finalize_newer_audio() -> None:
    stt = SimulatedSTT()
    first = stt.accept_frame("one", 100)
    stt.accept_frame("two", 200)

    assert stt.finalize(timer_generation=first.generation) is None
    assert stt.finalize(timer_generation=stt.current_timer_generation()) is not None


def test_utterance_finalizes_exactly_once() -> None:
    stt = SimulatedSTT()
    partial = stt.accept_frame("hello", 100)

    assert stt.finalize(timer_generation=partial.generation) is not None
    assert stt.finalize(timer_generation=partial.generation) is None
    assert stt.finalize() is None


def test_frame_after_finalization_starts_new_utterance_without_generation_reuse() -> None:
    stt = SimulatedSTT()
    first = stt.accept_frame("first", 100)
    stt.finalize(first.generation)

    second = stt.accept_frame("second", 200)
    final = stt.finalize(second.generation)

    assert second.utterance_id == 2
    assert second.generation > first.generation
    assert second.text == "second"
    assert final is not None
    assert final.utterance_id == 2
    assert final.text == "second"
