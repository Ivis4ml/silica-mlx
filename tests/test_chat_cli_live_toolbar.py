"""Tests for ``silica.chat.cli.live_toolbar``.

CHAT-CLI-HARDENING-6 (F3). Pure-Python coverage of the
swappable live-toolbar backend interface, the ANSI sticky-line
implementation, the no-op fallback, and the rolling tok/s
estimator. No prompt_toolkit, no terminal — every test drives the
backend with a ``StringIO`` and asserts the emitted byte stream.
"""

from __future__ import annotations

import io

from silica.chat.cli.live_toolbar import (
    AnsiLiveToolbar,
    LiveToolbar,
    NullLiveToolbar,
    RollingTokRate,
    make_live_toolbar,
)
from silica.chat.cli.palette import Palette
from silica.chat.cli.state import ChatCliState, StreamState

# ---------------------------------------------------------------------------
# RollingTokRate
# ---------------------------------------------------------------------------


def test_rolling_rate_returns_none_with_zero_samples() -> None:
    r = RollingTokRate(window=20)
    assert r.rate() is None


def test_rolling_rate_returns_none_with_single_sample() -> None:
    r = RollingTokRate(window=20)
    r.record(0.0)
    assert r.rate() is None


def test_rolling_rate_two_samples_one_second_apart_yields_one_per_sec() -> None:
    """Two samples one wall-second apart imply one inter-token
    interval at 1.0s; the rate is 1 token / 1.0s = 1.0 tok/s.

    The estimator uses ``(N - 1) / (t_last - t_first)`` because
    N samples imply N-1 inter-token intervals."""
    r = RollingTokRate(window=20)
    r.record(0.0)
    r.record(1.0)
    rate = r.rate()
    assert rate is not None
    assert abs(rate - 1.0) < 1e-9


def test_rolling_rate_steady_stream_at_50_tok_s() -> None:
    """Eleven samples spaced 20ms apart represent 50 tokens / sec
    (10 intervals × 0.02s = 0.2s; 10 / 0.2 = 50.0)."""
    r = RollingTokRate(window=20)
    for i in range(11):
        r.record(i * 0.02)
    rate = r.rate()
    assert rate is not None
    assert abs(rate - 50.0) < 1e-6


def test_rolling_rate_window_truncates_to_recent_samples() -> None:
    """A window of 5 samples discards earlier history; the rate
    reflects only the last 5 records, not the cumulative-from-start
    rate."""
    r = RollingTokRate(window=5)
    # Slow start: 1 token/sec for 10 samples.
    for i in range(10):
        r.record(float(i))
    # Sudden burst: 5 more samples at 100 tok/s (10ms apart).
    base = 9.0
    for i in range(1, 6):
        r.record(base + i * 0.01)
    rate = r.rate()
    assert rate is not None
    # Window now holds 5 samples spaced 0.01s apart → 4 intervals,
    # 0.04s span, 4 / 0.04 = 100 tok/s. The slow earlier samples
    # are evicted.
    assert abs(rate - 100.0) < 1e-6


def test_rolling_rate_reset_clears_samples() -> None:
    r = RollingTokRate(window=20)
    r.record(0.0)
    r.record(1.0)
    assert r.rate() is not None
    r.reset()
    assert r.rate() is None
    # Post-reset records start a fresh window.
    r.record(10.0)
    r.record(11.0)
    rate = r.rate()
    assert rate is not None
    assert abs(rate - 1.0) < 1e-9


def test_rolling_rate_non_positive_elapsed_returns_none() -> None:
    """Identical timestamps (clock didn't advance between samples)
    must not divide-by-zero — the estimator returns ``None``."""
    r = RollingTokRate(window=20)
    r.record(5.0)
    r.record(5.0)
    assert r.rate() is None


def test_rolling_rate_window_below_two_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="window"):
        RollingTokRate(window=1)


# ---------------------------------------------------------------------------
# NullLiveToolbar
# ---------------------------------------------------------------------------


def _state() -> ChatCliState:
    return ChatCliState(model_name="Qwen3-0.6B", stream_state=StreamState.DECODE)


def test_null_toolbar_refresh_is_a_noop() -> None:
    """``NullLiveToolbar.refresh`` must not write anything anywhere."""
    bar = NullLiveToolbar()
    # The Null backend has no output stream — refresh should
    # succeed without raising and without side effects.
    bar.refresh(_state())


def test_null_toolbar_context_manager_protocol() -> None:
    """Enter / exit pair returns the backend itself and produces
    no output — the caller can use ``with`` unconditionally."""
    bar = NullLiveToolbar()
    with bar as got:
        assert got is bar
        got.refresh(_state())


def test_null_toolbar_exit_safe_when_refresh_never_called() -> None:
    """Exit must tolerate an empty lifecycle (caller aborted before
    the first token)."""
    bar = NullLiveToolbar()
    with bar:
        pass


def test_null_toolbar_is_a_live_toolbar() -> None:
    """``LiveToolbar`` is the protocol the chat-turn flow types
    against; the Null fallback must satisfy it."""
    assert isinstance(NullLiveToolbar(), LiveToolbar)


# ---------------------------------------------------------------------------
# AnsiLiveToolbar — output sequence assertions
# ---------------------------------------------------------------------------


def test_ansi_toolbar_enter_reserves_line_below_cursor() -> None:
    """On ``__enter__`` the backend writes save → next-line → clear
    → restore so the line below the cursor is blank-and-reserved.
    No toolbar text yet — the caller has not provided a state."""
    out = io.StringIO()
    bar = AnsiLiveToolbar(
        render=lambda s: "TOOLBAR",
        output_stream=out,
    )
    bar.__enter__()
    written = out.getvalue()
    assert written == (
        AnsiLiveToolbar.SAVE_CURSOR
        + AnsiLiveToolbar.NEXT_LINE
        + AnsiLiveToolbar.CLEAR_LINE
        + AnsiLiveToolbar.RESTORE_CURSOR
    )
    bar.__exit__(None, None, None)


def test_ansi_toolbar_uses_dec_save_restore_sequences() -> None:
    """DEC save/restore is the compatibility path for real terminals.

    The earlier CSI s/u pair works in some emulators but can be
    ignored in prompt-toolkit-hosted sessions; when restore is
    ignored, the live toolbar lands in the transcript as a normal
    ``state=...`` line.
    """
    assert AnsiLiveToolbar.SAVE_CURSOR == "\x1b7"
    assert AnsiLiveToolbar.RESTORE_CURSOR == "\x1b8"


def test_ansi_toolbar_refresh_writes_save_advance_clear_text_restore() -> None:
    """``refresh`` produces: save cursor → advance to reserved
    line → clear → write toolbar text → restore cursor. The
    streamed-text cursor is preserved across the redraw."""
    out = io.StringIO()
    bar = AnsiLiveToolbar(
        render=lambda s: "TOOLBAR_TEXT",
        output_stream=out,
    )
    with bar:
        out.truncate(0)
        out.seek(0)
        bar.refresh(_state())
        written = out.getvalue()
    assert AnsiLiveToolbar.SAVE_CURSOR in written
    assert AnsiLiveToolbar.NEXT_LINE in written
    assert AnsiLiveToolbar.CLEAR_LINE in written
    assert "TOOLBAR_TEXT" in written
    assert AnsiLiveToolbar.RESTORE_CURSOR in written
    # Order: save first, restore last.
    assert written.startswith(AnsiLiveToolbar.SAVE_CURSOR)
    assert written.endswith(AnsiLiveToolbar.RESTORE_CURSOR)


def test_ansi_toolbar_refresh_uses_current_state_each_call() -> None:
    """Each refresh re-invokes the render callable, so a state
    that changes between refreshes produces different toolbar
    text. Locks the F3 contract (toolbar reflects live state, not
    a stale snapshot)."""
    out = io.StringIO()
    seen_states: list[StreamState] = []

    def _render(state: ChatCliState) -> str:
        seen_states.append(state.stream_state)
        return f"state={state.stream_state.value}"

    state = ChatCliState(stream_state=StreamState.PREFILL)
    bar = AnsiLiveToolbar(render=_render, output_stream=out)
    with bar:
        bar.refresh(state)
        state.stream_state = StreamState.DECODE
        bar.refresh(state)

    assert seen_states == [StreamState.PREFILL, StreamState.DECODE]
    written = out.getvalue()
    assert "state=prefill" in written
    assert "state=decode" in written


def test_ansi_toolbar_exit_clears_line_without_displacing_cursor() -> None:
    """``__exit__`` clears the reserved line via the same save →
    advance → clear → restore dance and stops there. The caller's
    own turn-end ``\\n`` advances past the now-empty toolbar line."""
    out = io.StringIO()
    bar = AnsiLiveToolbar(
        render=lambda s: "TOOLBAR",
        output_stream=out,
    )
    with bar:
        out.truncate(0)
        out.seek(0)
    written = out.getvalue()
    expected = (
        AnsiLiveToolbar.SAVE_CURSOR
        + AnsiLiveToolbar.NEXT_LINE
        + AnsiLiveToolbar.CLEAR_LINE
        + AnsiLiveToolbar.RESTORE_CURSOR
    )
    assert written == expected


def test_ansi_toolbar_exit_runs_even_when_refresh_never_called() -> None:
    """Caller that aborts before the first token (e.g. KeyboardInterrupt
    during prefill) must still get a clean terminal — exit clears
    the reserved line."""
    out = io.StringIO()
    bar = AnsiLiveToolbar(render=lambda s: "X", output_stream=out)
    with bar:
        pass  # no refresh
    written = out.getvalue()
    # Enter wrote one save/advance/clear/restore; exit wrote
    # another. Both visible in the captured stream.
    assert written.count(AnsiLiveToolbar.SAVE_CURSOR) == 2
    assert written.count(AnsiLiveToolbar.RESTORE_CURSOR) == 2


def test_ansi_toolbar_exit_runs_when_refresh_raises() -> None:
    """An exception thrown from ``refresh`` (or from any caller
    code inside the ``with`` block) must still trigger clean
    teardown — that is the whole point of using a context manager."""
    out = io.StringIO()
    bar = AnsiLiveToolbar(render=lambda s: "X", output_stream=out)
    try:
        with bar:
            raise RuntimeError("simulated abort")
    except RuntimeError:
        pass
    written = out.getvalue()
    # Enter + exit each emitted the cleanup sequence; that's two
    # save markers regardless of how the body terminated.
    assert written.count(AnsiLiveToolbar.SAVE_CURSOR) == 2


def test_ansi_toolbar_exit_idempotent_on_double_invocation() -> None:
    """Calling ``__exit__`` a second time (defensive double-cleanup)
    must not emit further escapes — the backend has already cleared
    the reserved line."""
    out = io.StringIO()
    bar = AnsiLiveToolbar(render=lambda s: "X", output_stream=out)
    with bar:
        pass
    sequence_after_first_exit = out.getvalue()
    bar.__exit__(None, None, None)
    assert out.getvalue() == sequence_after_first_exit


def test_ansi_toolbar_refresh_is_noop_before_enter() -> None:
    """A refresh outside the lifecycle is a no-op — the backend
    has not yet reserved the line, so writing escapes would
    corrupt unrelated output."""
    out = io.StringIO()
    bar = AnsiLiveToolbar(render=lambda s: "X", output_stream=out)
    bar.refresh(_state())
    assert out.getvalue() == ""


# ---------------------------------------------------------------------------
# make_live_toolbar — backend selection
# ---------------------------------------------------------------------------


def test_make_live_toolbar_returns_null_when_not_a_tty() -> None:
    """Non-TTY output (file redirect, pipe) must use the no-op
    backend — ANSI escapes would litter the captured stream."""
    out = io.StringIO()  # StringIO has isatty() → False
    bar = make_live_toolbar(
        palette=Palette.truecolor(),
        output_stream=out,
    )
    assert isinstance(bar, NullLiveToolbar)


def test_make_live_toolbar_returns_null_when_term_is_dumb() -> None:
    """``TERM=dumb`` is the legacy opt-out for terminals that
    do not handle escape sequences reliably."""
    out = io.StringIO()
    bar = make_live_toolbar(
        palette=Palette.truecolor(),
        output_stream=out,
        is_tty=True,  # force TTY so the term check kicks in
        term="dumb",
    )
    assert isinstance(bar, NullLiveToolbar)


def test_make_live_toolbar_returns_null_when_palette_is_plain() -> None:
    """A plain palette implies the user already opted out of
    colour; live cursor dancing would surprise them. The
    post-turn toolbar still surfaces final values."""
    out = io.StringIO()
    bar = make_live_toolbar(
        palette=Palette.plain(),
        output_stream=out,
        is_tty=True,
    )
    assert isinstance(bar, NullLiveToolbar)


def test_make_live_toolbar_returns_ansi_when_capable() -> None:
    """TTY + non-dumb TERM + non-plain palette → ANSI backend."""
    out = io.StringIO()
    bar = make_live_toolbar(
        palette=Palette.truecolor(),
        output_stream=out,
        is_tty=True,
        term="xterm-256color",
    )
    assert isinstance(bar, AnsiLiveToolbar)


def test_make_live_toolbar_ansi_backend_uses_render_toolbar() -> None:
    """The factory's render closure routes through the public
    :func:`render_toolbar`; refreshing emits a recognisable
    toolbar field with the live ``tokens=N/max`` figure."""
    out = io.StringIO()
    bar = make_live_toolbar(
        palette=Palette.truecolor(),
        output_stream=out,
        is_tty=True,
        term="xterm-256color",
    )
    assert isinstance(bar, AnsiLiveToolbar)
    state = ChatCliState(
        model_name="Qwen3-0.6B", stream_state=StreamState.DECODE
    )
    state.tokens_generated = 7
    state.max_tokens = 100
    with bar:
        bar.refresh(state)
    written = out.getvalue()
    # render_toolbar must surface the live tokens=N/max field.
    assert "tokens=" in written
    assert "7/100" in written
    assert "state=" in written


# ---------------------------------------------------------------------------
# clear() — out-of-band cleanup without ending the lifecycle
# ---------------------------------------------------------------------------


def test_ansi_toolbar_clear_wipes_line_without_ending_lifecycle() -> None:
    """``clear()`` produces the same save → advance → clear →
    restore sequence as ``__exit__`` but leaves the backend
    active. Used by abort-message printing so the marker lands on
    a clean line; the ``with``-block exit still runs the final
    teardown afterwards."""
    out = io.StringIO()
    bar = AnsiLiveToolbar(render=lambda s: "X", output_stream=out)
    with bar:
        out.truncate(0)
        out.seek(0)
        bar.clear()
        cleared = out.getvalue()
    assert cleared == (
        AnsiLiveToolbar.SAVE_CURSOR
        + AnsiLiveToolbar.NEXT_LINE
        + AnsiLiveToolbar.CLEAR_LINE
        + AnsiLiveToolbar.RESTORE_CURSOR
    )


def test_ansi_toolbar_clear_then_refresh_repopulates_line() -> None:
    """After ``clear()`` the backend is still active; a subsequent
    ``refresh`` writes the toolbar text into the same reserved
    line. Locks the lifecycle invariant the abort branch in
    app.py depends on."""
    out = io.StringIO()
    bar = AnsiLiveToolbar(
        render=lambda s: "TOOLBAR", output_stream=out
    )
    with bar:
        bar.clear()
        out.truncate(0)
        out.seek(0)
        bar.refresh(_state())
        post_refresh = out.getvalue()
    assert "TOOLBAR" in post_refresh


def test_ansi_toolbar_clear_is_noop_before_enter() -> None:
    """Defensive: calling ``clear`` outside the lifecycle must
    not write escapes."""
    out = io.StringIO()
    bar = AnsiLiveToolbar(render=lambda s: "X", output_stream=out)
    bar.clear()
    assert out.getvalue() == ""


def test_null_toolbar_clear_is_a_noop() -> None:
    """Null backend's ``clear`` must not raise and must not
    side-effect — the chat-turn flow can call it
    unconditionally on the abort path regardless of which backend
    was selected."""
    bar = NullLiveToolbar()
    with bar:
        bar.clear()
