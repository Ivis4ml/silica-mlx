"""silica.chat.cli.app — prompt_toolkit chat REPL (C-3).

Wires the C-1 palette / state / toolbar and the C-2 slash-command
dispatcher into a live REPL using ``prompt_toolkit``'s
``PromptSession``. Streams tokens from ``ChatSession.chat`` to the
terminal between prompts. Two-surface toolbar story: the input
phase uses ``PromptSession.bottom_toolbar`` (refreshes on each
keystroke); the generation phase uses the live toolbar backend
in :mod:`silica.chat.cli.live_toolbar` (HARDENING-6 / F3) for
per-token ``tokens=N/max`` / ``tok/s`` / ``state=`` updates. See
``plans/CHAT_CLI_HARDENING.md`` Decision D for why the full
prompt-toolkit ``Application`` layout is deferred.

Invocation paths after C-3:

- ``silica chat --model Qwen/Qwen3-0.6B`` — explicit subcommand.
- ``python scripts/chat.py --model ...`` — script alias.
- ``silica`` (no subcommand) — bare-launch claude-style; lands at
  C-7 via argv preprocessing.

Sampling knobs are intentionally *not* CLI flags here per design
doc §6 — the launch surface stays minimal (``--model``,
``--system``, ``--kv-codec``); ``temperature`` / ``top_p`` / etc.
are adjusted mid-session via ``/config``.

Ctrl-C semantics:

- During input prompt: KeyboardInterrupt on a non-empty line clears
  the input; on an empty line raises EOF and exits the REPL.
- During generation (future C-5 with cooperative cancellation):
  first Ctrl-C signals abort, second Ctrl-C exits. v1 of this
  module relies on Python's default KeyboardInterrupt unwind from
  inside ``engine.generate`` — pressing Ctrl-C aborts the current
  turn and returns to the prompt.

Manual smoke checklist — see ``plans/CHAT_CLI_OPENING.md`` §10.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from silica.chat.cli.code_fence import (
    CodeFenceParser,
    EnterFence,
    ExitFence,
    PlainText,
)
from silica.chat.cli.commands import (
    CommandResult,
    dispatch_command,
    is_slash_command,
)
from silica.chat.cli.config import initial_config
from silica.chat.cli.live_toolbar import (
    RollingTokRate,
    make_live_toolbar,
)
from silica.chat.cli.palette import ColorName, Palette, detect_palette
from silica.chat.cli.persistence import (
    SessionFileError,
    load_session,
    save_session,
)
from silica.chat.cli.state import ChatCliState, StreamState
from silica.chat.cli.thinking_parser import (
    EnterThinking,
    ExitThinking,
    ReplyChunk,
    ThinkingChunk,
    ThinkingParser,
)
from silica.chat.cli.toolbar import (
    render_codec_hint,
    render_showcase,
    render_toolbar,
)


def _model_basename(repo: str) -> str:
    """Strip an HF org prefix and a trailing dtype suffix.

    ``Qwen/Qwen3-0.6B`` → ``Qwen3-0.6B``;
    ``mlx-community/Qwen3.5-35B-A3B-4bit`` → ``Qwen3.5-35B-A3B-4bit``.
    Used for the ``model=`` toolbar field — the prefix is consistent
    across vendors and adds no signal to the user.
    """
    return repo.split("/", 1)[-1]


def _model_supports_implicit_thinking(model_basename: str) -> bool:
    """Whether the model's chat template prepends ``<think>\\n`` to
    the assistant generation slot when ``enable_thinking=True``.

    True for the Qwen3 / Qwen3.5 / Qwen3.5-MoE families. Other
    models (Gemma4, Qwen2.x, etc.) emit thinking blocks (when they
    do at all) with explicit opening tags in the model output, so
    the parser's default IDLE start-state is correct for them.

    The match is case-insensitive on the basename's ``qwen3``
    prefix; the family list extends naturally as more Qwen3
    variants ship.
    """
    return model_basename.lower().startswith("qwen3")


def _format_user_input(text: str, palette: Palette) -> str:
    """Render the echoed user line in the conversation log."""
    prefix = palette.colorize("You ›", "green", bold=True)
    body = palette.colorize(text, "green_dim")
    return f"{prefix} {body}"


def _format_assistant_prefix(palette: Palette) -> str:
    return palette.colorize("silica ›", "orange", bold=True) + " "


def _print_phase_indicator(
    label: str, color: ColorName, palette: Palette
) -> None:
    """Render an inline phase indicator (``⠋ <label>...``) on the
    current line.

    Two callers:

    - C-5: ``label="prefilling"``, ``color="yellow"`` — the engine
      is consuming the prompt before producing any token.
    - C-8: ``label="thinking"``, ``color="magenta"`` — the model
      emitted ``<think>`` and is reasoning before producing the
      visible reply.

    Cleared by :func:`_clear_phase_indicator` once the phase
    transitions (first decoded reply token for prefill;
    ``</think>`` close tag for thinking).
    """
    msg = palette.colorize(f"⠋ {label}...", color, dim=True)
    sys.stdout.write(msg)
    sys.stdout.flush()


def _clear_phase_indicator() -> None:
    """Remove the inline phase indicator from the current line.
    ANSI: carriage-return + clear-to-end-of-line. Safe to call
    when no indicator is present (becomes a no-op cursor
    movement on most terminals)."""
    sys.stdout.write("\r\x1b[K")
    sys.stdout.flush()


def _highlight_code(code: str, language: str) -> str:
    """Run pygments syntax highlighting on a code block, returning
    ANSI-colourised text suitable for stdout.

    Falls back to monochrome (the original ``code`` unchanged)
    when:

    - pygments is not installed (chat extras not active);
    - the language identifier does not resolve to a known lexer
      (and ``guess_lexer`` also fails — rare but possible for
      pseudo-languages or fenced-but-unfenced content).

    Returns the highlighted string with a trailing newline; the
    caller writes it as-is to stdout. Style is ``monokai`` per
    design doc §4.1 (dark-terminal-friendly, common default in
    chat apps).
    """
    try:
        from pygments import highlight  # type: ignore[import-untyped]
        from pygments.formatters import (  # type: ignore[import-untyped]
            Terminal256Formatter,
        )
        from pygments.lexers import (  # type: ignore[import-untyped]
            get_lexer_by_name,
            guess_lexer,
        )
        from pygments.util import ClassNotFound  # type: ignore[import-untyped]
    except ImportError:
        return code
    lexer: Any = None
    if language:
        try:
            lexer = get_lexer_by_name(language, stripnl=False)
        except ClassNotFound:
            lexer = None
    if lexer is None:
        try:
            lexer = guess_lexer(code, stripnl=False)
        except (ClassNotFound, Exception):
            return code
    formatter = Terminal256Formatter(style="monokai")
    result: str = highlight(code, lexer, formatter)
    return result


def _format_feedback(result: CommandResult, palette: Palette) -> list[str]:
    """Render command result feedback lines with appropriate colour.

    Errors render in red; normal feedback in dim cyan to mark it
    as system output rather than assistant content.
    """
    colour: ColorName = "red" if result.error else "cyan"
    return [
        palette.colorize(line, colour, dim=not result.error)
        for line in result.feedback
    ]


def _print_lines(lines: list[str]) -> None:
    """Write rendered feedback to stdout. Each line gets a newline."""
    for line in lines:
        sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _print_separator(palette: Palette) -> None:
    """Thin grey separator between turns. Single line so the
    conversation log stays compact."""
    sep = palette.colorize("─" * 40, "grey", dim=True)
    sys.stdout.write("\n" + sep + "\n\n")
    sys.stdout.flush()


def run_chat(args: argparse.Namespace) -> int:
    """Entry point for ``silica chat``.

    Loads the model, builds a :class:`silica.chat.session.ChatSession`,
    and runs the prompt_toolkit REPL until the user issues ``/exit``,
    types Ctrl-D, or interrupts twice.
    """
    try:
        from prompt_toolkit import PromptSession  # type: ignore[import-not-found]
        from prompt_toolkit.formatted_text import ANSI  # type: ignore[import-not-found]
        from prompt_toolkit.history import FileHistory  # type: ignore[import-not-found]
        from prompt_toolkit.key_binding import (  # type: ignore[import-not-found]
            KeyBindings,
        )
        from prompt_toolkit.keys import Keys  # type: ignore[import-not-found]
    except ImportError:
        print(
            "silica chat requires prompt_toolkit. Install with:\n"
            "    uv pip install -e '.[chat]'",
            file=sys.stderr,
        )
        return 1

    # Defer engine + session imports so a `--help` on this module
    # does not pay the MLX warm-up cost.
    from silica.bench.codec_registry import get_codec_spec
    from silica.chat.session import ChatSession
    from silica.engine import Engine
    from silica.kvcache.prefix import RadixPrefixCache
    from silica.kvcache.store import SyntheticPrefixBlockStore
    from silica.models.factory import adapter_for_repo

    palette = detect_palette()

    # --- Load model + engine + chat session ---
    print(palette.colorize(f"Loading {args.model} ...", "grey", dim=True))
    sys.stdout.flush()
    adapter, kv = adapter_for_repo(args.model)
    engine = Engine(adapter, kv)

    # --- Build prefix cache (Tier 2 architectural showcase) ---
    # The session-scoped RadixPrefixCache is what makes multi-turn
    # chat reuse the conversation history's KV — silica-mlx's
    # equivalent of SGLang's RadixAttention. Without it every turn
    # would re-prefill the full history, defeating the framework's
    # signature optimisation in its own client.
    prefix_cache = _build_prefix_cache(
        adapter,
        codec_id=getattr(args, "kv_codec", None),
        get_codec_spec=get_codec_spec,
        store_cls=SyntheticPrefixBlockStore,
        cache_cls=RadixPrefixCache,
    )

    # mypy variance limitation: ChatSession's _EngineLike Protocol
    # treats kv_manager as a settable attribute, while Engine
    # exposes it as a read-only property; the runtime contract is
    # satisfied (Protocol member access works either way), only
    # the structural type-check trips.
    # --- Build state ---
    state = ChatCliState(
        model_name=_model_basename(args.model),
        codec_id=getattr(args, "kv_codec", None),
    )
    state.config.update(initial_config())
    if args.system:
        state.config["system_prompt"] = args.system
    state.stream_state = StreamState.IDLE

    # CHAT-CLI-HARDENING-2 (F2): pass the configured thinking_mode
    # through to ChatSession at construction so the very first chat
    # turn renders with the right ``enable_thinking`` kwarg. The
    # config schema's default is ``True`` (Qwen3 family default);
    # the shell re-syncs from state.config before each chat() call
    # (see the chat-turn loop below) so /config thinking_mode=on|off
    # takes effect on the next turn without requiring a session
    # rebuild.
    initial_thinking_mode = state.config.get("thinking_mode")
    if not isinstance(initial_thinking_mode, bool):
        initial_thinking_mode = None
    chat_session = ChatSession(
        adapter,
        engine,  # type: ignore[arg-type]
        system_prompt=args.system,
        prefix_cache=prefix_cache,
        thinking_mode=initial_thinking_mode,
    )

    # --- prompt_toolkit session ---
    history_path = Path.home() / ".cache" / "silica" / "chat_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    kb = KeyBindings()

    @kb.add(Keys.ControlJ)  # Shift+Enter / Alt+Enter inserts newline
    def _(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    def _bottom_toolbar() -> Any:
        return ANSI(render_toolbar(state, palette=palette))

    session: Any = PromptSession(
        history=FileHistory(str(history_path)),
        bottom_toolbar=_bottom_toolbar,
        multiline=False,
        key_bindings=kb,
        enable_history_search=True,
    )

    # --- Greeting ---
    greeting = palette.colorize(
        f"silica chat — {state.model_name} ({state.codec_id or 'fp16'}). "
        "Type /help for commands, /exit to quit.",
        "cyan",
        dim=True,
    )
    print(greeting)
    print()

    # --- REPL ---
    prompt_str = palette.colorize("You ›", "green", bold=True) + " "
    while True:
        # ``PromptSession.bottom_toolbar`` covers the input phase;
        # the generation phase below opens its own live toolbar
        # backend (HARDENING-6 / F3) so ``tokens=N/max`` / ``tok/s``
        # / ``state=`` update per-token while the model decodes.
        try:
            user_text = session.prompt(ANSI(prompt_str))
        except KeyboardInterrupt:
            # Empty line + Ctrl-C exits; non-empty just clears.
            continue
        except EOFError:
            print(palette.colorize("bye.", "grey", dim=True))
            break

        text = user_text.strip()
        if not text:
            continue

        # Slash command or chat?
        # ``regenerate_text`` is set by /regenerate when a prior turn
        # was successfully popped; the slash branch then falls through
        # into the chat-turn flow with that text instead of the
        # normal ``continue``. CHAT-CLI-HARDENING-4 (F4).
        # ``regenerate_snapshot`` / ``regenerate_thinking_snapshot``
        # capture the pre-pop session state so the abort / error
        # branches in the chat-turn flow can roll back if the
        # regenerated turn fails — without rollback the original
        # ``(user, assistant)`` pair would be permanently lost and
        # ``ChatSession`` would be left with a trailing user-only
        # message.
        regenerate_text: str | None = None
        regenerate_snapshot: list[dict[str, str]] | None = None
        regenerate_thinking_snapshot: str | None = None
        if is_slash_command(text):
            result = dispatch_command(text, state)
            _print_lines(_format_feedback(result, palette))
            if result.quit:
                break
            if result.request_reset:
                chat_session.reset()
                # Swap the prefix cache so prior-conversation
                # tokens cannot leak into the new session — see
                # ChatSession.reset() docstring for the contract.
                fresh_cache = _build_prefix_cache(
                    adapter,
                    codec_id=getattr(args, "kv_codec", None),
                    get_codec_spec=get_codec_spec,
                    store_cls=SyntheticPrefixBlockStore,
                    cache_cls=RadixPrefixCache,
                )
                chat_session.set_prefix_cache(fresh_cache)
                state.turn = 0
                state.last_turn_thinking = ""
                state.prefix_hit_blocks = None
                state.prefix_hit_max = None
                state.total_prefix_hit_tokens = 0
                state.total_decode_tokens = 0
                state.total_decode_seconds = 0.0
            if result.request_system_prompt is not None:
                # CHAT-CLI-HARDENING-1 (F1): propagate /system to
                # the live ChatSession alongside the config-side
                # update the dispatcher already performed. Empty
                # string clears; non-empty replaces. The prefix
                # cache is not explicitly invalidated — the
                # rendered prompt's leading tokens change with the
                # new system content, so the next chat() call's
                # peek mismatches the old radix nodes naturally.
                chat_session.set_system_prompt(
                    result.request_system_prompt or None
                )
            if result.request_expand_thinking:
                expanded = palette.colorize(
                    "── thinking ──\n" + state.last_turn_thinking,
                    "grey",
                    dim=True,
                )
                sys.stdout.write(expanded + "\n\n")
                sys.stdout.flush()
            if result.request_regenerate:
                # CHAT-CLI-HARDENING-4 (F4): pop the last
                # ``(user, assistant)`` pair from the session and
                # re-issue ``chat()`` with the same user prompt.
                # The prefix cache is **not** invalidated — the
                # popped user text re-tokenises to identical prompt
                # ids, so the next turn's peek hits every block of
                # the prior prefill (Q-012 cross-call reuse). On
                # an empty / fresh / aborted-turn history,
                # ``pop_last_exchange`` returns ``None``; report
                # the no-op and continue to the prompt.
                # Snapshot before pop so the abort / error branches
                # in the chat-turn flow can roll back if the
                # regenerated turn fails. ``messages`` is a shallow
                # copy of the live list; restore via
                # ``replace_messages`` which deep-copies entries.
                pre_pop_snapshot = chat_session.messages
                pre_pop_thinking = state.last_turn_thinking
                popped = chat_session.pop_last_exchange()
                if popped is None:
                    sys.stdout.write(
                        palette.colorize(
                            "(/regenerate: no prior turn to redo)\n",
                            "yellow",
                            dim=True,
                        )
                    )
                    sys.stdout.flush()
                else:
                    regenerate_text = popped
                    regenerate_snapshot = pre_pop_snapshot
                    regenerate_thinking_snapshot = pre_pop_thinking
            if result.request_session_save:
                _handle_session_save(
                    result.request_session_save,
                    chat_session=chat_session,
                    state=state,
                    palette=palette,
                )
            if result.request_session_load:
                load_outcome = _handle_session_load(
                    result.request_session_load,
                    chat_session=chat_session,
                    state=state,
                    palette=palette,
                )
                if load_outcome:
                    # On successful /load we swap in a fresh prefix
                    # cache: the restored history was produced
                    # against a different sequence of inserts, so
                    # the old radix tree is meaningless and reusing
                    # it would leak stale blocks.
                    fresh_cache = _build_prefix_cache(
                        adapter,
                        codec_id=getattr(args, "kv_codec", None),
                        get_codec_spec=get_codec_spec,
                        store_cls=SyntheticPrefixBlockStore,
                        cache_cls=RadixPrefixCache,
                    )
                    chat_session.set_prefix_cache(fresh_cache)
            if result.request_model_swap:
                swap_outcome = _swap_model(
                    result.request_model_swap,
                    args=args,
                    state=state,
                    keep_history=result.request_model_keep_history,
                    prior_session=chat_session,
                    get_adapter=adapter_for_repo,
                    engine_cls=Engine,
                    session_cls=ChatSession,
                    cache_builder=lambda new_adapter: _build_prefix_cache(
                        new_adapter,
                        codec_id=getattr(args, "kv_codec", None),
                        get_codec_spec=get_codec_spec,
                        store_cls=SyntheticPrefixBlockStore,
                        cache_cls=RadixPrefixCache,
                    ),
                    palette=palette,
                )
                if swap_outcome is not None:
                    adapter, engine, chat_session = swap_outcome
            if result.request_showcase:
                report = render_showcase(state, palette=palette)
                sys.stdout.write(report + "\n")
                sys.stdout.flush()
            if regenerate_text is None:
                continue
            # /regenerate fall-through: feed the popped user prompt
            # into the regular chat-turn flow below so the same
            # streaming / parser / metric plumbing applies. The
            # original ``/regenerate`` line is replaced by the
            # captured prompt text.
            text = regenerate_text

        # Regular chat turn. Defer the assistant prefix line until
        # the first reply token actually arrives — that way long-
        # prompt prefill shows a thinking indicator instead of a
        # bare ``silica ›`` followed by silence. C-8: a separate
        # ``<think>`` parser collapses the model's reasoning block
        # into a magenta indicator.
        params = _sampling_params_from_state(state, adapter)

        state.stream_state = StreamState.PREFILL
        state.tokens_generated = 0
        state.max_tokens = int(state.config.get("max_tokens", 1024))
        state.last_turn_thinking = ""
        # CHAT-CLI-HARDENING-6 (F3): live toolbar backend selection.
        # ANSI sticky-bottom-line on a capable TTY; no-op fallback
        # otherwise. ``RollingTokRate`` drives ``state.tok_per_sec``
        # live during the turn (the post-turn engine snapshot still
        # overwrites it with the precise figure once chat() returns).
        live_toolbar = make_live_toolbar(
            palette=palette,
            output_stream=sys.stdout,
            term=os.environ.get("TERM"),
        )
        tok_rate = RollingTokRate(window=20)
        # Per-token timestamp side-state for live tok/s; reset
        # between turns so the prior turn's samples cannot bleed
        # into the new window.
        state.tok_per_sec = None
        thinking_display = str(state.config.get("thinking", "auto"))
        # Qwen3 / Qwen3.5 chat templates append ``<think>\n`` to the
        # *prompt* when ``enable_thinking=True`` (the family default).
        # The model's output therefore starts with raw reasoning text
        # and ends with ``</think>`` — there is no opening tag in the
        # stream. We initialise the parser in THINKING state so the
        # first ``</think>`` correctly transitions out instead of
        # leaking the entire reasoning block as a ReplyChunk.
        implicit_thinking = (
            bool(state.config.get("thinking_mode", True))
            and _model_supports_implicit_thinking(state.model_name)
        )
        parser = ThinkingParser(
            start_in_thinking=implicit_thinking
        )
        thinking_started_at: list[float] = []  # mutable for closure write
        prefix_emitted: list[bool] = [False]
        fence_parser = CodeFenceParser()
        in_fence: list[bool] = [False]
        _print_phase_indicator("prefilling", "yellow", palette)

        def _emit_assistant_prefix_once() -> None:
            if not prefix_emitted[0]:
                _clear_phase_indicator()
                sys.stdout.write(_format_assistant_prefix(palette))
                sys.stdout.flush()
                prefix_emitted[0] = True

        def _emit_reply_text(text: str) -> None:
            """Send reply text through the code-fence parser and
            render its events. Plain text writes through; code
            inside a fence is buffered until the closing fence
            arrives, at which point it goes through pygments and
            emits as one highlighted block. While buffering, an
            inline ``writing code (lang)...`` indicator is
            visible (cyan) so the user knows the model is
            producing code that will appear shortly."""
            for fevent in fence_parser.feed(text):
                if isinstance(fevent, PlainText):
                    if fevent.text:
                        sys.stdout.write(fevent.text)
                        sys.stdout.flush()
                elif isinstance(fevent, EnterFence):
                    in_fence[0] = True
                    label = (
                        f"writing code ({fevent.language})"
                        if fevent.language
                        else "writing code"
                    )
                    _print_phase_indicator(label, "cyan", palette)
                elif isinstance(fevent, ExitFence):
                    in_fence[0] = False
                    _clear_phase_indicator()
                    highlighted = _highlight_code(
                        fevent.code, fevent.language
                    )
                    # Frame the block with a leading newline so
                    # the indicator's line break is preserved.
                    sys.stdout.write(
                        "\n" + highlighted
                    )
                    if not highlighted.endswith("\n"):
                        sys.stdout.write("\n")
                    sys.stdout.flush()

        def _stream_callback(delta: str) -> None:
            # One stream_to call == one decoded token; track the
            # token counter once per call regardless of how the
            # parser splits the delta into events.
            state.tokens_generated += 1
            # HARDENING-6: live tok/s via a rolling window. The
            # sample is taken before parser dispatch so the toolbar
            # refresh below sees the just-incremented rate.
            tok_rate.record(time.monotonic())
            rate = tok_rate.rate()
            if rate is not None:
                state.tok_per_sec = rate
            for event in parser.feed(delta):
                if isinstance(event, EnterThinking):
                    # Whatever indicator is currently up (prefill
                    # yellow OR a stale thinking line from a
                    # previous block in the same turn) gets cleared.
                    _clear_phase_indicator()
                    state.stream_state = StreamState.THINKING
                    thinking_started_at.append(time.monotonic())
                    if thinking_display != "hidden":
                        _print_phase_indicator(
                            "thinking", "magenta", palette
                        )
                elif isinstance(event, ThinkingChunk):
                    state.last_turn_thinking += event.text
                    if thinking_display == "show":
                        sys.stdout.write(
                            palette.colorize(event.text, "grey", dim=True)
                        )
                        sys.stdout.flush()
                elif isinstance(event, ExitThinking):
                    _clear_phase_indicator()
                    if thinking_started_at and thinking_display != "hidden":
                        elapsed = time.monotonic() - thinking_started_at[-1]
                        sys.stdout.write(
                            palette.colorize(
                                f"thought for {elapsed:.1f}s\n",
                                "grey",
                                dim=True,
                            )
                        )
                    state.stream_state = StreamState.DECODE
                    _emit_assistant_prefix_once()
                elif isinstance(event, ReplyChunk):
                    if state.stream_state is StreamState.PREFILL:
                        state.stream_state = StreamState.DECODE
                    _emit_assistant_prefix_once()
                    _emit_reply_text(event.text)
            # HARDENING-6: refresh the bottom toolbar so the user
            # sees ``tokens=N/max``, live ``tok/s``, and the
            # current ``state=`` value update per-token. The
            # backend is Null (no-op) on non-TTY / TERM=dumb /
            # plain-palette environments, so this is safe to call
            # unconditionally.
            live_toolbar.refresh(state)

        # CHAT-CLI-HARDENING-2 (F2): re-sync the live thinking_mode
        # before each chat() turn so /config thinking_mode=on|off
        # takes effect immediately without requiring a session
        # rebuild. ``state.config["thinking_mode"]`` is bool (the
        # config schema enforces it); a non-bool value (defensive)
        # collapses to ``None`` so ChatSession omits the kwarg.
        sync_thinking_mode = state.config.get("thinking_mode")
        if not isinstance(sync_thinking_mode, bool):
            sync_thinking_mode = None
        chat_session.set_thinking_mode(sync_thinking_mode)

        # HARDENING-6: ``with live_toolbar`` guarantees ``__exit__``
        # runs on every exit path (success, ``continue`` from an
        # abort branch, exception bubbling out of the parser
        # drain). The initial ``refresh`` makes ``state=prefill``
        # visible during the prefill wait — without it the
        # reserved line stays blank for the 1-3 seconds of prefill
        # on bigger models. The Null backend's enter/exit/refresh
        # are no-ops, so this is safe on non-TTY paths.
        with live_toolbar:
            live_toolbar.refresh(state)
            try:
                metrics = chat_session.chat(
                    text,
                    sampling_params=params,
                    stream_to=_stream_callback,
                )
            except KeyboardInterrupt:
                if state.stream_state in (
                    StreamState.PREFILL,
                    StreamState.THINKING,
                ):
                    _clear_phase_indicator()
                # Clear the toolbar line before printing the abort
                # marker so the marker lands on a fresh line rather
                # than overlapping the toolbar text. The backend
                # stays active; ``with`` exit handles final teardown.
                live_toolbar.clear()
                sys.stdout.write(
                    "\n"
                    + palette.colorize("[generation aborted]", "red")
                    + "\n"
                )
                sys.stdout.flush()
                # CHAT-CLI-HARDENING-4 (F4): if this turn was a
                # regenerate, the pre-pop snapshot restores the
                # original ``(user, assistant)`` pair that
                # ``pop_last_exchange`` dropped, plus the trailing
                # user-only message that ``ChatSession.chat``
                # appended before the abort. Without this, the
                # user permanently loses the prior turn whenever
                # regenerate is interrupted.
                if regenerate_snapshot is not None:
                    chat_session.replace_messages(regenerate_snapshot)
                    if regenerate_thinking_snapshot is not None:
                        state.last_turn_thinking = (
                            regenerate_thinking_snapshot
                        )
                state.stream_state = StreamState.IDLE
                continue
            except Exception as exc:  # pragma: no cover — defensive
                if state.stream_state in (
                    StreamState.PREFILL,
                    StreamState.THINKING,
                ):
                    _clear_phase_indicator()
                live_toolbar.clear()
                sys.stdout.write(
                    "\n"
                    + palette.colorize(f"[error: {exc}]", "red")
                    + "\n"
                )
                sys.stdout.flush()
                if regenerate_snapshot is not None:
                    chat_session.replace_messages(regenerate_snapshot)
                    if regenerate_thinking_snapshot is not None:
                        state.last_turn_thinking = (
                            regenerate_thinking_snapshot
                        )
                state.stream_state = StreamState.IDLE
                continue
            # Drain any text the parser held back as a partial-tag
            # candidate (e.g. ``<th`` at end of stream without
            # follow-up).
            for event in parser.finish():
                if isinstance(event, ThinkingChunk):
                    state.last_turn_thinking += event.text
                elif isinstance(event, ReplyChunk):
                    _emit_assistant_prefix_once()
                    _emit_reply_text(event.text)
            # Drain any text the fence parser held back. A truncated
            # fence yields a final ExitFence so the highlighter
            # still gets to emit; a held-back partial-open marker
            # becomes plain text and flushes out untouched.
            for fevent in fence_parser.finish():
                if isinstance(fevent, PlainText):
                    if fevent.text:
                        sys.stdout.write(fevent.text)
                        sys.stdout.flush()
                elif isinstance(fevent, ExitFence):
                    _clear_phase_indicator()
                    highlighted = _highlight_code(
                        fevent.code, fevent.language
                    )
                    sys.stdout.write("\n" + highlighted)
                    if not highlighted.endswith("\n"):
                        sys.stdout.write("\n")
                    sys.stdout.flush()

            # Empty-reply edge case: generation ended without
            # emitting a single reply token (everything was
            # thinking, or no tokens at all). Surface a placeholder
            # so the log line still shows ``silica ›`` for visual
            # consistency.
            if not prefix_emitted[0]:
                _clear_phase_indicator()
                sys.stdout.write(
                    _format_assistant_prefix(palette)
                    + palette.colorize(
                        "(no reply — try /expand to see the model's reasoning)"
                        if state.last_turn_thinking
                        else "(no reply)",
                        "grey",
                        dim=True,
                    )
                )
                sys.stdout.flush()

        # ``with`` exited — toolbar reserved line cleared. Turn-end
        # newline now lands on a clean line.
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Update post-turn state.
        state.stream_state = StreamState.IDLE
        state.turn += 1
        state.last_ttft_ms = metrics.ttft_ms
        if metrics.peak_memory_mb is not None:
            state.peak_memory_mb = metrics.peak_memory_mb
        if metrics.decode_tok_s is not None:
            state.tok_per_sec = metrics.decode_tok_s
        # Cumulative session counters drive ``/showcase`` — total
        # prefix-token reuse, average decode tok/s, etc.
        if metrics.prefix_hit_tokens:
            state.total_prefix_hit_tokens += metrics.prefix_hit_tokens
        if (
            metrics.decode_tok_s is not None
            and metrics.decode_tok_s > 0
            and metrics.output_tokens
        ):
            state.total_decode_tokens += metrics.output_tokens
            state.total_decode_seconds += (
                metrics.output_tokens / metrics.decode_tok_s
            )
        # KV display: between turns the active KV cache is reclaimed
        # and ``engine.kv_manager.budget()`` reports zero, but the
        # prefix store still holds the cumulative cached blocks
        # from every completed turn. The user's "how much KV is in
        # use right now" answer is the prefix-store figure, not the
        # active figure. When the prefix store is empty (single-
        # request session, first turn before any blocks insert),
        # fall back to the engine's resident_mb so the field is
        # never permanently None.
        if metrics.prefix_store_resident_bytes is not None:
            state.kv_resident_mb = (
                metrics.prefix_store_resident_bytes / 1e6
            )
            state.prefix_store_mb = (
                metrics.prefix_store_resident_bytes / 1e6
            )
        elif metrics.resident_mb is not None:
            state.kv_resident_mb = metrics.resident_mb
        if metrics.prefix_store_logical_bytes is not None:
            state.kv_logical_mb = (
                metrics.prefix_store_logical_bytes / 1e6
            )
        elif metrics.logical_kv_bytes is not None:
            state.kv_logical_mb = metrics.logical_kv_bytes / 1e6
        # Tier 2 prefix-hit signal: surface block-aligned cache
        # reuse on the toolbar. Denominator is the prompt's total
        # token count so users see "256/640" — i.e. "256 tokens of
        # the 640-token prompt were fetched from the cache".
        if metrics.prefix_hit_blocks is not None:
            state.prefix_hit_blocks = metrics.prefix_hit_tokens
            state.prefix_hit_max = metrics.prompt_tokens

        # Codec hint (toolbar-adjacent, post-turn).
        threshold = float(state.config.get("kv_codec_hint_mb", 200.0))
        hint = render_codec_hint(
            state, palette=palette, threshold_mb=threshold
        )
        if hint is not None:
            sys.stdout.write(hint + "\n")
            sys.stdout.flush()

        _print_separator(palette)

    return 0


def _handle_session_save(
    path: str,
    *,
    chat_session: Any,
    state: ChatCliState,
    palette: Palette,
) -> None:
    """Persist the running session to ``path`` as JSON. Errors are
    reported as red feedback; success prints the resolved path
    so the user sees where the file actually landed (``~`` is
    expanded, parent directories are created)."""
    try:
        resolved = save_session(
            path,
            model=state.model_name,
            codec_id=state.codec_id,
            messages=chat_session.messages,
            config=dict(state.config),
        )
    except (SessionFileError, OSError) as exc:
        sys.stdout.write(
            palette.colorize(f"/save failed: {exc}", "red") + "\n"
        )
        sys.stdout.flush()
        return
    sys.stdout.write(
        palette.colorize(
            f"saved {len(chat_session.messages)} messages to {resolved}",
            "cyan",
            dim=True,
        )
        + "\n"
    )
    sys.stdout.flush()


def _handle_session_load(
    path: str,
    *,
    chat_session: Any,
    state: ChatCliState,
    palette: Palette,
) -> bool:
    """Restore a saved session from ``path``. Returns ``True`` on
    success so the caller can swap in a fresh prefix cache (the
    restored history was produced against a different sequence
    of inserts; reusing the old cache would leak stale blocks).

    A model mismatch between the file's ``model`` field and the
    currently-loaded model is **not** an error — the user may have
    saved with one model and loaded into another deliberately. We
    print a yellow warning so the discrepancy is visible without
    blocking the restore.
    """
    try:
        data = load_session(path)
    except SessionFileError as exc:
        sys.stdout.write(
            palette.colorize(f"/load failed: {exc}", "red") + "\n"
        )
        sys.stdout.flush()
        return False

    saved_model = str(data.get("model") or "")
    saved_basename = saved_model.split("/", 1)[-1]
    if saved_basename and saved_basename != state.model_name:
        sys.stdout.write(
            palette.colorize(
                f"/load: file was saved under {saved_basename!r}; "
                f"running against {state.model_name!r} — "
                "tokenisation may differ.",
                "yellow",
                dim=True,
            )
            + "\n"
        )
        sys.stdout.flush()

    messages: list[dict[str, str]] = list(data["messages"])
    chat_session.replace_messages(messages)

    # Re-apply config overrides from the file. Only keys that look
    # like our schema (string-coerce-able) are accepted; unknown
    # keys are kept as-is so a forward-compatible file does not
    # lose data on the round trip.
    saved_config = data.get("config") or {}
    if isinstance(saved_config, dict):
        for k, v in saved_config.items():
            state.config[str(k)] = v

    # Reset the cumulative session counters and the per-turn
    # snapshot fields. The toolbar's "live" signals (tok/s, ttft)
    # have no meaning until the first post-load turn runs.
    state.turn = sum(1 for m in messages if m.get("role") == "assistant")
    state.last_turn_thinking = ""
    state.prefix_hit_blocks = None
    state.prefix_hit_max = None
    state.total_prefix_hit_tokens = 0
    state.total_decode_tokens = 0
    state.total_decode_seconds = 0.0
    state.tok_per_sec = None
    state.last_ttft_ms = None
    state.tokens_generated = 0

    sys.stdout.write(
        palette.colorize(
            f"loaded {len(messages)} messages from {path}",
            "cyan",
            dim=True,
        )
        + "\n"
    )
    sys.stdout.flush()
    return True


def _swap_model(
    new_repo: str,
    *,
    args: argparse.Namespace,
    state: ChatCliState,
    keep_history: bool = False,
    prior_session: Any = None,
    get_adapter: Any,
    engine_cls: Any,
    session_cls: Any,
    cache_builder: Any,
    palette: Palette,
) -> tuple[Any, Any, Any] | None:
    """Re-load the model with id ``new_repo`` and rebuild the
    chat session around it. On success returns ``(adapter, engine,
    chat_session)`` so the caller can rebind its locals; on failure
    returns ``None`` and the previous (adapter, engine, chat_session)
    triple stays live.

    System prompt is always preserved (read from
    ``state.config["system_prompt"]``). Conversation history is
    preserved iff ``keep_history`` is True (CHAT-CLI-HARDENING-5 /
    F5): the new ``ChatSession`` is repopulated with the prior
    session's text-level message log via ``replace_messages``, and
    the new tokeniser re-tokenises the stored text on the next
    turn. The fresh prefix cache is empty, so the first
    post-swap turn's prefill runs full-cost — only the message
    text is preserved, not the cache state.

    Default ``keep_history=False`` matches the pre-HARDENING-5
    behaviour: history is dropped and the user gets a notice
    explaining why. This default avoids silently inheriting a
    stale conversation when the user picks a model whose chat
    template / EOS handling differs incompatibly.
    """
    sys.stdout.write(
        palette.colorize(
            f"loading {new_repo} (this takes a moment) ...",
            "grey",
            dim=True,
        )
        + "\n"
    )
    sys.stdout.flush()
    try:
        new_adapter, new_kv = get_adapter(new_repo)
    except Exception as exc:
        sys.stdout.write(
            palette.colorize(
                f"/model failed: {exc}", "red"
            )
            + "\n"
        )
        sys.stdout.flush()
        return None
    # Capture text history before building the new session — both
    # the capture and ``replace_messages`` deep-copy entries, so the
    # prior session is no longer referenced after this call. Note
    # that downstream construction failures (engine_cls /
    # session_cls / cache_builder raising) propagate past
    # ``_swap_model``; only the ``get_adapter`` raise is caught.
    captured_messages: list[dict[str, str]] | None = None
    if keep_history and prior_session is not None:
        captured_messages = prior_session.messages
    new_engine = engine_cls(new_adapter, new_kv)
    new_cache = cache_builder(new_adapter)
    sys_prompt = state.config.get("system_prompt") or None
    sys_prompt_str = str(sys_prompt) if sys_prompt else None
    new_session = session_cls(
        new_adapter,
        new_engine,
        system_prompt=sys_prompt_str,
        prefix_cache=new_cache,
    )
    if captured_messages is not None:
        # ``replace_messages`` deep-copies entries; the prior
        # session is no longer referenced after this call.
        new_session.replace_messages(captured_messages)

    # Reset model-derived state regardless of keep_history — the
    # cache, KV manager, and per-turn metric figures all refer to
    # the swapped-in model and would mislead if carried forward.
    # ``tok_per_sec`` is the live decode speed for the in-flight
    # turn (see ChatCliState.tok_per_sec docstring), not a
    # conversation-level rolling figure, so it resets here too.
    state.model_name = _model_basename(new_repo)
    state.prefix_hit_blocks = None
    state.prefix_hit_max = None
    state.last_ttft_ms = None
    state.tokens_generated = 0
    state.tok_per_sec = None
    state.kv_resident_mb = None
    state.kv_logical_mb = None
    state.prefix_store_mb = None
    # Conversation-level state: keep iff the history did.
    if not keep_history:
        state.turn = 0
        state.last_turn_thinking = ""
        state.total_prefix_hit_tokens = 0
        state.total_decode_tokens = 0
        state.total_decode_seconds = 0.0
    # Carry the codec_id / system prompt forward unchanged.
    state.codec_id = getattr(args, "kv_codec", None)

    if keep_history:
        notice = (
            f"model swapped to {state.model_name}. "
            "Conversation history preserved; the new tokeniser will "
            "re-tokenise stored messages on the next turn."
        )
    else:
        notice = (
            f"model swapped to {state.model_name}. Conversation "
            "history reset (tokenisation differs across models)."
        )
    sys.stdout.write(
        palette.colorize(notice, "cyan", dim=True) + "\n"
    )
    sys.stdout.flush()
    return new_adapter, new_engine, new_session


_PREFIX_CACHE_BLOCK_SIZE = 4
"""Block size for the chat-CLI's prefix cache.

The bench harness uses 16 (matches `_maybe_build_prefix_cache`).
Chat is different — between turns, the chat template re-renders the
conversation and the deterministic shared prefix grows by ~10-30
tokens per turn (one user message + chat-template wrapping). Block
sizes larger than that boundary lose ALL prefix reuse on short
turns. Concrete example with Qwen3.5-4B and a `Hi, who are you?`
opening: turn 1's prompt is 16 tokens ending in `<think>\\n`, but
turn 2's prompt at position 14 starts the assistant message text;
the 14-token shared prefix is below `block_size=16` so 0 blocks
reuse. With `block_size=4`, the same 14-token prefix yields 3
blocks reused = 12 tokens — and the reuse grows linearly with
conversation length thereafter. The trade-off is more nodes in
the radix tree (negligible at chat scale).
"""


def _build_prefix_cache(
    adapter: Any,
    *,
    codec_id: str | None,
    get_codec_spec: Any,
    store_cls: Any,
    cache_cls: Any,
) -> Any:
    """Construct a session-scoped ``RadixPrefixCache`` for the chat REPL.

    When ``codec_id`` is ``None`` the cache is fp16 (no compression);
    otherwise the named codec from
    ``silica.bench.codec_registry`` is instantiated against the
    adapter's KV layout and installed on a
    :class:`SyntheticPrefixBlockStore`.

    Helper-injected arguments mirror the imports inside
    :func:`run_chat` so this builder stays pure-Python and can be
    unit-tested without the heavy MLX / engine warm-up — see
    ``tests/test_chat_cli_app_prefix_cache_factory.py`` (planned
    follow-up; for now the function is exercised end-to-end via
    the live REPL smoke).
    """
    layout = adapter.kv_layout()
    codec: Any = None
    if codec_id is not None:
        spec = get_codec_spec(codec_id)
        codec = spec.factory(
            block_size=_PREFIX_CACHE_BLOCK_SIZE,
            n_kv_heads=layout.n_kv_heads,
            head_dim=layout.head_dim,
            dtype=layout.dtype,
            seed=42,
        )
    store = store_cls(
        block_size=_PREFIX_CACHE_BLOCK_SIZE, codec=codec
    )
    return cache_cls(
        block_size=_PREFIX_CACHE_BLOCK_SIZE, store=store
    )


def _sampling_params_from_state(
    state: ChatCliState, adapter: Any
) -> Any:
    """Build :class:`silica.core.sampling.SamplingParams` from
    state.config + the model's tokeniser EOS ids.

    Reads runtime overrides from ``/config``. Falls back to schema
    defaults when a key has not been overridden mid-session.
    """
    from silica.core.sampling import SamplingParams

    tokenizer = adapter.tokenizer()
    eos_ids = tuple(sorted(getattr(tokenizer, "eos_token_ids", set()) or ()))
    raw_top_k = state.config.get("top_k")
    top_k_int: int | None = (
        int(raw_top_k) if isinstance(raw_top_k, int) else None
    )
    return SamplingParams(
        temperature=float(state.config.get("temperature", 0.7)),
        top_p=float(state.config.get("top_p", 0.9)),
        top_k=top_k_int,
        max_tokens=int(state.config.get("max_tokens", 1024)),
        stop_token_ids=eos_ids,
    )


__all__ = ["run_chat"]
