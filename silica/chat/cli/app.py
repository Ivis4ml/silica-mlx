"""silica.chat.cli.app — prompt_toolkit chat REPL (C-3).

Wires the C-1 palette / state / toolbar and the C-2 slash-command
dispatcher into a live REPL using ``prompt_toolkit``'s
``PromptSession``. Streams tokens from ``ChatSession.chat`` to the
terminal between prompts; the bottom toolbar refreshes during
input phases. Live mid-generation toolbar updates land in C-5
(full-Application layout); C-3 ships the working end-to-end shell.

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

Manual smoke checklist — see ``docs/CHAT_CLI_OPENING.md`` §10.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from silica.chat.cli.commands import (
    CommandResult,
    dispatch_command,
    is_slash_command,
)
from silica.chat.cli.config import initial_config
from silica.chat.cli.palette import ColorName, Palette, detect_palette
from silica.chat.cli.state import ChatCliState, StreamState
from silica.chat.cli.toolbar import render_codec_hint, render_toolbar


def _model_basename(repo: str) -> str:
    """Strip an HF org prefix and a trailing dtype suffix.

    ``Qwen/Qwen3-0.6B`` → ``Qwen3-0.6B``;
    ``mlx-community/Qwen3.5-35B-A3B-4bit`` → ``Qwen3.5-35B-A3B-4bit``.
    Used for the ``model=`` toolbar field — the prefix is consistent
    across vendors and adds no signal to the user.
    """
    return repo.split("/", 1)[-1]


def _format_user_input(text: str, palette: Palette) -> str:
    """Render the echoed user line in the conversation log."""
    prefix = palette.colorize("You ›", "green", bold=True)
    body = palette.colorize(text, "green_dim")
    return f"{prefix} {body}"


def _format_assistant_prefix(palette: Palette) -> str:
    return palette.colorize("silica ›", "orange", bold=True) + " "


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
    chat_session = ChatSession(
        adapter,
        engine,  # type: ignore[arg-type]
        system_prompt=args.system,
        prefix_cache=prefix_cache,
    )

    # --- Build state ---
    state = ChatCliState(
        model_name=_model_basename(args.model),
        codec_id=getattr(args, "kv_codec", None),
    )
    state.config.update(initial_config())
    if args.system:
        state.config["system_prompt"] = args.system
    state.stream_state = StreamState.IDLE

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
        # Bottom toolbar refreshes only during the prompt; that's
        # acceptable for C-3 — live mid-generation toolbar lands
        # at C-5 with a full-Application layout.
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
            if result.request_expand_thinking:
                expanded = palette.colorize(
                    "── thinking ──\n" + state.last_turn_thinking,
                    "grey",
                    dim=True,
                )
                sys.stdout.write(expanded + "\n\n")
                sys.stdout.flush()
            if result.request_regenerate:
                # C-4 will wire actual regenerate; for v1 print a
                # "not yet implemented" notice so the command surface
                # exists but the heavy lift waits its sub-unit.
                print(
                    palette.colorize(
                        "(/regenerate lands at C-4; not wired yet)",
                        "yellow",
                        dim=True,
                    )
                )
            if result.request_session_save or result.request_session_load:
                print(
                    palette.colorize(
                        "(/save and /load land at C-7; not wired yet)",
                        "yellow",
                        dim=True,
                    )
                )
            if result.request_model_swap:
                print(
                    palette.colorize(
                        "(/model swap lands at C-7; not wired yet)",
                        "yellow",
                        dim=True,
                    )
                )
            if result.request_showcase:
                print(
                    palette.colorize(
                        "(/showcase lands at C-7; not wired yet)",
                        "yellow",
                        dim=True,
                    )
                )
            continue

        # Regular chat turn.
        params = _sampling_params_from_state(state, adapter)
        sys.stdout.write(_format_assistant_prefix(palette))
        sys.stdout.flush()

        state.stream_state = StreamState.PREFILL
        state.tokens_generated = 0
        state.max_tokens = int(state.config.get("max_tokens", 1024))
        try:
            metrics = chat_session.chat(
                text,
                sampling_params=params,
                stream_to=lambda delta: _on_token(delta, state),
            )
        except KeyboardInterrupt:
            sys.stdout.write(
                "\n"
                + palette.colorize("[generation aborted]", "red")
                + "\n"
            )
            sys.stdout.flush()
            state.stream_state = StreamState.IDLE
            continue
        except Exception as exc:  # pragma: no cover — defensive
            sys.stdout.write(
                "\n"
                + palette.colorize(f"[error: {exc}]", "red")
                + "\n"
            )
            sys.stdout.flush()
            state.stream_state = StreamState.IDLE
            continue

        # Newline closes the streamed assistant line.
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Update post-turn state.
        state.stream_state = StreamState.IDLE
        state.turn += 1
        state.last_ttft_ms = metrics.ttft_ms
        if metrics.peak_memory_mb is not None:
            state.peak_memory_mb = metrics.peak_memory_mb
        if metrics.resident_mb is not None:
            state.kv_resident_mb = metrics.resident_mb
        if metrics.logical_kv_bytes is not None:
            state.kv_logical_mb = metrics.logical_kv_bytes / 1e6
        if metrics.decode_tok_s is not None:
            state.tok_per_sec = metrics.decode_tok_s
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


_PREFIX_CACHE_BLOCK_SIZE = 16
"""Matches the block size used in :func:`silica.bench.runner._maybe_build_prefix_cache`
so chat-CLI prefix caches share the size convention with bench rows."""


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


def _on_token(delta: str, state: ChatCliState) -> None:
    """Stream callback — write the decoded token delta to stdout
    and update the lightweight tokens-generated counter.

    ``ChatSession.chat`` calls this once per emitted token. C-5
    will hook richer per-token bookkeeping (live tok/s, state
    machine transitions to ``DECODE``); for C-3 the callback is
    deliberately minimal so the streaming output is byte-for-byte
    what the model produced.
    """
    if state.stream_state is StreamState.PREFILL:
        # First token observed → we are decoding. C-8 will
        # interpose <think> detection here; C-3 just transitions.
        state.stream_state = StreamState.DECODE
    state.tokens_generated += 1
    sys.stdout.write(delta)
    sys.stdout.flush()


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
