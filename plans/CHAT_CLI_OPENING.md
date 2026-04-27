# Chat CLI redesign — opening doc

| Field        | Value                                                              |
| ------------ | ------------------------------------------------------------------ |
| Track        | Side track 2 - outside the P-1..P-5 mainline, ahead of P-6/7/8/9   |
| Status       | design                                                             |
| Last updated | 2026-04-26                                                         |
| Related      | `scripts/chat.py`, `silica/chat/session.py`                        |

## 1. Why redesign

The current `scripts/chat.py` is functionally complete (REPL loop,
streaming, /reset / /stats / /exit, per-turn metrics line on stderr)
but does not match where silica-mlx is heading. Two specific gaps:

1. **The chat client does not exercise silica's own prefix cache.**
   `ChatSession` rebuilds the full conversation as a fresh prompt
   on every turn and runs `engine.generate(prompt)` — single-request,
   no `RadixPrefixCache`. The framework's signature feature
   (cross-turn KV reuse) is unused in the framework's own client.

2. **The CLI is not a showcase for the framework's advantages.**
   silica-mlx's value over running `mlx-lm` directly is: native MLX
   speed on Apple Silicon, KV codec compression, memory savings on
   long conversations via prefix reuse. The current chat surface
   exposes none of these to the user — there is no live signal of
   speed, no visible memory state, no compression ratio.

## 2. Long-term direction (out of scope for side track 2)

The user's stated long-term goal is a Claude Code / GPT Codex-style
local app on top of silica-mlx, after P-6 / P-7 / P-8 / P-9 land.
Design choices in side track 2 should be **directionally compatible**
with that goal:

- The CLI should feel like a conversational interface, not a
  parameterised `argparse` form. Sampling knobs (`--temperature`,
  `--top-p`, `--max-tokens`) should not be the primary user
  surface.
- The bottom toolbar should leave room for additional state in
  future (current model, current session, tools loaded, agent
  step counter).
- The conversation log should support code fences with syntax
  highlighting since the eventual users are coders.

Side track 2 explicitly does **not** add: tool calling, file edits,
bash execution, project context awareness, agentic loops,
sub-agents, diff approval flows, or sandboxing. Those are side
tracks 3..6, sequenced after P-6/7/8/9.

## 3. Scope

### 3.1 In scope (Tier 1 — user-experience layer)

- **prompt_toolkit-based REPL.** Replace the raw `input()` loop.
  Gives multi-line editing, ↑/↓ history navigation, Ctrl-R search,
  Ctrl-C interruption with first-press = abort current generation
  and second-press = exit, persistent bottom toolbar — all
  standard prompt_toolkit affordances.
- **Bottom status bar.** Persistent toolbar pinned to the bottom
  row. Updates in real time during generation. Fields below.
- **Conversation log.** Above the toolbar. Each turn rendered as:
  - User input: bold + colour A, prefixed `You ›`.
  - Assistant reply: streamed colour B, prefixed `silica ›`.
  - Code fences inside assistant output: syntax-highlighted via
    `pygments` (built-in language detection).
  - Blank line + thin separator between turns.
- **No primary-surface sampling knobs.** `silica-chat --model X`
  should just work. Sampling defaults (temperature 0.7, top-p 0.9,
  top-k None, max-tokens 1024) live as chat-style sensible
  defaults, *not* as required CLI arguments. Power users can run
  `/config temperature=0.5` inside the REPL to override mid-session.
- **Slash commands.** `/help` enumerates everything. `/reset`,
  `/exit`, `/stats` (now superseded by the always-visible toolbar
  but kept for cumulative session totals), `/system "..."`,
  `/regenerate`, `/save path.json`, `/load path.json`,
  `/config <key>=<value>`, `/model <repo>` (warm reload).
- **`NO_COLOR` honoured.** Standard environment variable to disable
  ANSI colour output for users who pipe to files or run in
  colour-hostile terminals.
- **Single-command launch — `silica` opens chat.** The user types
  `silica` (no subcommand, no flags) and the chat REPL opens
  immediately on a sensible default model. Mirrors the
  `claude` CLI ergonomic. `silica chat --model X` is the explicit
  form; `silica run` for single-shot stays as today. See §7
  Module layout / launcher wiring for details.

### 3.2 In scope (Tier 2 — architectural showcase)

- **`ChatSession` integrates `RadixPrefixCache`.** Each turn passes
  the cumulative chat-template-rendered prompt through
  `engine.generate_batch([prompt], prefix_cache=session_pc)`. After
  the first turn, every subsequent turn's prefill phase pays only
  for the *delta* (the new user message + previous assistant tokens
  that were not yet block-aligned).
- **Toolbar surfaces the prefix-cache effect.** A `prefix_hit=N/M`
  field displays "M tokens of the current prompt matched the cache;
  N of those landed on a block boundary so were reused as KV". When
  the conversation is long enough that block-aligned reuse fires,
  the user sees prefill TTFT drop visibly turn-over-turn — the
  silica-mlx vs. mlx-lm difference made visible at the user surface.

### 3.2.1 Thinking-mode interaction

Qwen3 / Qwen3.5 family models emit `<think>...</think>` blocks
before the actual reply when reasoning mode is on (the default for
those families). Streamed verbatim, the thinking text floods the
conversation log with internal chain-of-thought; the user sees
multi-paragraph deliberation before any usable answer arrives.
Modern chat apps (Claude, ChatGPT) collapse this into a single-line
"Thinking..." indicator and stream the actual reply afterwards. The
silica chat CLI should match this behaviour.

Streaming state machine (extends §4 toolbar's state field):

  - `idle → prefill → thinking → replying → idle`
  - The transition from `prefill` to `thinking` triggers when the
    first streamed token is or is followed by `<think>`.
  - The transition from `thinking` to `replying` triggers when
    `</think>` appears in the stream. Tokens after that flow into
    the assistant reply normally.
  - A model with no `<think>` tag (e.g. thinking disabled, or a
    non-reasoning family) skips `thinking` entirely:
    `prefill → replying`.

Conversation log rendering:

  - During `thinking`: a single line `silica › ⠋ thinking... 4.2s`
    (Braille spinner + elapsed seconds) replaces what would be
    streamed text. Thinking tokens are buffered to `ChatCliState.
    last_turn_thinking` but not rendered into the main log.
  - On `</think>`: the spinner line is rewritten as
    `silica › thought for 6.3s` (or removed entirely under
    `thinking_display=hidden`) and the actual reply begins
    streaming below.
  - Slash command `/expand` reprints the previous turn's thinking
    text with a dimmed-italic style so the user can inspect what
    the model reasoned through. `/expand <turn_idx>` for older
    turns.

Config and control:

  - `/config thinking=auto` (default) — collapse during stream,
    `/expand` available afterwards.
  - `/config thinking=show` — render thinking text inline as it
    streams, dimmed italic.
  - `/config thinking=hidden` — collapse during stream, do not even
    show the elapsed-time line; thinking is silent.
  - `/config thinking_mode=on|off` — passes `enable_thinking=True/
    False` into the chat template (Qwen3 / Qwen3.5 honour the
    flag). Off makes the model skip reasoning entirely; faster
    TTFT and shorter total tokens, at the cost of reasoning depth.

Edge cases:

  - Thinking interrupted by `max_tokens`: show
    `[thinking truncated at N tokens]` before any partial reply.
  - Model with no thinking support: `<think>` never appears, state
    machine never enters `thinking`, no behaviour change.
  - Nested `<think>` tags: not produced by Qwen3 in practice;
    unspecified — buffer all and exit on first `</think>`.

This is implemented in C-8 of §8 sub-units below.

### 3.3 Out of scope (deferred to side track 3+)

- Tool calling / file edits / bash execution / web fetch.
- Agent loops (model proposes a tool call → runtime executes →
  result fed back).
- Project context auto-loading (`CLAUDE.md`-style protocol).
- Diff approval flow for destructive operations.
- Permission / sandbox layer.
- Multi-agent / sub-agent orchestration.
- Speculative decoding integration (P-7 still stub).
- Per-expert MoE residency demonstration (P-6 still stub).
- HTTP server endpoints (P-8 still stub).

## 4. Bottom toolbar — field list

The toolbar is one or two physical lines at the bottom of the
terminal, updated continuously while a generation is running and
sticky-rendered between turns. Fields, left to right:

| Field | Source | Meaning |
| --- | --- | --- |
| `state` | scheduler state machine + stream parser | one of `idle` / `prefill` / `thinking` / `decode` / `paused` — what the engine + parser is doing right now |
| `model` | `--model` / `/model` | basename of the currently-loaded HF repo (e.g. `Qwen3-0.6B`) |
| `MLX` badge | always | static "MLX" marker (the silica-mlx-vs-everything-else signal) |
| `tok/s` | `engine.metrics` | live decode tok/s during the decode phase; "—" otherwise |
| `ttft` | last completed turn | time to first token of the most recent reply |
| `peak` | `mx.get_peak_memory()` | device peak memory in MB / GB |
| `kv_resident` | `engine.metrics.resident_kv_bytes` | resident KV bytes in MB |
| `kv_logical` | `engine.metrics.logical_kv_bytes` | what the same data would cost at fp16 (codec-relevant) |
| `compr` | `kv_logical / kv_resident` | KV compression ratio (only when codec is active; "—" under fp16) |
| `prefix_hit` | `RadixPrefixCache` stats | tokens reused from the cache on the most recent admit (Tier 2) |
| `turn` | `ChatSession` | current turn index |

A second toolbar line (optional, toggleable via `/config toolbar=2`)
surfaces session-level totals: cumulative input / output tokens,
average TTFT, average decode tok/s, wall-clock since session start.

## 4.1 Colour palette

A single default theme. User preference: green and orange as the
primary identity colours. Palette uses 24-bit ANSI where supported,
8-colour fallback under `TERM=xterm` etc. `NO_COLOR=1` disables
the entire layer.

| Element | Colour | Notes |
| --- | --- | --- |
| `You ›` user prompt prefix | bright green | user input identity |
| `silica ›` assistant prefix | bright orange | brand identity |
| User input text (echoed) | green (medium) | matches the prompt prefix |
| Assistant streamed text | default foreground | readable on any terminal |
| Toolbar `MLX` badge | bright orange + bold | brand affordance |
| Toolbar `state=decode` | bright green | "we're producing tokens right now" |
| Toolbar `state=thinking` | bright magenta | "model is reasoning inside `<think>...</think>`" |
| Toolbar `state=prefill` | yellow | "consuming prompt tokens" — informs user the wait is normal |
| Toolbar `state=idle` | dim gray | resting |
| Toolbar `tok/s` | cyan | the headline performance number |
| Toolbar `compr=Xx` | bright orange | KV codec savings (silica-only signal) |
| Toolbar `prefix_hit=N/M` | cyan | architectural-win signal (Tier 2) |
| Toolbar `peak` / `kv_*` | dim white | secondary stats |
| Slash command echo | dim cyan | `/help`, `/reset` etc. |
| Error text (model load fail, etc.) | red | hard-error signalling |
| Code fences inside reply | pygments theme `monokai` | syntax-highlighted, dark background friendly |
| Turn separator line | dim gray | thin line between turns |

Rationale: green = user voice (active, bright); orange = silica
identity (brand colour, used for the MLX badge and the codec
compression callout); cyan = "look here, this is the framework
working" (tok/s, prefix_hit) — pulls the eye to the silica-mlx
USP signals.

## 5. Showcasing silica-mlx's advantages

Concrete signals the user sees that they would not see on `mlx-lm`:

1. **MLX badge in the toolbar** — static brand affordance.
2. **`compr=Xx` field** — when running with `--kv-codec block_tq_b64_b4`,
   the user sees "compr=3.8x" in real time. Try the same with mlx-lm:
   the field is "—".
3. **`prefix_hit=N/M` after a few turns** — by turn 3 of a long
   conversation, prefix_hit shows "256/640" or similar; the
   accompanying `ttft` drops sharply. The narrative: "silica-mlx
   reuses your conversation prefix block-by-block; mlx-lm re-runs
   prefill on the full history every turn."
4. **`kv_logical` vs `kv_resident`** — explicit memory savings.
   With BlockTQ b64 b4 on a long context, `kv_logical=120MB` /
   `kv_resident=32MB` is the kind of split that justifies running a
   compressed KV at all.
5. **`tok/s` live during decode** — peak Apple Silicon throughput
   visible in real time, no stopwatch needed.

A `/showcase` command (Tier 1) prints a one-paragraph report
summarising the current session in narrative form: "this session
ran X turns, reused Y tokens of prefix, saved Z MB vs fp16
baseline, peaked at A GB device memory, averaged B tok/s decode."
Useful for screenshots and for first-time users to feel the
difference.

## 6. Sampling defaults — what the user does NOT have to set

| Parameter | Default | Override path |
| --- | --- | --- |
| `temperature` | 0.7 | `/config temperature=<value>` mid-session |
| `top_p` | 0.9 | `/config top_p=<value>` |
| `top_k` | None (off) | `/config top_k=<value>` |
| `max_tokens` | 1024 | `/config max_tokens=<value>` (or per-turn auto-extend on EOS) |
| `system_prompt` | empty | `--system "..."` at launch or `/system "..."` mid-session |
| `kv_codec` | none (fp16) | `--kv-codec block_tq_b64_b4` at launch (chat doesn't switch codec mid-session in v0.1) |

Launch surface intentionally minimal:

```text
silica                                              # chat with all defaults (Qwen3-0.6B + fp16)
silica --model Qwen/Qwen3-4B                         # bare silica accepts chat flags
silica --kv-codec block_tq_b64_b4                    # bare silica accepts codec flag
silica --kv-codec block_tq_b64_b4 --model Qwen/Qwen3-4B
silica chat --model Qwen/Qwen3-0.6B                  # explicit subcommand form, equivalent
silica chat --kv-codec block_tq_b64_b4 --system "You are concise."
silica run --prompt "..."                            # single-shot path, unchanged from today
```

Bare `silica` and `silica chat` are equivalent. Implementation: in
`silica/server/cli.py:main`, pre-process `argv` — if `argv[0]` is
not a known subcommand (`run` / `chat` / `--help` / `--version`),
prepend `chat` so all chat flags land on the chat subparser. This
is the same pattern `pytest`, `cargo`, and many CLIs use; it keeps
the root parser simple (single source of truth for chat flags
lives on the chat subparser).

Everything else lives behind `/config` so the launch flow feels
like opening a chat app, not configuring a tool.

### 6.1 Codec default — why fp16

The `kv_codec` row above defaults to `none (fp16)`, not to
`block_tq_b64_b4`, even though the v1.7.7 (b-static) close gate
shows BlockTQ b64 b4 is statistically indistinguishable from fp16
on Qwen3.5-4B WikiText-2. Three reasons:

1. **Cross-model safety.** The lossless-at-measurement-precision
   result is on Qwen3.5-4B. On Qwen3-0.6B (the default bare-launch
   model) the D.2a 3-seed measurement showed mean ΔPPL = +0.51 PPL
   — small but visible. Codec losses scale inversely with model
   size (smaller models are more sensitive to KV reconstruction
   noise). Defaulting to codec would silently change quality on
   the default model without user consent.
2. **Single-user short-chat economics.** The codec compresses what
   sits in the prefix store, not what flows through active
   `BatchKVCache`. For a single user's short chat, prefix store
   size is typically 10-100 MB; 3.8× compression saves tens of MB
   on a 48 GB machine — small absolute. The TTFT / quality / "feel"
   trade-off does not break even at this scale.
3. **No silent quality changes.** Codec activation is a meaningful
   architectural choice; the user should opt in knowingly via
   `--kv-codec` or `/config kv_codec=...`. Toolbar exposes
   `compr=—` by default so the feature is discoverable.

### 6.2 When codec actually pays off

| Workload | Prefix store size | fp16 footprint | Codec footprint | Absolute savings | Worth opting in? |
| --- | --- | --- | --- | --- | --- |
| Short chat (5-10 turns, brief prompts) | ~1-2K tokens | ~20-40 MB | ~5-10 MB | ~15-30 MB | No — under noise |
| Long chat (50+ turns, code / writing) | ~10-30K tokens | ~200-600 MB | ~50-150 MB | ~150-450 MB | Marginal |
| Single-user RAG / agent (long-doc context) | ~50-200K tokens | 1-4 GB | 0.25-1 GB | 0.75-3 GB | Yes |
| Multi-user serving (shared system prompt / few-shot) | cross-user accumulated | 5-20 GB | 1.3-5 GB | 4-15 GB | Required |
| 48 GB tight deployment (27B model + long session) | any | overflow | fits | enables run | Required |

The chat CLI's primary user is the first or second row. The codec
is structurally available from C-7 onward but **not the default**.
The toolbar surfaces a soft hint when prefix store growth crosses
a threshold (default 200 MB):

```text
tip: --kv-codec block_tq_b64_b4 would compress prefix store ~3.8x
```

This makes the feature discoverable without forcing it. The
threshold is `/config kv_codec_hint_mb=<value>`. For workloads
that consistently sit in row 3 or below (RAG, agents, multi-user),
the user opts in once at launch and toolbar shows live `compr=`
ratio.

### 6.3 Codec under multi-user serving (post P-8)

When the OpenAI HTTP server (P-8) lands, the multi-user case
becomes the default deployment shape, and the economics flip:
shared prefix across users, prefix store size dominated by
session count. At that point silica-server may default to codec on
(or auto-tune based on residency pressure) — out of scope for
side track 2 chat CLI but recorded as the trajectory.

## 7. Module layout

```text
silica/chat/
  session.py          # existing — minor changes to take prefix_cache
  cli/                # NEW
    __init__.py
    app.py            # prompt_toolkit Application + KeyBindings + run_chat()
    toolbar.py        # bottom-toolbar formatter + state struct
    commands.py       # slash command dispatch table
    formatter.py      # conversation rendering + code-fence syntax highlight
    state.py          # ChatCliState (live metrics, conversation log, config dict)
    config.py         # /config key=value parser + defaults
    palette.py        # ANSI colour constants + NO_COLOR / TERM detection
silica/server/cli.py  # existing — extended with `chat` subcommand +
                      # bare-`silica` default = chat (claude-style)
scripts/chat.py       # kept as a thin alias for `silica chat` (existing
                      # docs / muscle memory) — same `run_chat()` target
```

### Launcher wiring

The user types `silica` (no subcommand, no flags). The CLI opens
the chat REPL immediately, mirroring `claude`'s ergonomic.

`silica/server/cli.py` changes:

- `add_subparsers(dest="cmd", required=True)` →
  `add_subparsers(dest="cmd", required=False)`. When no subcommand
  is provided, the dispatcher routes to `chat` with all defaults.
- New subparser `chat` with optional `--model`, `--system`,
  `--kv-codec`. No sampling-knob flags (per §6 design rule).
- `silica chat` and bare `silica` both call into
  `silica.chat.cli.app.run_chat(args)`.
- `silica run` continues to behave as today.

`pyproject.toml` `[project.scripts]` keeps the single `silica`
entry. An optional `silica-chat` alias console script can be added
in C-7 if the user wants direct binary access without typing the
subcommand. Default model when bare `silica` is invoked: read from
`~/.config/silica/default.toml` if present, else fall back to a
hard-coded `Qwen/Qwen3-0.6B` (small, fast, broadly available).

The bulk of the new code lives under `silica/chat/cli/` so it is
unit-testable independently of `prompt_toolkit`'s event loop. The
`scripts/chat.py` shim stays small: parse CLI args, build engine
plus session, dispatch into `silica.chat.cli.app.run_chat`.

## 8. Sub-units / landing order

1. **C-1 — `palette.py` + `ChatCliState` + `Toolbar` formatter
   (no UI yet).** Pure-Python: ANSI palette constants
   (green / orange / cyan / yellow / red / dim gray, with `NO_COLOR`
   and 8-colour fallback), state struct, and
   `render_toolbar(state) -> str` producing the coloured toolbar
   line. Unit-tested in `tests/test_chat_cli_toolbar.py` against
   a fake state — both colour and `NO_COLOR=1` paths covered. Lands
   the data model + palette before any UI dependency.
2. **C-2 — `Commands` dispatch + `Config` parser.** `/help`,
   `/reset`, `/exit`, `/system`, `/regenerate`, `/save`, `/load`,
   `/config`, `/model`, `/showcase`. Lands as plain functions
   operating on `ChatCliState`. Unit-tested.
3. **C-3 — prompt_toolkit `app.py` shell.** Wires C-1 + C-2 into
   the prompt_toolkit `Application` with persistent bottom toolbar,
   conversation log buffer, key bindings (Enter to submit,
   Shift+Enter for newline, Ctrl-C semantics). Manual smoke test
   only — prompt_toolkit's event loop is hard to unit-test.
4. **C-4 — `silica/chat/session.py` integrates `RadixPrefixCache`.**
   ChatSession takes an optional `prefix_cache` arg; when provided,
   `chat()` routes through `engine.generate_batch([prompt],
   prefix_cache=pc)` and surfaces `prefix_hit_blocks` / `prefix_hit_tokens`
   on the returned `TurnMetrics`.
5. **C-5 — Live toolbar updates during generation.** Hook the
   engine's per-token callback so `tok/s` and `tokens=N/max` update
   as tokens stream. State machine transitions
   `idle → prefill → decode → idle` per turn.
6. **C-6 — `formatter.py` code fence syntax highlight.** Pygments
   integration. Detects ` ```python … ``` ` blocks in streaming
   assistant output and re-renders them with token colours.
   Streaming-friendly: only re-renders the code block when the
   closing fence arrives.
7. **C-7 — `--kv-codec` and showcase polish.** Wire `--kv-codec`
   on the chat CLI launcher (today only `scripts/bench.py` carries
   it). Bare-`silica` argv preprocessing in `silica/server/cli.py`
   lands here. `/showcase` prints the session narrative. Toolbar
   200 MB prefix-store-size hint that suggests
   `--kv-codec block_tq_b64_b4` lands here.
8. **C-8 — Thinking-mode interaction.** Stream parser tracks
   `<think>` / `</think>` boundaries; state machine extends to
   `prefill → thinking → replying`; conversation log collapses
   thinking into a single spinner line by default; `/expand` shows
   the buffered thinking from a previous turn; `/config thinking=
   {auto,show,hidden}` toggles visibility; `/config thinking_mode=
   {on,off}` propagates `enable_thinking` through the chat
   template. Unit-tested for the parser (in
   `tests/test_chat_cli_thinking_parser.py`) — fed a stream with
   and without `<think>` tags, asserts state transitions and
   buffer contents. UI integration validated manually.

C-1, C-2, C-4, C-8 (parser portion) are pure-Python and testable.
C-3, C-5, C-6, C-8 (UI portion) are interactive-only and validated
manually.

## 9. Dependencies added

- `prompt_toolkit>=3.0` — the REPL framework.
- `pygments>=2.17` — code-fence syntax highlighting.

Both are MIT, mature, dependency-light. Added to the `chat` extras
group rather than the core install so users who only need the
Python API or `silica run` single-shot do not pay the import cost.

## 10. Acceptance — when is side track 2 done

- [ ] C-1..C-8 all landed.
- [ ] **Bare `silica` opens the chat REPL** (claude-style) on a
      sensible default model with no subcommand and no flags.
- [ ] **Bare-silica accepts chat flags:**
      `silica --kv-codec block_tq_b64_b4 --model Qwen/Qwen3-4B`
      is equivalent to the explicit `silica chat ...` form.
- [ ] **Default codec is fp16.** Toolbar shows `compr=—` on a
      bare launch; codec only activates when `--kv-codec` is
      supplied at launch or `/config kv_codec=...` is issued.
- [ ] `silica chat --model Qwen/Qwen3-0.6B` launches into a chat
      with persistent bottom toolbar, no sampling-knob flags
      required.
- [ ] **Colour identity visible:** `You ›` rendered in bright
      green, `silica ›` and the toolbar `MLX` badge in bright
      orange. `state=thinking` shows in bright magenta when the
      model emits a `<think>` block. `NO_COLOR=1` flips all of
      these to plain text.
- [ ] First user message → `prefill` → first token visible →
      `decode` → finished, with toolbar live-updating throughout.
- [ ] **Thinking-mode collapse:** on a Qwen3 family model with
      reasoning enabled, the conversation log shows
      `silica › ⠋ thinking... 4.2s` during the `<think>...</think>`
      block, then the actual reply streams below. `/expand` reprints
      the buffered thinking from the previous turn.
- [ ] Three back-to-back turns with `RadixPrefixCache` active →
      `prefix_hit` toolbar field grows turn-over-turn → `ttft`
      drops correspondingly. Visible to the eye, not just in
      logs.
- [ ] `silica --kv-codec block_tq_b64_b4 --model Qwen/Qwen3-4B`
      → `compr=3.8x` (or similar) appears in toolbar, `kv_resident`
      stays small as conversation grows.
- [ ] **Hint surface:** under fp16 default, when prefix store size
      crosses 200 MB the toolbar shows a one-line hint suggesting
      `--kv-codec block_tq_b64_b4`. Hint disappears once codec is
      engaged.
- [ ] `/help` lists all commands; each command's behaviour matches
      its docstring.
- [ ] `NO_COLOR=1 silica ...` produces colour-free output.
- [ ] Existing `tests/` suite still 1781 passed (no runtime engine
      regressions); new chat-CLI tests cover palette + toolbar
      formatter + command dispatch + config parser + thinking
      parser.

## 11. Non-goals

- Speed parity with `mlx-lm`'s own `mlx_lm.generate` benchmark.
  silica-chat is allowed a small overhead for the toolbar-render
  and prompt_toolkit event loop. Acceptance: `tok/s` overhead vs
  the equivalent `silica run` single-shot < 5%.
- Windows terminal compatibility. macOS Terminal.app, iTerm2, and
  Linux xterm / Alacritty are the targeted matrix.
- Localisation. English UI; user prompt and assistant output are
  whatever the model produces.
- Custom themes. One default colour scheme; `NO_COLOR` opt-out.

## 12. Open questions

- **Q-CHAT-1**: how to handle generation interruption mid-decode?
  prompt_toolkit's Ctrl-C semantics need to be wired to a cooperative
  cancellation point inside `engine.generate_batch`. Resolved at C-3
  implementation time.
- **Q-CHAT-2**: when `/model <repo>` swaps models, what happens to
  the in-memory conversation history? Default: kept; the new model
  re-tokenises against its own chat template. Edge case: if the new
  model's vocab cannot encode some assistant token from the previous
  history, the swap fails loudly. Resolved at C-2 implementation time.
- **Q-CHAT-3**: prefix_cache invalidation on `/reset`. The cache must
  be dropped or fully invalidated; preserving it across `/reset`
  would leak prior-conversation tokens into the new conversation.
  Resolved at C-4 implementation time.
- **Q-CHAT-4**: `<think>` tag detection robustness. Streamed tokens
  may split the literal `<think>` across multiple chunks (e.g.
  `<th` and `ink>` in separate decode iterations). The C-8 parser
  must accumulate a tag-fragment buffer until a complete tag is
  matched; otherwise transitions are missed. Resolved at C-8
  implementation time. Test coverage in
  `tests/test_chat_cli_thinking_parser.py` includes
  per-character feeds to expose any fragmentation issue.
