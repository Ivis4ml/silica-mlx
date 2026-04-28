# CHAT.README — running `silica chat` on macOS

Setup and usage notes for the chat REPL (`silica chat`). Side track 2
adds a prompt_toolkit-based interactive shell with persistent bottom
toolbar (tok/s, KV memory, compression ratio, prefix-cache hit count
— see `plans/CHAT_CLI_OPENING.md` for the full design).

This README captures the install path that survives macOS Sequoia's
iCloud-Drive interaction with editable Python installs. Read this
once, set it up once, then `silica chat` works for the rest of the
project's lifetime.

## TL;DR — first-time setup

```bash
# Project lives at ~/Desktop/silica-mlx; the venv lives outside
# iCloud-synced storage to avoid the UF_HIDDEN issue (see Background
# below).

cd ~/Desktop/silica-mlx

# 1. Create the venv outside iCloud
uv venv --python 3.13 ~/.cache/uv/silica-mlx-venv

# 2. Activate it
source ~/.cache/uv/silica-mlx-venv/bin/activate

# 3. Install the package + chat extras into the active venv
pip install -e '.[chat]'

# 4. Verify
silica chat --help
```

The activate-then-pip flow leaves nothing in `~/.zshrc`. Once
activated, `pip` and `silica` both resolve to the venv's `bin/`;
`pip install -e '.[chat]'` lands inside the venv with no
environment-variable juggling.

If you prefer one-shot installation without activating first, set
`UV_PROJECT_ENVIRONMENT` inline:

```bash
UV_PROJECT_ENVIRONMENT=$HOME/.cache/uv/silica-mlx-venv \
    uv pip install -e '.[chat]'
```

Both paths end up at the same place.

## Daily usage

Three equivalent ways to launch the REPL — pick the one that fits
your workflow.

### Option 1 — activate the venv (recommended for interactive use)

```bash
source ~/.cache/uv/silica-mlx-venv/bin/activate
silica chat
silica chat --model Qwen/Qwen3.5-4B --kv-codec block_tq_b64_b4
deactivate                                 # when done
```

The activation prepends the venv's `bin/` to `PATH` for the current
shell only. No global `PATH` pollution. `deactivate` reverses it.

Optional one-liner alias for activation:

```bash
echo 'alias silica-env="source ~/.cache/uv/silica-mlx-venv/bin/activate"' \
    >> ~/.zshrc
```

Then `silica-env` activates and `silica chat` is ready.

### Option 2 — `uv run` from the project directory

```bash
cd ~/Desktop/silica-mlx
UV_PROJECT_ENVIRONMENT=$HOME/.cache/uv/silica-mlx-venv uv run silica chat
```

Inline `UV_PROJECT_ENVIRONMENT` tells uv where the venv lives. No
activation needed; no zshrc change. If you find yourself doing this
often, persist the env var in `~/.zshrc` so `uv run silica chat`
works from the project directory without the inline prefix —
trade-off: one line of zshrc pollution for shorter command.

### Option 3 — absolute path (works from anywhere, no setup)

```bash
~/.cache/uv/silica-mlx-venv/bin/silica chat
```

## Inside the REPL

Once the REPL is open you should see:

- A grey "Loading Qwen/Qwen3-0.6B ..." line during model load.
- A cyan greeting: `silica chat — Qwen3-0.6B (fp16). Type /help for
  commands, /exit to quit.`
- A bright-green `You ›` prompt.
- A persistent bottom toolbar with the live engine state.

Slash commands available out of the box:

| Command | Purpose |
| --- | --- |
| `/help` | List every command + the full `/config` schema |
| `/config` | Print current config + schema |
| `/config temperature=0.3` | Adjust sampling mid-session |
| `/config max_tokens=2048` | Adjust per-turn token ceiling |
| `/config thinking=auto\|show\|hidden` | Display side: live render of `<think>` blocks |
| `/config thinking_mode=on\|off` | Model side: thread `enable_thinking` to the chat template |
| `/config thinking_history=strip\|keep` | History side: drop or preserve `<think>` content in the next turn's prompt |
| `/config live_toolbar=on\|off` | Opt in to the per-token Ansi toolbar overlay (default off; `SILICA_LIVE_TOOLBAR=1` env override) |
| `/system "You are concise."` | Set / replace the system prompt |
| `/reset` | Clear conversation log + invalidate prefix cache |
| `/regenerate` | Redo the previous turn with a fresh sample |
| `/continue` | Extend the previous turn when it stopped at `max_tokens` |
| `/save <path>` · `/load <path>` | Persist / restore conversation as JSON |
| `/model <repo> [--keep-history]` | Swap the active model; flag re-tokenises stored text against the new tokeniser instead of resetting history |
| `/expand` | Reprint the previous turn's collapsed `<think>` content |
| `/showcase` | One-paragraph session narrative (turns, prefix reuse, last finish, last-turn reasoning/visible char split, `/continue` calls) |
| `/exit` | Quit the REPL |

The **three-axis thinking model** (`thinking_mode` / `thinking` /
`thinking_history`) keeps the model side, the display side, and the
history side independent — see `plans/CHAT_CLI_RESPONSE_POLICY.md`
Decision E. The recommended chat default after the response-policy
side track is `thinking_mode=on` + `thinking=hidden` +
`thinking_history=strip`, which lets the model reason silently
without re-feeding the reasoning into every subsequent turn's
prompt.

When a turn ends at `finish_reason=max_tokens` the shell prints
`[truncated: /continue]` on a fresh line below the reply (the
marker is rendered via direct `sys.stdout.write` and never reaches
`messages`). Run `/continue` to extend the same assistant turn
in place — no new `(user, assistant)` pair is created. Chained
`/continue` is supported; the assistant message stays raw across
the boundary so `apply_chat_template(continue_final_message=True)`
can rebuild the original generation prompt's `<think>\n` boundary
on Qwen3-family templates.

Sampling knobs (temperature, top_p, top_k, max_tokens) are
intentionally **not** CLI flags — they live behind `/config`
inside the REPL. Launch surface stays minimal: `--model`,
`--system`, `--kv-codec`. See `plans/CHAT_CLI_OPENING.md` §6 for the
rationale.

### Default system prompt

If `--system` is not passed, `silica chat` ships a concise default
that steers the model towards short, direct replies and away from
preamble / self-narration / over-elaboration:

```text
You are a concise assistant. Answer directly: skip preamble, skip
self-narration, do not over-elaborate. Stop when the answer is
complete. Reply in the user's language. For code questions, show
the code first.
```

Override or clear:

| Launch | Effect |
| --- | --- |
| `silica chat --model X` | Default prompt above (silica-mlx pre-set) |
| `silica chat --model X --system "You are a senior reviewer."` | Custom prompt verbatim |
| `silica chat --model X --system ""` | No system prompt at all (vanilla model behaviour) |
| `/system "..."` mid-session | Replace the live system prompt |
| `/system` (no arg) mid-session | Clear the live system prompt |

The default is appropriate for local Apple-Silicon inference of
Qwen3 / Qwen3.5 / Gemma4 class models where decode time and
`max_tokens` budget are usually the user's bottleneck rather than
reply quality. For Qwen3 with `thinking_mode=on` the prompt is
read inside the implicit thinking slot too, so the same
"don't over-elaborate" guidance applies to reasoning the user
never sees.

### Disabling the model's reasoning entirely

The default prompt asks the model to be concise but does NOT
disable Qwen3's reasoning phase. To skip the `<think>` block
entirely (saves tokens, faster TTFT-to-visible-reply on cap-bound
turns):

```text
/config thinking_mode=off
```

This threads `enable_thinking=False` to `apply_chat_template`, so
Qwen3 generates the visible reply directly — no `<think>...</think>`
block in the output. Use `/config thinking_mode=on` to re-enable.

The two related axes are independent (see `plans/CHAT_CLI_RESPONSE_POLICY.md`
Decision E):

- `/config thinking=hidden` — display side; the model still
  reasons, the visible transcript just folds the block into a
  magenta "thinking..." indicator.
- `/config thinking_history=keep` — history side; preserves
  `<think>` content in the next turn's prompt for archival
  workflows. Default `strip` keeps history clean.

## Background — why the venv lives outside iCloud

macOS Sequoia (15.x) tags certain installed files with the
`com.apple.provenance` extended attribute and the BSD `UF_HIDDEN`
flag. When iCloud Drive is syncing the directory containing a Python
venv (e.g. `~/Desktop` under "Desktop and Documents"), every file
in `<venv>/lib/python3.13/site-packages/` ends up flagged.

Python 3.13's `site.py` treats `UF_HIDDEN`-flagged `.pth` files as
hidden and skips them as a security measure (it reports
`Skipping hidden .pth file:` under `python -v`). The editable
install ships its package-finder via a `.pth`:

```text
__editable__.silica_mlx-0.0.1.pth
```

Skipping it means the editable finder is never registered on
`sys.meta_path`, and `import silica` fails:

```text
ModuleNotFoundError: No module named 'silica'
```

Clearing the flag manually (`chflags nohidden`) is not durable —
iCloud Drive re-applies it. The robust fix is to keep the venv
outside iCloud-synced storage, which is what the TL;DR does.

The project source itself (`~/Desktop/silica-mlx/`) can stay on the
synced Desktop without trouble — only the venv's `site-packages` is
sensitive to the flag, because that is where `.pth` files live.

## Troubleshooting

### `command not found: silica`

The venv's `bin/` is not on `PATH`. Activate the venv (Option 1) or
prefix with the absolute path (Option 3).

### `ModuleNotFoundError: No module named 'silica'` from the script entry

The `.pth` file is being skipped — the venv is on iCloud-synced
storage. Verify with:

```bash
ls -lO <your-venv>/lib/python3.13/site-packages/__editable__.silica_mlx-0.0.1.pth
```

If the line shows `hidden` between `staff` and the byte count, the
venv is in iCloud range. Re-create it under `~/.cache/uv/...` per
the TL;DR.

### `silica chat requires prompt_toolkit. Install with: ...`

The `[chat]` extras are not installed in the active venv. Install:

```bash
UV_PROJECT_ENVIRONMENT=$HOME/.cache/uv/silica-mlx-venv \
    uv pip install -e '.[chat]'
```

### Two installs in the wild (`which silica` returns the wrong path)

If both miniconda and the project venv have silica installed, `PATH`
ordering decides which one wins. Either uninstall from miniconda
(`pip uninstall silica-mlx prompt_toolkit pygments` while miniconda
is the active environment), or activate the project venv to put its
`bin/` first on `PATH`.

### `Warning: Input is not a terminal (fd=0).`

prompt_toolkit complains when stdin is piped instead of a real TTY.
Expected behaviour for non-interactive smoke tests; harmless under
normal interactive use.

## Where things live

| Location | Contents |
| --- | --- |
| `~/Desktop/silica-mlx/` | Project source (kept on Desktop; iCloud sync OK here) |
| `~/.cache/uv/silica-mlx-venv/` | Python venv (outside iCloud — required) |
| `~/.cache/uv/silica-mlx-venv/bin/silica` | Console-script entry point |
| `~/.cache/silica/chat_history` | prompt_toolkit history file (auto-created) |
| `~/.claude/projects/-Users-xinyu-Desktop-silica-mlx/memory/` | Claude Code project memory (path-keyed; do not move) |

## Updating / reinstalling

When `pyproject.toml` changes (new optional deps, a `[project.scripts]`
entry, etc.), reinstall. Easiest path is activate-then-pip:

```bash
source ~/.cache/uv/silica-mlx-venv/bin/activate
pip install -e '.[chat]'
```

Or one-shot without activating:

```bash
UV_PROJECT_ENVIRONMENT=$HOME/.cache/uv/silica-mlx-venv \
    uv pip install -e '.[chat]'
```

The editable install means source-code changes are picked up
automatically — you only re-run `pip install -e .` when the package
metadata (entry points, dependencies) changes.

## Tests

Unit tests for the chat-CLI layers (palette, state, toolbar,
commands, config, app helpers, persistence, live-toolbar backends,
swap-model) plus the full `ChatSession` surface
(`chat`, `continue_last`, `pop_last_exchange`, `replace_messages`,
deferred-finalise + continuation-snapshot lifecycle, three-tier
strip fallback) run in any environment that has the project
installed:

```bash
uv run pytest tests/test_chat_cli_*.py tests/test_chat_session.py
```

Helper-level coverage hits the `_evaluate_continue_request`,
`_assistant_ends_in_thinking`, `_print_truncation_marker`,
`_capture_rollback_snapshot` / `_apply_rollback_snapshot`,
`_resolve_thinking_mode` / `_resolve_thinking_history` /
`_resolve_live_toolbar_enabled` extracts so REPL-side regressions
land on a focused test rather than only on manual smoke.

The prompt_toolkit Application's event loop itself is covered by
manual smoke (open the REPL, exercise each slash command, check
colour rendering, verify the toolbar fields update between turns;
see HARDENING-9 / RP-6 manual acceptance templates under `plans/`).
