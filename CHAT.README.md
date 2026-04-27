# CHAT.README — running `silica chat` on macOS

Setup and usage notes for the chat REPL (`silica chat`). Side track 2
adds a prompt_toolkit-based interactive shell with persistent bottom
toolbar (tok/s, KV memory, compression ratio, prefix-cache hit count
— see `docs/CHAT_CLI_OPENING.md` for the full design).

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

# 2. Install the package + chat extras into that venv
UV_PROJECT_ENVIRONMENT=$HOME/.cache/uv/silica-mlx-venv \
    uv pip install -e '.[chat]'

# 3. Persist the env-var so `uv run` finds the venv from anywhere
echo 'export UV_PROJECT_ENVIRONMENT=$HOME/.cache/uv/silica-mlx-venv' \
    >> ~/.zshrc
source ~/.zshrc

# 4. Verify
~/.cache/uv/silica-mlx-venv/bin/silica chat --help
```

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
uv run silica chat
uv run silica chat --kv-codec block_tq_b64_b4
```

`UV_PROJECT_ENVIRONMENT` (set in `.zshrc` per step 3 above) tells uv
where the venv lives. No activation needed. Best fit for one-off
invocations from inside the project.

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
| `/config thinking=hidden` | Hide `<think>` blocks (lands fully at C-8) |
| `/system "You are concise."` | Set / replace the system prompt |
| `/reset` | Clear conversation log + invalidate prefix cache |
| `/regenerate` | Redo the previous turn (lands at C-4) |
| `/save <path>` / `/load <path>` | Persist / restore (lands at C-7) |
| `/model <repo>` | Swap the active model (lands at C-7) |
| `/exit` | Quit the REPL |

Sampling knobs (temperature, top_p, top_k, max_tokens) are
intentionally **not** CLI flags — they live behind `/config`
inside the REPL. Launch surface stays minimal: `--model`,
`--system`, `--kv-codec`. See `docs/CHAT_CLI_OPENING.md` §6 for the
rationale.

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
entry, etc.), reinstall:

```bash
UV_PROJECT_ENVIRONMENT=$HOME/.cache/uv/silica-mlx-venv \
    uv pip install -e '.[chat]'
```

The editable install means source-code changes are picked up
automatically — you only re-run `uv pip install -e .` when the
package metadata (entry points, dependencies) changes.

## Tests

Unit tests for the chat-CLI layers (palette, state, toolbar,
commands, config) run in any environment that has the project
installed:

```bash
uv run pytest tests/test_chat_cli_*.py
```

The prompt_toolkit Application (C-3) itself is not unit-tested —
the event loop is covered by manual smoke (open the REPL, exercise
each slash command, check colour rendering, verify the toolbar
fields update between turns).
