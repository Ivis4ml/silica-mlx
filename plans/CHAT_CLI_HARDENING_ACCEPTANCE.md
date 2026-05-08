# CHAT-CLI-HARDENING — manual acceptance

| Field | Value |
| --- | --- |
| Status | **draft / pending manual run** |
| Side track | `plans/CHAT_CLI_HARDENING.md` (HARDENING-1 .. HARDENING-9) |
| Sub-unit | HARDENING-9 |
| Author | xxzhou |
| Last updated | 2026-04-28 |
| GA decision | **Pending** — gated on a completed run + sign-off below |

This document is the manual-acceptance record for the
CHAT-CLI-HARDENING side track. HARDENING-1 through HARDENING-8
landed as code + automated tests; HARDENING-9 is the real-model
smoke run that confirms the chat REPL behaves the way the
automated suite says it does. Until §7 is filled in with a real
verdict, treat the side track as **side-track beta**, not GA.

The template uses ``<TODO: ...>`` placeholders for fields that
must be filled in during the real run. Replace them inline; do
not delete the surrounding structure (a future review wants to
see "what was checked", not just "what passed"). When all
placeholders are gone and §7 has a final verdict, the side track
graduates from beta to GA and the trailing "Last updated" date
moves to the run-completion date.

---

## 1. Metadata

| Field | Value |
| --- | --- |
| Run date | `<TODO: YYYY-MM-DD>` |
| Operator | `<TODO: name>` |
| Machine | `<TODO: e.g. M5 Pro 48GB / M2 Max 96GB>` |
| macOS version | `<TODO: e.g. 14.5 (23F79)>` |
| Python | `<TODO: e.g. 3.13.12>` |
| MLX version | `<TODO: e.g. mlx 0.32.1>` |
| mlx-lm version | `<TODO: e.g. 0.31.2>` |
| silica-mlx commit | `<TODO: full SHA at run time>` |
| Models on disk | `<TODO: e.g. Qwen/Qwen3-0.6B (cache hit), Qwen/Qwen3.5-4B (cold load)>` |
| HF cache notes | `<TODO: any download issues, gated repo handling, etc.>` |

---

## 2. Interactive REPL acceptance — Qwen3-0.6B fp16

**Launch:** `python scripts/chat.py --model Qwen/Qwen3-0.6B`

For each command below, record what you **expected** vs what you
**observed**, and whether the row passes. "Pass" means the
command's user-visible behaviour matches the help text and the
HARDENING-1..6 commit messages. Notes column is for anything
surprising worth a follow-up.

| # | Command | Expected | Observed | Pass | Notes |
| --- | --- | --- | --- | --- | --- |
| 2.1 | `Hi, who are you?` | One assistant turn, ``state=prefill→decode`` live in toolbar, post-turn TTFT/peak settle | `<TODO>` | `<TODO>` | `<TODO>` |
| 2.2 | `/system You are a terse Qwen assistant. Reply in <= 2 sentences.` then `What is 2+2?` | System prompt takes effect this turn (HARDENING-1 / F1) | `<TODO>` | `<TODO>` | `<TODO>` |
| 2.3 | `/config thinking_mode=off` then `Plan a 3-step argument.` | Reply has no `<think>` block (HARDENING-2 / F2) | `<TODO>` | `<TODO>` | `<TODO>` |
| 2.4 | `/config thinking_mode=on` then `Plan a 3-step argument.` | `state=thinking` visible in toolbar, then `state=decode`; post-turn `/expand` reveals reasoning | `<TODO>` | `<TODO>` | `<TODO>` |
| 2.5 | `/regenerate` after a normal turn | Last (user, assistant) pair dropped, fresh assistant reply produced (HARDENING-4 / F4) | `<TODO>` | `<TODO>` | `<TODO>` |
| 2.6 | `/regenerate` on a fresh session | Yellow `(/regenerate: no prior turn to redo)` notice; no crash | `<TODO>` | `<TODO>` | `<TODO>` |
| 2.7 | `/regenerate`, then Ctrl-C during the regenerated turn | `[generation aborted]` printed; the **original** prior turn is restored on screen (HARDENING-4 rollback) | `<TODO>` | `<TODO>` | `<TODO>` |
| 2.8 | `/save /tmp/silica-chat-h9.json` | File created, JSON well-formed | `<TODO>` | `<TODO>` | `<TODO>` |
| 2.9 | `/exit`, relaunch, `/load /tmp/silica-chat-h9.json` | Conversation restored; `/expand` works on prior reasoning | `<TODO>` | `<TODO>` | `<TODO>` |
| 2.10 | `/model Qwen/Qwen3.5-4B` (no `--keep-history`) | Notice "history reset"; conversation cleared (HARDENING-5 default) | `<TODO>` | `<TODO>` | `<TODO>` |
| 2.11 | After 2.10: `/model Qwen/Qwen3-0.6B --keep-history` | Notice "history preserved"; messages survive; next turn re-tokenises | `<TODO>` | `<TODO>` | `<TODO>` |
| 2.12 | `/help` | `/model` row advertises `[--keep-history]` (HARDENING-5 lock) | `<TODO>` | `<TODO>` | `<TODO>` |
| 2.13 | `/showcase` | Multi-line narrative; `prefix reused` non-zero after at least one repeat-context turn | `<TODO>` | `<TODO>` | `<TODO>` |

---

## 3. Live toolbar acceptance — HARDENING-6 / F3

**Backend resolution:** confirm in §1 that `TERM` and the
detected palette correspond to the AnsiLiveToolbar path; if the
test machine is non-TTY or `TERM=dumb`, the harness short-
circuits to `NullLiveToolbar` and the live-update assertions in
this section do not apply (record that explicitly).

| # | Observation | Expected | Observed | Pass | Notes |
| --- | --- | --- | --- | --- | --- |
| 3.1 | During the 1-3 second prefill on a 4B+ model | Toolbar shows `state=prefill`, `tokens=0/<max>`, `tok/s=—` (no blank line below `silica ›`) | `<TODO>` | `<TODO>` | `<TODO>` |
| 3.2 | During reply streaming | `tokens=N/<max>` and `tok/s=<value>` update visibly per token; `state=decode` (or `thinking` during a `<think>` block) | `<TODO>` | `<TODO>` | `<TODO>` |
| 3.3 | Mid-generation Ctrl-C | `[generation aborted]` lands on a clean line — toolbar text is not overlaid by the marker; subsequent prompt is on a fresh line | `<TODO>` | `<TODO>` | `<TODO>` |
| 3.4 | Triggered error path (e.g. ill-typed `/config max_tokens=foo` then a turn) | If the turn raises, `[error: ...]` lands on a clean line; no terminal corruption after returning to prompt | `<TODO>` | `<TODO>` | `<TODO>` |
| 3.5 | After `/exit` | Cursor is on a fresh line; no leftover toolbar text; shell prompt unaffected | `<TODO>` | `<TODO>` | `<TODO>` |
| 3.6 | Pipe stdout to a file: `python scripts/chat.py --model Qwen/Qwen3-0.6B < /tmp/in > /tmp/out` | File contains streamed text only — no ANSI escape sequences (NullLiveToolbar selected on non-TTY) | `<TODO>` | `<TODO>` | `<TODO>` |
| 3.7 | After several turns with thinking blocks and code fences | Toolbar text never appears mid-transcript above the latest output; assistant prefix and reply are not eaten by toolbar overlap. Locks the regression mode fixed at commit `eb337d8` (clear toolbar before any cursor-moving streamed text). | `<TODO>` | `<TODO>` | `<TODO>` |

Decision-D reminder: this section verifies the ANSI sticky-bottom-line
backend, not a full prompt-toolkit `Application`. The latter is
deferred per `plans/CHAT_CLI_HARDENING.md` §6 Decision D.

---

## 4. Chat-bench harness runs — HARDENING-7

Two real-model invocations of `scripts/chat_bench.py`, default
3 turns each. Capture both the text report and the exit code; a
failed Q-012 verdict here is a regression of the v1.7.15 P5.9
step 2(b) cross-call prefix-reuse claim and must be triaged
before declaring GA.

### 4.1 Qwen3-0.6B fp16

```
$ python scripts/chat_bench.py --model Qwen/Qwen3-0.6B
<TODO: paste full text report — header line, per-turn rows, Q-012 verdict>
$ echo $?
<TODO: 0 (passed) or 1 (failed)>
```

| Field | Value |
| --- | --- |
| Q-012 verdict | `<TODO: passed / failed / n/a>` |
| Turn 1 TTFT | `<TODO>` |
| Turn 2 TTFT | `<TODO>` |
| Turn 2 prefix_hit_blocks | `<TODO>` |
| Turn 3 prefix_hit_blocks | `<TODO>` |
| Notes | `<TODO: any anomalies, e.g. unexpectedly slow turn 2>` |

### 4.2 Qwen3.5-4B + BlockTQ

```
$ python scripts/chat_bench.py --model Qwen/Qwen3.5-4B --kv-codec block_tq_b64_b4
<TODO: paste full text report>
$ echo $?
<TODO>
```

| Field | Value |
| --- | --- |
| Q-012 verdict | `<TODO: passed / failed / n/a>` |
| Turn 1 TTFT | `<TODO>` |
| Turn 2 TTFT (cache-hit) | `<TODO>` |
| Compression ratio observed | `<TODO: e.g. 3.8x via stats endpoint>` |
| Prefix-store residency, end of run | `<TODO: MB>` |
| Notes | `<TODO>` |

### 4.3 Optional sanity: longer run

```
$ python scripts/chat_bench.py --model Qwen/Qwen3-0.6B --turns 5 --max-tokens 128 --json
<TODO: paste JSON summary OR record path of saved output>
```

| Field | Value |
| --- | --- |
| Q-012 verdict | `<TODO>` |
| Notes | `<TODO: optional; skip if 4.1 / 4.2 already covered the surface>` |

---

## 5. F1-F6 sign-off

Each finding from `plans/CHAT_CLI_HARDENING.md` §2 needs a
"yes I observed it works" entry, anchored to the relevant §2 /
§3 / §4 evidence above. "Status" should read `closed` only
when both the relevant rows above pass AND the named commit
matches.

| Finding | Symptom (pre-fix) | Commit | Evidence | Status |
| --- | --- | --- | --- | --- |
| F1 | `/system` writes config but live session keeps old prompt | `cec2c7f` | §2 row 2.2 | `<TODO: closed / failed>` |
| F2 | `enable_thinking` not threaded → reasoning leaks when `thinking_mode=off` | `9601f64` | §2 rows 2.3–2.4 | `<TODO>` |
| F3 | Toolbar refreshes only at prompt phase; tok/s / tokens / state are post-turn snapshots | `b79ee58` (+ hotfix `eb337d8`) | §3 rows 3.1–3.7 | `<TODO>` |
| F4 | `/regenerate` printed "(not wired yet)" | `52fc44d` | §2 rows 2.5–2.7 | `<TODO>` |
| F5 | `/model` silently drops history despite plan saying default keeps | `410db40` | §2 rows 2.10–2.12 | `<TODO>` |
| F6 | ChatSession reads `_store` / `_detached` / `_k_codec` private fields | `3241ab7` | Indirect — automated tests cover the public stats API; live REPL exercises it via `/showcase` (§2 row 2.13) | `<TODO>` |

---

## 6. Known gaps and deferrals

Things this acceptance run intentionally does NOT verify, with
the rationale for each. Filling these in helps a future review
distinguish "we forgot" from "we decided not to".

- **Full prompt-toolkit `Application` live toolbar.** Path-A
  rewrite is deferred per `plans/CHAT_CLI_HARDENING.md` §6
  Decision D. The current acceptance verifies the ANSI
  sticky-bottom-line backend (§3) and accepts terminal-fragility
  edge cases (e.g. weird-terminal scrollback, multi-line reply
  overflowing the screen) as out-of-scope for this side track.
- **Real terminal rendering on every emulator.** Manual smoke is
  scoped to one terminal per machine listed in §1. Other
  emulators (gnome-terminal, Windows Terminal under PowerShell,
  ssh-tmux nested sessions) are not tested.
- **App-loop integration tests.** HARDENING-8 covers the four
  named helpers; the full `run_chat` REPL loop (token streaming,
  parser drain, with-block exit) is exercised only via this
  manual run, not via automated tests. A future side track could
  add a prompt-toolkit-driver-based integration suite if drift
  warrants it.
- **Model-specific issues.** Anything model-specific encountered
  during the run that is *not* a chat-CLI bug (tokeniser quirks,
  MLX kernel issues, weight loading) gets recorded here and
  redirected to the appropriate side track.

```
<TODO: fill in any model-specific issues encountered during the
acceptance run; remove this placeholder if none>
```

---

## 7. GA decision

| Field | Value |
| --- | --- |
| Verdict | `<TODO: PASS / FAIL>` |
| Date | `<TODO: YYYY-MM-DD — same as §1 run date>` |
| Operator | `<TODO>` |
| F1-F6 all closed in §5 | `<TODO: yes / no>` |
| §3 live-toolbar passes | `<TODO: yes / no / N-A non-TTY>` |
| §4 chat-bench Q-012 verdicts | `<TODO: 4.1=… 4.2=…>` |
| Outstanding blockers | `<TODO: list, or "none">` |

**Verdict prose** (one paragraph, fill after the run):

```
<TODO: Replace this with the operator's plain-language summary —
what passed, what surprised, whether the chat CLI is GA-ready or
needs another round. Pre-fill nothing here; the prose is the
record.>
```

When this verdict reads `PASS` and §5 shows `closed` in every
row, the side track graduates from beta to GA and the chat REPL
is the supported entry point for `python scripts/chat.py` and
`silica chat`. Until then the side track stays in beta and any
remaining placeholders are open work items.

---

## 8. Cross-references

- `plans/CHAT_CLI_HARDENING.md` — side-track opening doc + Decision D.
- `plans/CHAT_CLI_OPENING.md` — original C-1..C-8 design.
- HARDENING-1..8 commits: `cec2c7f`, `9601f64`, `3241ab7`,
  `52fc44d`, `410db40`, `b79ee58`, `df3bd24`, `9a9d5d3`.
- HARDENING-6 hotfix: `eb337d8` (live toolbar overlap with
  generation output; locked by §3 row 3.7).
- `silica/bench/chat_bench.py` + `scripts/chat_bench.py` — the
  non-interactive harness §4 invokes.
- `silica/chat/cli/live_toolbar.py` — the §3 backend under test.
