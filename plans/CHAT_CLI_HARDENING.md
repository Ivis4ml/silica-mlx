# Chat CLI Hardening — side track opening

| Field | Value |
| --- | --- |
| Phase | side track (not a numbered Phase; not P-6 / P-7 / P-8) |
| Status | drafted; pending sub-unit landings |
| Last updated | 2026-04-27 |
| Trigger | GPT-5.5 verification round against `silica.chat` (post v1.7.16) |
| Scope owner | Xin Zhou |
| Predecessor | `plans/CHAT_CLI_OPENING.md` (the original C-1..C-8 design doc) |
| Successor | none — feeds into P-8 (mini-sglang HTTP server) when that phase opens |

This document anchors a small hardening phase for `silica.chat`
that lifts the CLI from "architecture skeleton + Python-layer
testing solid" to "user-visible product-grade acceptance". It
exists alongside `plans/CHAT_CLI_OPENING.md`, which captured the
C-1..C-8 build-out design; this opening captures the gaps
between that design's intent and the actual code at
`cd0332b`.

The hardening is **not** a P-6 phase concern — P-6 is about
performance on dense / MoE big models. Chat CLI hardening is
about the chat REPL's correctness and live-metric story, and
fits the side-track convention (`CHAT_CLI_OPENING.md` was the
first such doc).

---

## 1. What's already solid

- `silica.chat.session.ChatSession` (663 lines) — apply_chat_template
  + prefix reuse + per-turn metrics + persistence schema.
- `silica.chat.cli.app` (1002 lines) — prompt-toolkit REPL, slash
  commands, persistence, palette, code-fence highlighting,
  thinking-block parser.
- 250 chat / CLI tests pass.
- ruff + mypy clean on `silica.chat` + `silica.server.cli` +
  `scripts/chat.py`.
- Q-012 cross-call prefix reuse landed at v1.7.15 (P5.9 step 2(b)),
  so the chat REPL already gets cross-turn prefix hits without
  caller workarounds.

---

## 2. What's still gappy — the six GPT-5.5 findings

Verified against the `cd0332b` worktree. Each row names the gap +
the file:line it lives at + the user-visible symptom:

| # | Severity | File:line | Gap | Symptom |
| --- | --- | --- | --- | --- |
| F1 | High | `silica/chat/cli/commands.py:172` | `/system` writes `state.config["system_prompt"]` but never updates the live `ChatSession`. | Slash-command help promises "for the rest of the session"; reality is the in-flight session uses the prompt it was constructed with. |
| F2 | High | `silica/chat/session.py:572-609` | `_render_prompt` calls `apply_chat_template(...)` without `enable_thinking`. | `/config thinking_mode=off` only flips the parser-side fold; the model can still emit reasoning tokens. The parser then doesn't fold them — they leak into the visible answer. |
| F3 | High | `silica/chat/cli/app.py:329` | Bottom toolbar refreshes only during prompt phase. Generation uses inline indicator. | `tok/s` / `tokens=N/max` / `state=thinking|decode` are post-turn snapshots, not live. |
| F4 | Medium | `silica/chat/cli/app.py:379-389` | `/regenerate` registered in command surface, prints "(not wired yet)" when invoked. | Help advertises a feature that doesn't exist. |
| F5 | Medium | `silica/chat/cli/app.py:846-850` | `_swap_model` resets conversation; plan said default keeps history. | `/model new-repo` silently drops the chat. The session stores text messages, so re-tokenising under the new tokeniser is straightforward. |
| F6 | Medium | `silica/chat/session.py:394-410` | ChatSession reads `_store` / `_detached` / `_k_codec` private fields off `RadixPrefixCache`. | Toolbar metric numbers will drift if the prefix-cache internals refactor. |

---

## 3. Sub-unit decomposition

Each sub-unit is bounded enough to land + verify + pause inside one
incremental commit. Numbered for the commit log; each has a single
goal stated so a failed verification rolls back a single change
rather than a multi-feature commit.

| ID | Goal | Files touched | Tests added |
| --- | --- | --- | --- |
| **HARDENING-1** | Wire `/system` to the live `ChatSession` (fix F1). New `ChatSession.set_system_prompt(text)`; `/system` calls it; prefix cache stays valid (system prompt sits at index 0 in messages, so existing-cached prefixes for the *previous* system prompt invalidate, which the prefix cache handles natively via radix-tree mismatch). | `silica/chat/session.py`, `silica/chat/cli/app.py`, `silica/chat/cli/commands.py` | `tests/test_chat_session.py` (new test); `tests/test_chat_cli_commands.py` (wire-up assert). |
| **HARDENING-2** | Thread `enable_thinking` from `state.config["thinking_mode"]` through to `apply_chat_template` (fix F2). `ChatSession.set_thinking_mode(bool)`; `_render_prompt` reads it + passes `enable_thinking` to the template. Parser side (`start_in_thinking`) stays as-is. | `silica/chat/session.py`, `silica/chat/cli/app.py`, `silica/chat/cli/commands.py` (only if `/config thinking_mode=…` propagation needs tweaking) | `tests/test_chat_session.py` (apply_chat_template kwarg pinned via fake tokenizer that records calls). |
| **HARDENING-3** | Public stats API on `RadixPrefixCache` + store (fix F6). Add `RadixPrefixCache.stats() -> PrefixCacheStats` (resident_bytes, logical_bytes, num_blocks, num_codec_codes if applicable). `ChatSession` switches off the `_store` / `_detached` / `_k_codec` reads. | `silica/kvcache/prefix.py`, `silica/kvcache/store.py`, `silica/chat/session.py` | `tests/test_prefix_cache.py` (or new `tests/test_prefix_cache_stats.py`); `tests/test_chat_session.py` (toolbar metrics path no longer touches private fields). |
| **HARDENING-4** | Implement `/regenerate` (fix F4). Pop the last assistant turn from `ChatSession`, re-issue `chat()` with the same user prompt + sampling params. Prefix cache wins through Q-012 affirmative resolution — second turn's prefix is already stored. | `silica/chat/session.py` (drop_last_assistant_turn or similar), `silica/chat/cli/app.py` | `tests/test_chat_session.py` (drop-last invariant); `tests/test_chat_cli_commands.py` (request_regenerate flow). |
| **HARDENING-5** | `/model --keep-history` honours plan (fix F5). Default behaviour stays "drop history" but adds an explicit flag (or default flips, with explicit `--reset` to drop). Either way, the plan / impl mismatch resolves. New tokeniser re-tokenises stored text messages; prefix cache invalidates because the new model has a different cache namespace. | `silica/chat/cli/app.py`, possibly `silica/chat/cli/commands.py` | `tests/test_chat_cli_commands.py` (model-swap with kept history). |
| **HARDENING-6** | Live bottom toolbar backend (fix F3). Pluggable backend so the chat-turn flow can request live refreshes per token without committing to one rendering strategy. Ship the ANSI sticky-bottom-line backend now (cursor save / restore around the streamed text); leave the prompt-toolkit `Application` backend optional and deferred — see Decision D below. The chat-turn flow updates `tokens=N/max`, live `tok/s` (rolling window), and `state=prefill|thinking|decode` per token; post-turn fields (TTFT, peak, prefix-hit, compr) settle once. Null backend on non-TTY / `TERM=dumb` / plain palette so file-redirected output stays clean. | `silica/chat/cli/live_toolbar.py` (new), `silica/chat/cli/app.py` (generation phase wires `with live_toolbar:` and a `RollingTokRate`) | `tests/test_chat_cli_live_toolbar.py` (StringIO sequence checks; fake-clock rolling tok/s; backend-selection table; abort-path `__exit__` lock). |
| **HARDENING-7** | Non-interactive metric harness (`scripts/chat_bench.py`?). Three-turn shared-prompt run; reports first-turn TTFT, second-turn TTFT (prefix-hit), third-turn TTFT, codec compr if installed. Locks the Q-012 cross-turn prefix-reuse claim end-to-end on real prompts (not just the unit-test fake cohort). | new `scripts/chat_bench.py`, possibly extends `silica.bench` | `tests/test_chat_bench.py`. |
| **HARDENING-8** | App-layer unit tests for the helper functions GPT named: `_build_prefix_cache`, `_sampling_params_from_state`, system-prompt propagation flow, thinking-template kwargs flow. | `tests/test_chat_cli_app.py` (new). | New tests only. |
| **HARDENING-9** | Manual real-model acceptance pass (Qwen3-0.6B fp16; Qwen3.5-4B + BlockTQ; three-turn long-context prefix hit). | none (acceptance run only); produces `plans/CHAT_CLI_HARDENING_ACCEPTANCE.md`. | none. |

---

## 4. Sequencing

Sequential. F1-F4 are user-visible correctness; landing them
first restores trust between command help and reality. F5 / F6
are smaller correctness / refactor; F6 (public stats API) is
listed third because F4 / F7 / F8 will all touch the metric
path, and a clean public API makes those three cheaper.
F3 (live toolbar) is heaviest; deferred to last so it lands
on top of an already-honest command surface.

```text
HARDENING-1 (/system)
        │
        ▼
HARDENING-2 (thinking_mode)
        │
        ▼
HARDENING-3 (public stats API)   ← load-bearing for downstream
        │
        ▼
HARDENING-4 (/regenerate)
        │
        ▼
HARDENING-5 (/model keep-history)
        │
        ▼
HARDENING-6 (live toolbar)
        │
        ▼
HARDENING-7 (metric harness)
        │
        ▼
HARDENING-8 (app-layer unit tests)
        │
        ▼
HARDENING-9 (manual acceptance)
```

Each unit is independently committable; the chain doesn't lock
the whole side track on one hard-to-finish item.

---

## 5. Acceptance for the side track as a whole

This side track exits when:

- All six GPT-5.5 findings are closed (F1..F6).
- Toolbar updates live across `tokens=N/max`, `state=...`, rolling
  `tok/s` (HARDENING-6).
- A non-interactive harness reproduces the v1.7.15 Q-012 cross-turn
  prefix-hit claim on real prompts (HARDENING-7).
- App-layer unit tests cover the four helper functions GPT named
  (HARDENING-8).
- One manual real-model run on Qwen3-0.6B fp16 + Qwen3.5-4B
  BlockTQ produces a recorded session showing the "command says
  it / does it" contract holds and the toolbar tracks generation
  live (HARDENING-9).

When that lands, the chat CLI graduates from "side-track beta" to
"side-track GA". P-8 (mini-sglang HTTP server) builds on top of
the same `ChatSession` + `RadixPrefixCache` surface and gets the
hardening for free.

---

## 6. Decisions

### Decision D — full prompt-toolkit `Application` backend deferred (HARDENING-6)

**Date:** 2026-04-27.
**Author:** Xin Zhou.

The original sub-unit table for HARDENING-6 called for a "real
prompt-toolkit `Application` layout that updates per-token". On
review, that path materially exceeds what F3 needs: it requires a
worker-thread engine driver, queue-based token transport, an
`Application` event loop running concurrently with the synchronous
chat-turn flow, full-screen layout with a scrollable text Window,
key bindings, and TTY teardown discipline. The risk surface
(KeyboardInterrupt propagation, MLX threading semantics, terminal
state on crash) is concentrated on exactly the paths that the
pure-Python test infrastructure cannot exercise.

F3's user-visible symptom is narrow: `tokens=N/max`, `tok/s`, and
`state=prefill|thinking|decode` should update per token during
generation. A swappable-backend approach with an ANSI
sticky-bottom-line implementation closes that symptom without
introducing a parallel UI runtime. The chat-turn flow stays
synchronous; the live update is one cursor-save / clear / write /
cursor-restore sequence per token; non-TTY environments fall
through to a `NullLiveToolbar` that the post-turn
`PromptSession.bottom_toolbar` already covers.

The full `Application` path remains addressable later. The
backend interface (`LiveToolbar` ABC) is the seam; an
`ApplicationLiveToolbar` could replace `AnsiLiveToolbar` at the
construction site without changing the chat-turn flow. That work
is appropriate when (and only when) the chat CLI grows additional
TUI affordances that genuinely require an event loop —
scrollback panes, click-to-pause-resume, live config panel,
multi-conversation tabs. None of those are F1-F6 concerns.

This decision narrows HARDENING-6's scope from "Application
backend" to "live toolbar backend, ANSI now, Application
optional later". The acceptance bar in §5 ("Toolbar updates live
across `tokens=N/max`, `state=...`, rolling `tok/s`") is
unchanged.

---

## 7. Cross-references

- `plans/CHAT_CLI_OPENING.md` — original C-1..C-8 design doc.
- PLAN.md §10 Q-012 — initial-cohort prefix-cache consultation
  (resolved at v1.7.15 P5.9 step 2(b)). Q-012's resolution is
  what makes HARDENING-4 `/regenerate` and HARDENING-7's
  three-turn prefix-hit claim work without scheduler-side
  workarounds.
- PLAN.md §7 P-8 — mini-sglang HTTP server. The HARDENING side
  track lands ahead of P-8 because P-8's session manager wraps
  `ChatSession`; cleaner contract here means cleaner P-8 there.
- `plans/P6_REVIEW_HANDOFF.md` — for the broader review-driven
  hardening pattern (rounds 3 and 4 absorbed via stale-text
  cleanup). The CHAT-CLI-HARDENING side track applies the same
  pattern to the chat REPL.
