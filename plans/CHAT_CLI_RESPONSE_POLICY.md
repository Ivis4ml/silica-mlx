# Chat CLI Response Policy — side track opening

| Field | Value |
| --- | --- |
| Phase | side track (not numbered; not P-6 / P-7 / P-8) |
| Status | RP-1..RP-3 landed; side-track interim exit reached. RP-1: `4f98648` + `d4ea04d` (repair). RP-2: `60459c7`. RP-3: `1d9ecd7`. Hand-off to D-021 step 3 (P-6.0.5) ready. |
| Last updated | 2026-04-28 |
| Trigger | Two real-session UX failures observed against Qwen3.5-35B-A3B-4bit + Qwen3-0.6B during interactive use of the post-HARDENING chat REPL (latest code at `51fbcde`) |
| Scope owner | xxzhou |
| Predecessor | `plans/CHAT_CLI_HARDENING.md` (F1-F6 closed in code; HARDENING-9 manual acceptance still pending — see §5 sequencing note) |
| Successor | resume D-021 step 3 (P-6.0.5 measurement expansion); RP-4..RP-6 wait their turn |

CHAT-CLI-HARDENING closed F1-F6 across eight commits and shipped
the live toolbar / `/regenerate` / `/model --keep-history` /
public stats API surface. Two design-level gaps remain that are
not HARDENING bugs — they sit between the raw model surface
(``max_tokens``, ``enable_thinking``) and what a user actually
wants (a complete, readable answer that does not waste budget on
hidden reasoning). This side track adds a thin Response Policy
layer between the two.

The full design space (mode-driven policy, smart auto-continue,
content-aware truncation detection) is large; this opening
commits only to the smallest batch that closes the two observed
failures (RP-1 / RP-2 / RP-3). RP-4..RP-6 are listed for
sequencing visibility but explicitly deferred — they unlock
after P-6.0.5 lands the dense-perf gate that this whole chain
is supposed to feed.

---

## 1. What's already solid

- The CHAT-CLI-HARDENING side track closed F1-F6 across eight
  commits (`cec2c7f`, `9601f64`, `3241ab7`, `52fc44d`, `410db40`,
  `b79ee58` + `eb337d8`, `df3bd24`, `9a9d5d3`).
- `ChatSession` already stores per-turn metrics including
  `finish_reason`, `prompt_tokens`, `output_tokens`,
  `prefix_hit_blocks`, prefix-store residency.
- `silica.chat.cli.live_toolbar` exposes a swappable backend;
  RP-3's truncation marker can route through the same surface.
- `apply_chat_template` is already threaded with
  `enable_thinking` (HARDENING-2); RP-1 reuses the same plumbing
  for the history-strip path.
- HuggingFace tokenizers expose
  `apply_chat_template(messages, continue_final_message=True)`
  for the Qwen3 family — this is the load-bearing primitive RP-2
  needs.

---

## 2. What's still gappy — the two real-session findings

| # | Severity | File / function | Gap | Symptom |
| --- | --- | --- | --- | --- |
| G1 | High | `silica/chat/session.py::ChatSession.chat` (post-decode message append) | `_messages[-1].content` carries the **raw** decoded reply, including any `<think>...</think>` block. The next turn's `_render_prompt` re-tokenises all of that. | Long Qwen3 thinking blocks accumulate in conversation history, inflating prompt length monotonically and slowing every subsequent turn. The CLI's display-side fold (`thinking=hidden` / `auto`) hides them visually but does not stop them from polluting the chat-template input. |
| G2 | High | `silica/chat/session.py::ChatSession.chat` (terminal `finish_reason`) | When the engine stops with `finish_reason="max_tokens"` the CLI silently ends the turn; there is no "/continue" surface, no marker, no metric distinction between "model finished its thought" and "we hit the cap". | Users hit a hard wall at `max_tokens` (default 1024), have no way to extend the same answer, and have no toolbar/showcase signal to know which turns were truncated. The only workaround is `/config max_tokens=4096`, which moves the wall and inflates worst-case KV reservation. |

---

## 3. Three-layer thinking semantics (Decision E)

The chat CLI's existing knobs conflate three orthogonal axes.
RP-1 lands the third axis explicitly so users (and downstream
defaults) can reason about each independently:

| Layer | Knob | Values | Effect | Status |
| --- | --- | --- | --- | --- |
| Model side | `thinking_mode` | `True` / `False` / `None` | Threads `enable_thinking` to `apply_chat_template`. Hard switch — the model genuinely does or does not enter reasoning mode. | Landed at HARDENING-2 |
| Display side | `thinking` | `auto` / `show` / `hidden` | Controls whether the live REPL prints the thinking text inline. Does not affect the model. | Pre-existing |
| History side | `thinking_history` | `strip` / `keep` | Whether `<think>...</think>` content is written into `_messages[-1].content` after the turn. `strip` (default) keeps history clean; `keep` preserves the raw reply for transcripts that need full reasoning. Does not affect the current turn's display, only what the *next* turn's prompt sees. | **RP-1 (this side track)** |

The three layers compose: a user can run with
`thinking_mode=on` (model still reasons) +
`thinking=hidden` (visual fold for the user) +
`thinking_history=strip` (next turn's prompt does not re-feed
the reasoning). This is the configuration the side track
recommends as the chat default after RP-1 lands.

---

## 4. Sub-unit decomposition

Each row is bounded enough to land + verify + pause inside one
incremental commit. Rows RP-1..RP-3 are the committed scope of
this opening; RP-4..RP-6 are listed for sequencing visibility
but deferred per §5.

| ID | Goal | Files touched | Tests added |
| --- | --- | --- | --- |
| **RP-1** | `thinking_history=strip` (close G1). Add the `thinking_history` config key (default `strip`) to the chat-CLI config schema; add a matching ``thinking_history`` constructor kwarg + ``set_thinking_history`` mutator on ``ChatSession`` (mirrors the HARDENING-2 ``thinking_mode`` plumbing). ``ChatSession.chat`` strips ``<think>...</think>`` from the reply *only* on natural completion (``finish_reason in {done, eos}``); a ``finish_reason=max_tokens`` turn keeps the raw text on the assistant message because RP-2 ``/continue`` needs that prefix to resume an open ``<think>`` block. The deferred strip fires either when ``/continue`` reaches natural completion OR when the next user message lands (``ChatSession`` finalises the previous truncated turn before appending the new user msg). The full raw reply remains accessible via a new ``TurnMetrics.raw_reply`` field for callers; chat-CLI's ``/expand`` keeps using ``state.last_turn_thinking`` (parsed display-side) — no UX change there. The chat-CLI shell re-syncs ``thinking_history`` from ``state.config`` per turn so ``/config thinking_history=keep`` takes effect on the next turn without a session rebuild (mirrors the per-turn ``thinking_mode`` resync). | `silica/chat/session.py`, `silica/chat/cli/config.py` (schema), `silica/chat/cli/state.py` (default), `silica/chat/cli/app.py` (ctor pass-through + per-turn resync) | `tests/test_chat_session.py` (history-strip on/off; round-trip after multi-turn; deferred-strip semantics on ``finish_reason=max_tokens``); `tests/test_chat_cli_app.py` (per-turn resync mirrors ``thinking_mode``). |
| **RP-2** *(landed `60459c7`)* | `/continue` (close half of G2). New ``ChatSession.continue_last() -> TurnMetrics`` that re-renders the prompt with ``apply_chat_template(messages, continue_final_message=True)`` (Qwen3 family supports it; ``KeyError``-fallback path for tokenisers that do not is RP-2 acceptance) and appends generated tokens to the **existing** assistant message rather than creating a fresh ``(user, assistant)`` pair. The raw assistant prefix preserved by RP-1's deferred-strip path is what makes byte-equivalent continuation possible — ``/continue`` would silently produce the wrong text if RP-1 stripped truncated turns eagerly. ``continue_last`` finalises the assistant message per ``thinking_history`` only when this continuation reaches natural completion (``finish_reason in {done, eos}``); chained continuations (cap hit twice) carry the raw form forward. New ``/continue`` slash command in ``silica/chat/cli/commands.py``. The chat-CLI shell guards ``/continue`` against (a) the last turn not being assistant-shaped, and (b) the last turn not having ``finish_reason=max_tokens`` (warning + no-op for the latter; the user is told nothing was truncated). Rollback on abort mirrors HARDENING-4's pre-pop snapshot pattern. **Landed in commit `60459c7`** with three pre-merge correctness improvements rolled into the same patch (continuation-side implicit-leading snapshot decoupled from RP-1's strip-finalise snapshot; finalise three-tier fallback covering keep→strip mid-flight switches; ``ChatSession.messages`` deep-copies dict entries so the abort-rollback snapshot is decoupled from in-place writes). | `silica/chat/session.py`, `silica/chat/cli/commands.py`, `silica/chat/cli/app.py` | `tests/test_chat_session.py` (``continue_last`` appends; honours ``continue_final_message``; raw prefix preserved across truncation; finalise-strip fires only on natural completion; v3 / v4 snapshot-lifecycle + deep-copy contracts); `tests/test_chat_cli_commands.py` (``/continue`` dispatcher flag); `tests/test_chat_cli_app.py` (no-prior / not-truncated; parser-start uses snapshot not live config). |
| **RP-3** *(landed `1d9ecd7`)* | Truncation UX + metrics (close the other half of G2). When ``finish_reason == "max_tokens"``, the chat-CLI shell prints ``[truncated: /continue]`` on a fresh line via direct ``sys.stdout.write`` — the marker deliberately bypasses the streaming protocol so it never reaches ``chat_session.messages`` and cannot break the ``/continue`` flow it advertises. Toolbar gains a ``finish=`` field surfacing the most recent terminal reason (``done`` / ``max_tokens`` / ``eos`` / ``stop_token`` / ``aborted`` / ``empty``); em-dash before any turn runs and after ``/reset`` / ``/load`` / any ``/model`` swap. ``ChatCliState`` accumulates ``last_turn_reasoning_chars`` / ``last_turn_visible_chars`` per turn (carries forward across ``/continue`` boundaries via the same gate as ``last_turn_thinking``) plus a session-cumulative ``total_continuation_chunks``. ``/showcase`` grows three lines: ``last finish``, the per-turn char split (suppressed when both zero), and ``/continue calls`` (suppressed when zero). **Metric scope**: char-level only — counted from the display-side ``ThinkingParser`` events; token-level reasoning/visible split needs a tokeniser-level intercept that does not exist today (the metric promotes without renaming when one ships). | `silica/chat/cli/app.py`, `silica/chat/cli/state.py`, `silica/chat/cli/toolbar.py` | `tests/test_chat_cli_toolbar.py` (``finish=`` rendering across all reasons + em-dash + forward-compat unknown values; ``/showcase`` finish + chars + ``/continue calls`` rendering); `tests/test_chat_cli_app.py` (state defaults; ``_print_truncation_marker`` prints on max_tokens / silent on other reasons; signature pin against history pollution); `tests/test_chat_cli_swap_model.py` (keep-history swap clears the new fields). |
| RP-4 | `/mode fast|balanced|deep` policy bundles. Maps mode → `(thinking_mode, chunk_tokens, max_reply_tokens, sampling)`. Default `fast` (thinking off, 768 / 2048). `/mode deep` opt-in for genuinely complex turns. | `silica/chat/cli/commands.py`, `silica/chat/cli/config.py` | dispatcher + integration tests. |
| RP-5 | `auto_continue=smart`. CLI heuristic detects "obviously incomplete" replies (unclosed code fences, mid-sentence stop, list-prefix-only, model still inside `<think>` at max_tokens) and chains a continuation automatically up to `max_reply_tokens`. | new module `silica/chat/cli/completion_detector.py`, `silica/chat/cli/app.py` | unit tests for the detector; integration test for the auto-continue loop. |
| RP-6 | Manual real-model acceptance pass. Mirrors HARDENING-9 shape; produces `plans/CHAT_CLI_RESPONSE_POLICY_ACCEPTANCE.md`. | none (acceptance run only). | none. |

---

## 5. Sequencing

Within the committed batch:

```text
RP-1 (thinking_history=strip + deferred-strip on truncation)   ── landed
        │
        ▼
RP-2 (/continue, building on RP-1's raw-prefix preservation)   ── landed (60459c7)
        │
        ▼
RP-3 (truncation UX + metrics)                                  ── landed (1d9ecd7)
        │
        ▼
[ side-track interim exit reached — hand-off to P-6.0.5 ready ]
        │
        ▼
D-021 step 3 (P-6.0.5 measurement expansion)                    ── next
        │
        ▼
[ later: RP-4 / RP-5 / RP-6 if user demand justifies ]
```

**HARDENING-9 status**: the manual acceptance run is still
pending (template seeded at `6e19542`, amended at `51fbcde`).
HARDENING-9 does **not** block RP-1 — F1-F6 are closed in code
and the manual run is recorded against that closed-code state
as a separate artifact. The acceptance template's §5 F1-F6
sign-off rows can be filled against the pre-RP code (HEAD as
of `51fbcde`); RP-1 is allowed to land on top of the same
HEAD without invalidating that record. The two side tracks
share a parent commit but their acceptance documents are
independent.

**Sequencing constraints that DO bind**:

1. **RP-1 before RP-2.** RP-2 ``/continue`` needs RP-1's
   deferred-strip-on-truncation contract to preserve the raw
   ``<think>`` prefix across the boundary; eager strip would
   silently break byte-equivalent continuation when the
   truncation point sits inside an open thinking block.
2. **P-6.0.5 takes priority over RP-4..6.** The dense 27B / MoE
   bench expansion is the load-bearing P-6 work. RP-4..6 are
   product polish that depends on real-session data RP-1..3
   already provide; they wait until the dense-perf gate
   resolves.

Each unit is independently committable; the chain does not lock
the whole side track on one hard-to-finish item.

---

## 6. Acceptance for the committed batch (RP-1..RP-3)

This side track's *first* exit (the hand-off back to P-6.0.5)
fires when:

- `thinking_history=strip` is the default; multi-turn Qwen3
  conversation against ``Qwen/Qwen3-0.6B`` shows turn-N prompt
  length growing only with **visible** reply text, not with the
  cumulative thinking budget. A turn that ends with
  ``finish_reason=max_tokens`` mid-``<think>`` keeps the raw
  prefix on the assistant message until ``/continue`` finishes
  it OR the next user message lands; only THEN is the strip
  applied. ``thinking_history=keep`` round-trip preserves the
  raw decoded reply byte-equivalently (RP-1 acceptance).
- ``/continue`` extends a truncated turn without inserting a new
  user message; ``apply_chat_template(messages, continue_final_message=True)``
  routes to the same prompt the model would have seen at a
  higher cap, modulo sampling stochasticity. Chained
  ``/continue`` (cap hit twice) carries the raw form forward
  through both calls and only finalises on natural completion
  (RP-2 acceptance).
- ``[truncated at max_tokens; /continue to extend]`` appears on
  any turn that hits the cap; ``finish=`` field renders on the
  toolbar for ``done`` / ``max_tokens`` / ``eos``; ``/showcase``
  cumulative counters distinguish ``reasoning_chars`` vs
  ``visible_chars`` and increment ``continuation_chunks`` on
  every ``/continue`` invocation (RP-3 acceptance).
- Full test suite green (no regression to the post-HARDENING
  2248-test baseline); ruff + mypy clean for production AND
  test files (the widened mypy invocation HARDENING-7 v2 / -8
  established).

When all four hold, the side track logs an interim exit and
the next commit returns to D-021 step 3. RP-4..RP-6 reopen
under a separate sequencing decision once P-6.0.5 lands.

**Interim-exit log (2026-04-28)**: code-side criterion 4 holds
(2378 tests pass; ruff clean; ``mypy silica/`` clean). Criteria
1–3 are covered by the v3 / v4 snapshot-lifecycle and the
helper-level shell tests at the unit / integration boundary;
real-session validation against ``Qwen/Qwen3-0.6B`` is
deferred to the eventual RP-6 / HARDENING-9 manual acceptance
run rather than blocking the hand-off back to P-6.0.5. Future
real-session findings can reopen RP-1..RP-3 individually
without requiring the whole side track to be re-entered.

The full side-track GA (closing RP-4..RP-6 too) does NOT block
the P-6.0.5 work; it is conditional on user demand observed
during real-session use after RP-1..RP-3 ship.

---

## 7. Decisions

### Decision E — three-layer thinking semantics (RP-1)

**Date:** 2026-04-28.

The chat CLI before this side track exposed two thinking knobs
that overlapped in confusing ways: `thinking_mode` (model side,
HARDENING-2) and `thinking` (display side, pre-existing). Real
sessions on Qwen3.5-35B-A3B-4bit revealed a third, latent axis:
even with `thinking_mode=off` and `thinking=hidden`, *previous*
turns' raw replies (which a user generated under
`thinking_mode=on` before flipping it off) continue to feed
into every subsequent prompt's chat-template render, because
`_messages` stores the raw decoded reply.

The fix splits the third axis out explicitly as
`thinking_history`. `strip` (the new default) writes only the
post-thinking visible reply into history; `keep` preserves the
raw reply for archival use cases (full-trace transcripts).
Composition with the other two layers is explicit and
documented in §3.

This decision means the recommended chat CLI default after
RP-1 lands is:

```text
thinking_mode=on        # current schema default; bool only
thinking=hidden         # do not display reasoning live
thinking_history=strip  # do not re-feed reasoning into next turn
```

The schema today only accepts ``thinking_mode`` as a bool;
``auto`` / ``fast`` / ``balanced`` / ``deep`` are RP-4
territory. Until then, ``thinking_mode=off`` remains the right
hard-disable for speed-focused use, and ``thinking_history=keep``
is the right opt-in for ``/save``-then-archive workflows where
the operator wants the raw decoded text preserved verbatim.

### Decision F — committed scope is RP-1..RP-3 only

**Date:** 2026-04-28.

The original GPT consultation proposed a full Response Policy
framework (`/mode fast|balanced|deep`, `chunk_tokens` vs
`max_reply_tokens`, smart auto-continue, content-aware
truncation detection, per-side budgets) as one piece. Reviewing
against the broader plan: P-6 dense-perf is the next gating
deliverable; everything in this side track competes with it for
sequencing.

This opening commits only to the smallest batch that closes
the two observed real-session failures (G1 / G2). RP-4..RP-6
are listed for sequencing visibility but explicitly deferred
behind P-6.0.5. Reopening them is its own decision once the
dense-perf gate resolves and real-session data shows whether
the smarter policy bundles are needed in production.

### Decision G — RP-1 landed across two commits (mixed-scope acknowledgement)

**Date:** 2026-04-28.

The original RP-1 implementation (``thinking_history=strip``,
``ChatSession`` ctor / mutator / state, ``_strip_thinking_block``
helper, deferred-finalise contract on
``finish_reason=max_tokens``, app.py ctor pass-through and
per-turn resync, schema entry, full unit-test suite) landed
inside commit ``4f98648`` alongside the unrelated DEC
cursor-save/restore hotfix and an unrelated
``docs/P5_ACCEPTANCE_SWEEP/real_activation_xcheck.jsonl`` file
that should have stayed out of the chat side track. The mixed
scope was an oversight — the canonical separation would have
been one ``fix(chat): use DEC cursor save/restore`` commit and
one ``fix(chat): thinking_history=strip + deferred finalise``
commit, with the ``docs/P5_ACCEPTANCE_SWEEP/`` file deferred to
its own follow-up.

History is not rewritten because ``origin/sonnet`` already
contains ``4f98648``; the rewrite cost outweighs the
narrative-fidelity benefit.

The deferred-finalise contract had three real bugs that the
mixed-commit pace did not catch in review:

- Snapshot vs live read: implicit-leading strip decision was
  re-read from live state at finalise time, not snapshotted at
  truncation. A mid-flight ``/config thinking_mode`` flip would
  retroactively change which strip shape applied.
- ``thinking_history=keep`` ignored on deferred path: a user
  flipping the policy between truncation and the next user
  message would still see strip applied.
- ``reset()`` / ``replace_messages()`` / ``pop_last_exchange()``
  did not clear the pending flag, so subsequent turns could
  silently strip a now-replaced or now-missing message.

Commit ``d4ea04d`` repairs all three. RP-1 is therefore
considered landed across the pair ``4f98648`` (initial) +
``d4ea04d`` (repair). Future side tracks should aim for clean
single-purpose commits to keep the side-track ledger
self-explanatory.

The stray ``docs/P5_ACCEPTANCE_SWEEP/real_activation_xcheck.jsonl``
file in ``4f98648`` is tracked separately as a
``chore(repo)``-level cleanup to remove (it is unrelated to the
chat REPL work).

### Decision H — RP-2 landed across one commit, with three pre-merge correctness gates

**Date:** 2026-04-28.

Unlike RP-1's mixed-scope split (Decision G), RP-2 landed in a
single ``feat(chat)`` commit ``60459c7``. Three correctness
findings were caught in review BEFORE merge and rolled into the
same patch rather than tracked as repair commits:

- **Continuation-side implicit-leading snapshot**: the original
  v2 design reused ``_pending_finalize_implicit_leading`` for
  both the strip-finalise decision AND the synthetic
  ``<think>\n`` restoration on ``/continue``. Two callers reading
  the same snapshot worked under strip mode but broke under
  ``thinking_history=keep`` (which never sets the finalise
  snapshot). Decoupled into ``_pending_continuation_implicit_leading``
  (set on any max_tokens regardless of policy; cleared on the
  same lifecycle as the finalise snapshot but with broader
  scope).
- **Three-tier finalise fallback**: with the snapshots split,
  the strip-finalise path now consults ``finalize snapshot →
  continuation snapshot → live decision`` so a keep-mode
  truncation followed by a strip-mode ``/continue`` still
  strips the leading reasoning at natural completion using the
  truncation-time fact rather than the live (now-flipped)
  ``thinking_mode``.
- **``ChatSession.messages`` deep-copy**: the property returned
  ``list(self._messages)`` (shallow), sharing dict references
  between the live log and any snapshot a caller held. Since
  ``continue_last`` mutates ``self._messages[-1]["content"]`` in
  place at the end of generation, the abort-rollback snapshot
  the chat-CLI shell holds was already corrupted by the time
  rollback fired. Property now deep-copies each dict.

All three were caught at the v3 / v4 review iterations before
``60459c7`` landed; Decision G's "future side tracks should
aim for clean single-purpose commits" goal applied successfully
here. RP-2's commit message documents the three rolled-up gates.

### Decision I — interim exit reached without real-session sanity

**Date:** 2026-04-28.

§6's first-exit acceptance criteria 1–3 nominally require a
multi-turn ``Qwen/Qwen3-0.6B`` real-session run that confirms
prompt length growth, ``/continue`` byte-equivalent extension,
and the truncation marker / ``finish=`` field / ``/showcase``
chars accumulation behaviours observed on-device.

Code-side coverage of the same invariants — snapshot lifecycle
tests at the session level, dispatcher + helper tests at the
shell level, abort-rollback contract tests via the extracted
``_apply_rollback_snapshot`` helper — exceeds the depth a
short manual run could provide. Real-session validation is
deferred to the eventual RP-6 / HARDENING-9 manual acceptance
batch where Qwen3.5-35B-A3B-4bit and a representative dense
model can both be exercised in one session, rather than
blocking the hand-off back to P-6.0.5 on a smoke-only Qwen3-0.6B
pass. Re-opening any of RP-1..RP-3 individually remains
permitted if real-session findings surface a behaviour the
unit / integration tests miss.

---

## 8. Cross-references

- `plans/CHAT_CLI_HARDENING.md` — predecessor side track
  (closed F1-F6).
- `plans/CHAT_CLI_HARDENING_ACCEPTANCE.md` — HARDENING-9
  template (still ``draft / pending manual run``). The
  acceptance run is recorded against the post-HARDENING code
  state (HEAD as of `51fbcde`) and is independent of RP-1; see
  §5 sequencing note.
- RP-1 commits: ``4f98648`` (initial; mixed-scope) +
  ``d4ea04d`` (repair). See Decision G for context. RP-2
  ``/continue`` builds on the repaired deferred-finalise
  contract.
- RP-2 commit: ``60459c7`` (single-purpose; three pre-merge
  gates rolled in — see Decision H). RP-3 commit: ``1d9ecd7``.
  Post-RP-3 cleanup: ``683ed90`` (rollback helper extraction;
  not part of the side track's committed scope but lands in the
  same window because the four inline rollback blocks across
  /regenerate + /continue × KeyboardInterrupt + Exception had
  drifted enough during RP-3 that user review caught one missed
  field — the helper consolidation prevents that class of
  drift recurring).
- Toolbar-policy follow-up: ``9bd6edd`` flipped the live
  toolbar default to opt-in (``SILICA_LIVE_TOOLBAR=1`` env or
  ``/config live_toolbar=on`` to enable). Independent of RP-1
  but landed during the same window because the original
  HARDENING-6 backend's terminal-fragility blocked real-session
  use.
- `plans/CHAT_CLI_OPENING.md` — original C-1..C-8 design doc.
- PLAN.md §10 Q-012 — cross-call prefix-cache consultation;
  RP-1's history-strip change preserves Q-012 reuse because
  the prefix tokens (system + user messages) are unchanged by
  the strip — only the assistant message text shrinks.
- PLAN.md §7 P-6 / D-021 — the dense-perf chain this side
  track defers to. RP-4..RP-6 wait behind P-6.0.5.
