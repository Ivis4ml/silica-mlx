"""P-4.5-C.1 VectorCodec runtime-integration tests (side-level API after P-5-A.0.4).

Verifies that the Option (B) hook in ``SyntheticPrefixBlockStore``
(see plans/P4_5_C_KVCODEC_OPENING.md §3.2 / §6 / §8) turns
``IdentityCodec.encode_tensor`` / ``decode_tensor`` into runtime callers
on the real Qwen3-0.6B forward path. Five assertions, matching the
opening doc's §8 acceptance specification:

  §8.1 — Encode counter on cold admission (miss path:
         ``_extract_and_insert_prefix`` runs on row termination →
         ``RadixPrefixCache.insert_detached`` →
         ``store.register_detached`` → ``k_codec.encode_tensor`` +
         ``v_codec.encode_tensor`` per layer per aligned block —
         two side calls per layer rather than one pair call, hence
         the counter arithmetic multiplies by 2 under the shorthand).
  §8.2 — Decode counter on a second prompt admitted mid-run over the
         *same* ``shared_pc`` (hit path:
         ``_admit_waiting_requests`` → ``peek`` →
         ``_admit_single_hit_row`` → ``lookup`` →
         ``fetch_detached_blocks`` → ``store.fetch_detached`` →
         ``k_codec.decode_tensor`` + ``v_codec.decode_tensor``). The
         hit path only fires under mid-run admission, never under the
         initial cohort seal in ``_prepare_cohort``; the test therefore
         uses a single ``generate_batch([p, p], max_batch_size=1)``
         call so the second prompt enters the waiting queue and
         lights up the hit branch after row 0 reclaims and registers
         its prefix.
  §8.3 — ``store.resident_bytes()`` equals the radix-node-derived
         total ``len(store.live_block_ids()) × num_layers ×
         block_size × (2 × n_kv_heads × head_dim × dtype.size)`` and
         the ``prefix_cache.node_count()`` view agrees. Explicitly
         **not** ``_count_evictable_prefix_blocks × _kv_bytes_per_block``
         (that budgeter helper counts leaf-zero-hit only and
         under-reports internal radix nodes).
  §8.4 — Token-stream byte-identity between the no-codec baseline
         path (``codec=None`` pass-through) and the IdentityCodec
         path under the same paired workload.

Plus two defensive invariants from the C.0 review follow-ups:

  D-1 — Tokenized prompt length pins to a specific value, must be
        ``>= 2 × block_size + 1`` (the ``+1`` satisfies batcher
        invariant S-5 edge 1 — see ``silica/scheduler/batcher.py``
        ~ line 1011's ``max_aligned = ((len - 1) // block_size) ×
        block_size``) AND must **not** be an exact multiple of
        block_size (otherwise cold encode counts
        ``floor(len/bs)`` blocks but paired-hit decode counts
        ``floor((len-1)/bs) = floor(len/bs) - 1`` blocks and the two
        acceptances become asymmetric). Parametrized via the shared
        ``_PROMPT_FIXTURE`` — a tokenizer change that violates either
        half is caught here rather than in §8.1 / §8.2 with a
        confusing count mismatch.

  D-2 — ``len(store._detached) == len(store.live_block_ids())`` after
        every admission under Option (B). A future code path that
        retains a source ref without calling ``register_detached``
        (the ``insert(tokens, block_ids)`` pattern rather than
        ``insert_detached``) would leave a block live on the source
        side but not in the detached dict, and ``resident_bytes()``
        would under-report vs ``live_block_ids`` count. The current
        batcher miss path exclusively uses ``insert_detached``, so
        this invariant is trivially true today; the assertion is a
        tripwire for future drift.

**Gating.** Single gate: HF cache must have ``Qwen/Qwen3-0.6B``.
0.6B is small enough that no strong env-var gate (of the
``SILICA_REAL_QWEN3_5_MOE`` style used for the MoE / 27B rows) is
required — mirrors ``tests/test_engine_admission_reorder.py`` §5.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import pytest

from silica.core.events import BatchEvent
from silica.core.sampling import SamplingParams
from silica.engine import Engine
from silica.kvcache.codec import IdentityCodec, RawFp16Payload, VectorCodec
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.models.factory import adapter_for_repo

_REPO = "Qwen/Qwen3-0.6B"
_BLOCK_SIZE = 16
_PROMPT_MIN_TOKENS = 2 * _BLOCK_SIZE + 1  # 33 — see module docstring D-1.

# Pinned test prompt. Chosen so Qwen3-0.6B tokenization yields a length
# that (a) is >= 33 and (b) is not an exact multiple of 16. Under the
# Qwen3 tokenizer shipped with Qwen/Qwen3-0.6B this string measures at
# 34 tokens (mod 16 == 2). If a tokenizer update shifts either half,
# ``test_prompt_tokenization_invariants`` fails loudly rather than
# letting §8.1 / §8.2 produce a misleading count asymmetry.
_PROMPT_FIXTURE = (
    "The continuous-batching scheduler allocates fresh KV blocks at "
    "admission time, retains source references through the radix "
    "tree, and releases detached storage only once hit counts drop "
    "to zero."
)


def _hf_cache_has_repo(repo: str) -> bool:
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    cache_dir = (
        Path(hf_home)
        / "hub"
        / f"models--{repo.replace('/', '--')}"
    )
    return cache_dir.exists()


_SKIP = not _hf_cache_has_repo(_REPO)
_SKIP_REASON = (
    f"{_REPO} not present in the local HF cache. Run any cache-only "
    f"P-4 bench scenario once to populate."
)


class _CountingIdentityCodec:
    """Side-level ``VectorCodec[RawFp16Payload]`` wrapper that counts
    ``encode_tensor`` / ``decode_tensor`` invocations.

    Delegates to an inner ``IdentityCodec`` so byte-identity against the
    no-codec baseline is preserved. When installed via the ``codec=X``
    shorthand on ``SyntheticPrefixBlockStore``, the store sets
    ``k_codec = v_codec = self`` so K and V calls flow through the same
    instance: ``encode_calls`` tallies ``2 × num_layers`` per
    ``register_detached`` call (K-side + V-side combined through this
    one counter). Split-mode K/V dispatch independence is regression-
    locked separately in ``tests/test_prefix_store.py``.
    """

    def __init__(
        self,
        *,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
    ) -> None:
        self._inner = IdentityCodec(
            block_size=block_size,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        self.block_size = block_size
        self.dtype = dtype
        self.encode_calls = 0
        self.decode_calls = 0

    def encode_tensor(self, x: mx.array) -> RawFp16Payload:
        self.encode_calls += 1
        return self._inner.encode_tensor(x)

    def decode_tensor(self, payload: RawFp16Payload) -> mx.array:
        self.decode_calls += 1
        return self._inner.decode_tensor(payload)

    def logical_bytes(self, num_tokens: int) -> int:
        return self._inner.logical_bytes(num_tokens)

    def resident_bytes(self, num_blocks: int) -> int:
        return self._inner.resident_bytes(num_blocks)


# --- Shared fixtures ---------------------------------------------------

if not _SKIP:
    # Module-scoped: loading Qwen3-0.6B is ~2 s even from cache; avoid
    # reloading per test. Each test constructs its own Engine, store,
    # and RadixPrefixCache against the shared adapter — ContinuousBatcher
    # is re-instantiated inside every ``generate_batch`` call, so the
    # shared adapter does not carry any per-test state.
    @pytest.fixture(scope="module")
    def adapter_kv():
        return adapter_for_repo(_REPO)


@pytest.fixture
def counting_codec(adapter_kv) -> _CountingIdentityCodec:
    adapter, _ = adapter_kv
    layout = adapter.kv_layout()
    return _CountingIdentityCodec(
        block_size=_BLOCK_SIZE,
        n_kv_heads=layout.n_kv_heads,
        head_dim=layout.head_dim,
    )


@pytest.fixture
def greedy_params() -> SamplingParams:
    # ``temperature=0.0`` triggers the sampler's argmax fast path
    # (silica/core/sampling.py ``is_greedy``). ``stop_token_ids=()``
    # disables EOS so the full 4-token budget is spent regardless of
    # what the model wants to emit — matches the byte-identity
    # invariant's need for deterministic length.
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=4,
        stop_token_ids=(),
    )


def _drain(events_iter) -> list[BatchEvent]:
    return list(events_iter)


def _tokens_by_req(events: list[BatchEvent]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for ev in events:
        if ev.kind == "token" and ev.token_id is not None:
            out.setdefault(ev.req_index, []).append(ev.token_id)
    return out


# --- Section 1: tokenization-length invariant (D-1) --------------------


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_prompt_tokenization_invariants(adapter_kv):
    """D-1 guard: pin the test prompt's Qwen3-0.6B tokenized length.

    Must be >= 33 (``2 × block_size + 1``) and **not** an exact
    multiple of block_size. A tokenizer change that violates either
    half invalidates the §8.1 / §8.2 block-count arithmetic; fail
    loudly here before the downstream counter assertions do.
    """
    adapter, _ = adapter_kv
    tokenizer = adapter.tokenizer()
    tokens = list(tokenizer.encode(_PROMPT_FIXTURE))
    # Pin the exact tokenized length. The opening doc + test module
    # docstring both refer to "34 tokens (mod 16 == 2)" when walking
    # through block-count arithmetic; an even-35 or 47 drift would
    # still satisfy the >= 33 / %16 != 0 guards below but invalidate
    # those block-count worked examples. Fail loudly on any change.
    assert len(tokens) == 34, (
        f"prompt must tokenize to exactly 34 tokens under the pinned "
        f"Qwen3-0.6B tokenizer shipped with this fixture (opening doc "
        f"§8.1 + module docstring quote block counts against this "
        f"specific length); got len={len(tokens)}. A tokenizer change "
        f"is the most likely cause — re-select _PROMPT_FIXTURE to a "
        f"value that again yields 34 tokens (mod 16 == 2), or update "
        f"every downstream block-count reference together."
    )
    assert len(tokens) >= _PROMPT_MIN_TOKENS, (
        f"prompt must tokenize to >= {_PROMPT_MIN_TOKENS} tokens "
        f"(2 × block_size + 1) to satisfy batcher invariant S-5 "
        f"edge 1 and produce at least 2 aligned blocks; got "
        f"len={len(tokens)}. Lengthen _PROMPT_FIXTURE."
    )
    assert len(tokens) % _BLOCK_SIZE != 0, (
        f"prompt tokenizes to {len(tokens)} tokens, an exact multiple "
        f"of block_size={_BLOCK_SIZE}. Cold encode would count "
        f"floor({len(tokens)}/{_BLOCK_SIZE}) blocks but paired-hit "
        f"decode would count floor({len(tokens) - 1}/{_BLOCK_SIZE}) = "
        f"one fewer — §8.1 and §8.2 acceptances become asymmetric. "
        f"Adjust _PROMPT_FIXTURE."
    )


# --- Section 2: §8.1 + §8.2 encode + decode counters (single call) -----


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_encode_and_decode_counters_on_paired_prompts(
    adapter_kv, counting_codec, greedy_params
):
    """Combines §8.1 and §8.2 into one run because the decode path
    only fires under **mid-run admission** (``_admit_waiting_requests``
    → ``_admit_single_hit_row``), not under the initial cohort seal in
    ``_prepare_cohort``.

    Workload: ``generate_batch([p, p], max_batch_size=1)``. Prompt 0
    admits into the initial cohort; prompt 1 enters the waiting queue.
    Prompt 0 reaches terminal state after ``max_tokens=4`` and the
    next ``step()`` reclaims it — ``_extract_and_insert_prefix``
    slices the aligned prefix out of its BatchKVCache and calls
    ``insert_detached`` → ``register_detached`` → ``encode_block``.
    The same step then calls ``_admit_waiting_requests``, which sees
    prompt 1 and a populated prefix cache: peek finds the full
    aligned prefix, so ``_admit_single_hit_row`` → ``lookup`` →
    ``fetch_detached_blocks`` → ``fetch_detached`` →
    ``decode_block`` fires.

    Encode-side assertion (side-level API, P-5-A.0.4):
      encode_calls >= floor(len / block_size) × num_layers × 2
      (``_extract_and_insert_prefix`` uses
      ``aligned_tokens = (computed_len // block_size) × block_size``;
      ``computed_len`` at row-0 termination is ``prompt_len +
      max_tokens``, so the floor divides by block_size without the
      ``- 1`` adjustment. Under the side-level codec, each aligned
      block generates **one K-side encode plus one V-side encode** —
      the ``× 2`` factor. Installed here via ``codec=`` shorthand so
      both sides flow through the same counter.)

    Decode-side assertion:
      decode_calls >= floor((len - 1) / block_size) × num_layers × 2
      (``_admit_single_hit_row`` uses ``max_aligned = ((prompt_len -
      1) // block_size) × block_size`` to leave at least one suffix
      token for the first-token prefill — ``silica/scheduler/batcher.py``
      ~ line 1011 / S-5 edge 1. The ``× 2`` factor is the K-side +
      V-side split per the side-level API.)

    Under the guarded tokenization (§1: ``len % block_size != 0``),
    these two lower bounds are equal.
    """
    adapter, kv = adapter_kv
    engine = Engine(adapter, kv)
    tokenizer = adapter.tokenizer()

    store = SyntheticPrefixBlockStore(
        block_size=_BLOCK_SIZE, codec=counting_codec
    )
    shared_pc = RadixPrefixCache(block_size=_BLOCK_SIZE, store=store)

    events = _drain(
        engine.generate_batch(
            [_PROMPT_FIXTURE, _PROMPT_FIXTURE],
            greedy_params,
            prefix_cache=shared_pc,
            max_batch_size=1,
        )
    )

    # Scheduler hygiene — an ``aborted`` event would mask a codec
    # integration bug behind a scheduler fault.
    aborts = [e for e in events if e.kind == "aborted"]
    assert aborts == [], f"unexpected aborted events: {aborts}"
    dones = {e.req_index for e in events if e.kind == "done"}
    assert dones == {0, 1}, f"expected both rows done, got {sorted(dones)}"

    prompt_len = len(list(tokenizer.encode(_PROMPT_FIXTURE)))
    n_layers = adapter.config.num_layers

    # Encode-side: row 0 termination registers
    # floor((prompt_len + max_tokens) / block_size) aligned blocks.
    # Lower-bound by floor(prompt_len / block_size) which is always
    # <= the actual aligned count and does not depend on decode
    # length: if future decode-tokens count changes, this assertion
    # stays valid. Under the side-level API every aligned block fires
    # one K-side + one V-side encode_tensor call through the shared
    # counting codec, so multiply by 2.
    expected_encode_blocks = prompt_len // _BLOCK_SIZE
    expected_encode_calls = expected_encode_blocks * n_layers * 2
    assert counting_codec.encode_calls >= expected_encode_calls, (
        f"expected encode_calls >= {expected_encode_blocks} blocks × "
        f"{n_layers} layers × 2 sides = {expected_encode_calls}; got "
        f"{counting_codec.encode_calls}"
    )

    # Decode-side: row 1 mid-run admit hits the prefix with
    # usable_hit_blocks = floor((prompt_len - 1) / block_size).
    # Same × 2 factor for the K-side + V-side decode_tensor calls.
    usable_hit_blocks = (prompt_len - 1) // _BLOCK_SIZE
    expected_decode_calls = usable_hit_blocks * n_layers * 2
    assert counting_codec.decode_calls >= expected_decode_calls, (
        f"expected decode_calls >= {usable_hit_blocks} usable-hit "
        f"blocks × {n_layers} layers × 2 sides = {expected_decode_calls}; "
        f"got {counting_codec.decode_calls}"
    )

    # D-2 defensive invariant: every source-ref'd block has detached K/V.
    live_ids = store.live_block_ids()
    assert len(store._detached) == len(live_ids), (
        f"source-ref'd blocks {sorted(live_ids)} must all have "
        f"detached K/V; store._detached covers "
        f"{sorted(store._detached.keys())}"
    )


# --- Section 4: §8.3 resident_bytes parity with radix-node total -------


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_resident_bytes_matches_radix_node_total(
    adapter_kv, counting_codec, greedy_params
):
    """§8.3 — ``store.resident_bytes()`` equals
    ``total_blocks × num_layers × block_size × (2 × n_kv_heads ×
    head_dim × dtype.size)`` under IdentityCodec. The right-hand side
    uses the **per-layer** K+V byte-per-token cost, not
    ``MemoryBudgeter.bytes_per_token`` which is already all-layer-summed
    (multiplying that by ``num_layers`` would double-count —
    see P-4.5-C.0 opening §6.2 "beware the name collision" paragraph).

    Also asserts ``prefix_cache.node_count() ==
    len(store.live_block_ids())`` (1:1 correspondence under
    ``insert_detached``).
    """
    adapter, kv = adapter_kv
    engine = Engine(adapter, kv)
    layout = adapter.kv_layout()

    store = SyntheticPrefixBlockStore(
        block_size=_BLOCK_SIZE, codec=counting_codec
    )
    shared_pc = RadixPrefixCache(block_size=_BLOCK_SIZE, store=store)
    # Single ``[p, p]`` workload — same shape as §8.1/§8.2 so the
    # store is populated by the time resident_bytes is read.
    _drain(
        engine.generate_batch(
            [_PROMPT_FIXTURE, _PROMPT_FIXTURE],
            greedy_params,
            prefix_cache=shared_pc,
            max_batch_size=1,
        )
    )

    total_blocks = len(store.live_block_ids())
    assert total_blocks > 0, (
        "prefix-cache store should be populated after the paired-prompts "
        "run; got zero live blocks. Row-0 reclaim / _extract_and_insert_prefix "
        "did not fire."
    )
    num_layers = adapter.config.num_layers
    bytes_per_token_per_layer = (
        2 * layout.n_kv_heads * layout.head_dim * layout.dtype.size
    )
    expected = (
        total_blocks * num_layers * _BLOCK_SIZE * bytes_per_token_per_layer
    )

    assert store.resident_bytes() == expected, (
        f"store.resident_bytes() = {store.resident_bytes()} "
        f"!= expected {expected} "
        f"({total_blocks} blocks × {num_layers} layers × "
        f"{_BLOCK_SIZE} tokens × {bytes_per_token_per_layer} B/token/layer)"
    )
    assert shared_pc.node_count() == total_blocks, (
        f"radix node_count ({shared_pc.node_count()}) must match "
        f"store.live_block_ids ({total_blocks}) — 1:1 under "
        f"insert_detached"
    )
    # D-2 defensive invariant.
    assert len(store._detached) == total_blocks


# --- Section 5: §8.4 byte-identity vs no-codec baseline ----------------


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_codec_path_token_stream_matches_no_codec_baseline(
    adapter_kv, greedy_params
):
    """§8.4 — paired ``generate_batch`` runs (cold + repeat) produce
    byte-identical token streams whether the store is constructed with
    ``codec=None`` (pass-through) or ``codec=IdentityCodec(...)``.

    Rationale: ``RawFp16Payload.t`` is the original tensor held by
    reference (``silica/kvcache/codec.py::IdentityCodec.encode_tensor``
    assigns without copy); both paths therefore hand
    ``build_seeded_batch_kv`` the same underlying ``mx.array`` objects,
    producing bitwise-identical concatenated K / V and bitwise-identical
    greedy-argmax token streams.

    Deliberately uses ``temperature=0.0`` and ``stop_token_ids=()`` so
    the sampler path is deterministic and both runs spend the full
    ``max_tokens`` budget.
    """
    adapter, kv = adapter_kv
    layout = adapter.kv_layout()

    def paired_run(codec: VectorCodec | None) -> dict[int, list[int]]:
        # Fresh Engine + pc + store each run so no state leaks between
        # the baseline and codec paths. Use the same ``[p, p]``
        # workload as §8.1-§8.3 so both encode (row 0 termination +
        # reclaim) and decode (row 1 mid-run admit hit) paths run on
        # the codec path.
        engine = Engine(adapter, kv)
        store = SyntheticPrefixBlockStore(
            block_size=_BLOCK_SIZE, codec=codec
        )
        shared_pc = RadixPrefixCache(
            block_size=_BLOCK_SIZE, store=store
        )
        events = _drain(
            engine.generate_batch(
                [_PROMPT_FIXTURE, _PROMPT_FIXTURE],
                greedy_params,
                prefix_cache=shared_pc,
                max_batch_size=1,
            )
        )
        by_req = _tokens_by_req(events)
        assert set(by_req.keys()) == {0, 1}
        return by_req

    baseline_tokens = paired_run(None)
    identity_codec = IdentityCodec(
        block_size=_BLOCK_SIZE,
        n_kv_heads=layout.n_kv_heads,
        head_dim=layout.head_dim,
    )
    codec_tokens = paired_run(identity_codec)

    assert baseline_tokens == codec_tokens, (
        "IdentityCodec path must produce byte-identical tokens vs "
        "no-codec pass-through baseline (both paths hand the same "
        "tensor refs to build_seeded_batch_kv — pass-through stores "
        "raw mx.array in _DetachedLayer, IdentityCodec wraps in "
        "RawFp16Payload whose .t holds the same reference). "
        f"baseline={baseline_tokens} codec={codec_tokens}"
    )
    # Row 1 walks the hit path on both runs; its token stream is where
    # any codec-induced drift would first appear. Both rows must spend
    # the full max_tokens budget since stop_token_ids=() disables EOS.
    for req in (0, 1):
        assert len(baseline_tokens[req]) == greedy_params.max_tokens, (
            f"row {req}: both runs must spend the full "
            f"max_tokens={greedy_params.max_tokens} budget; got "
            f"len={len(baseline_tokens[req])}"
        )


# --- Section 6: tensor-reference preservation (no model needed) --------


def test_identity_codec_path_preserves_tensor_references() -> None:
    """C.0 opening §8.4 C.1 note — assert at the tensor-reference
    level that ``store.fetch_detached`` hands back the same
    ``mx.array`` objects that were originally handed to
    ``store.register_detached``.

    Under the side-level API, ``IdentityCodec.encode_tensor`` returns
    ``RawFp16Payload(t=x)`` without copying ``x`` and
    ``decode_tensor(payload)`` returns ``payload.t``. A future codec
    that silently inserts a defensive copy anywhere in this chain
    would regress from "byte-identical reference" to "byte-identical
    value" without failing any other test in this suite. This
    tripwire uses ``is`` to detect the regression the moment it
    happens.

    Runs without HF cache — uses synthetic tensors end-to-end.
    """
    block_size = 8
    n_kv_heads = 2
    head_dim = 4
    shape = (1, n_kv_heads, block_size, head_dim)

    k = mx.arange(1 * n_kv_heads * block_size * head_dim, dtype=mx.float16).reshape(shape)
    v = mx.arange(1 * n_kv_heads * block_size * head_dim, dtype=mx.float16).reshape(shape) * 2

    codec = IdentityCodec(
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
    )
    store = SyntheticPrefixBlockStore(block_size=block_size, codec=codec)
    bid = store.allocate_id()
    store.retain_source(bid)
    store.register_detached(bid, [(k, v)])

    fetched = store.fetch_detached(bid)
    assert len(fetched) == 1
    (fk, fv) = fetched[0]

    assert fk is k, (
        "IdentityCodec path must return the same K tensor reference "
        "that was handed to register_detached; a defensive copy "
        "anywhere in encode_tensor / decode_tensor would break byte-"
        "identity silently. Got a different object."
    )
    assert fv is v, (
        "IdentityCodec path must return the same V tensor reference "
        "that was handed to register_detached."
    )


def test_no_codec_pass_through_also_preserves_tensor_references() -> None:
    """Same reference invariant for the ``codec=None`` pass-through
    branch of ``SyntheticPrefixBlockStore._encode_layer`` /
    ``_decode_layer``. Ensures the two paths (pass-through /
    IdentityCodec) agree on reference-level behaviour, so any C.1-
    visible byte-identity between them is inherited by §8.4 rather
    than arising from a coincidental value equality.
    """
    block_size = 8
    n_kv_heads = 2
    head_dim = 4
    shape = (1, n_kv_heads, block_size, head_dim)

    k = mx.zeros(shape, dtype=mx.float16)
    v = mx.ones(shape, dtype=mx.float16)

    store = SyntheticPrefixBlockStore(block_size=block_size)  # codec=None
    bid = store.allocate_id()
    store.retain_source(bid)
    store.register_detached(bid, [(k, v)])

    (fk, fv), = store.fetch_detached(bid)
    assert fk is k
    assert fv is v

    # resident_bytes equals tensor nbytes under pass-through.
    expected = k.nbytes + v.nbytes
    assert store.resident_bytes() == expected
