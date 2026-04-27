"""Measurement-only D.2a re-run with ``per_head_rotation=True``.

Item B follow-up to v1.7.8: the (4-b) D.2a 3-seed cross-check
(`plans/P5_D2_INVESTIGATION/d2a_verification_3seeds.jsonl`) measured
``mean_gap = silica.ΔPPL − vqbench.ΔPPL = -0.150 PPL`` at the
``qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned`` row, and
both the (4-b) two-part aggregated gate and silica's per-row
diagnostic warning fired the standing 0.61 PPL gap. The hypothesis:
silica's ``BlockTurboQuantMSE`` shares one Haar rotation across all
heads while vqbench draws a per-head rotation seeded
``actual_seed = run_seed * 1000 + head_idx`` at
`vqbench/scripts/variance_qwen35_4b.py:63`. v1.7.8 added
``per_head_rotation: bool = False`` as opt-in to BlockTQ /
RaBitQ1Bit / ExtRaBitQ. This script empirically pins whether the
rotation choice is the dominant residual contributor to the 0.150
PPL mean_gap.

Run pattern: load Qwen3-0.6B once, tokenize WikiText-2 first 512
tokens (chunk_size=256), compute fp16 baseline once
(seed-independent — fp16 forward is deterministic), then for each
seed in {42, 43, 44} install the per-head BlockTQ codec on
``k_proj`` / ``v_proj`` via the existing
``teacher_forced_chunked_nll_vqbench_aligned`` oracle and record the
ΔPPL.

Output: writes
``plans/P5_D2_INVESTIGATION/per_head_rotation_3seeds.jsonl`` (one
row per seed) and prints the aggregated mean / std / SEM plus the
new mean_gap vs vqbench's locked baseline (0.661 ± 0.347, SEM
0.200) so the (4-b) close decision can compare apples-to-apples.

Run: ``uv run python scripts/d2a_per_head_3seed.py``.

Dependencies: HF cache for ``Qwen/Qwen3-0.6B`` and the WikiText-2
cache at ``~/.cache/silica/wikitext2-test.txt`` (``uv run python
scripts/prepare_wikitext2_cache.py`` if absent).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402

from silica.bench.ppl_oracle import (  # noqa: E402
    perplexity_from_nll,
    teacher_forced_chunked_nll,
    teacher_forced_chunked_nll_vqbench_aligned,
)
from silica.bench.wikitext import load_wikitext_text, tokenize_for_ppl  # noqa: E402
from silica.kvcache.codec import VectorCodec  # noqa: E402
from silica.models.factory import adapter_for_repo  # noqa: E402
from silica.vq import BlockTurboQuantMSE  # noqa: E402

REPO = "Qwen/Qwen3-0.6B"
WIKITEXT_PATH = Path.home() / ".cache" / "silica" / "wikitext2-test.txt"
CHUNK_SIZE = 256
MAX_TOKENS = 512
SEEDS = (42, 43, 44)
OUT_PATH = (
    REPO_ROOT
    / "docs"
    / "P5_D2_INVESTIGATION"
    / "per_head_rotation_3seeds.jsonl"
)

# Vqbench-locked baseline from
# ``plans/P5_D2_INVESTIGATION/d2a_verification_3seeds.jsonl`` —
# vqbench's ΔPPL on the same workload at seeds 42/43/44 with its
# native per-head rotation. mean = 0.661, std = 0.347 (Bessel n-1),
# SEM = std / sqrt(3) = 0.200.
VQBENCH_DELTAS = (0.2699, 0.7848, 0.9289)


def _per_head_block_tq_factory(
    *,
    block_size: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: mx.Dtype = mx.float16,
    seed: int = 42,
) -> VectorCodec:
    """Per-head BlockTQ b64 b4 factory matching the
    ``CodecFactory`` signature in
    ``silica.bench.codec_registry``. Identical to the production
    ``block_tq_b64_b4`` factory except for ``per_head_rotation=True``,
    so the residual ΔPPL difference between the two is attributable
    only to the rotation axis."""
    return BlockTurboQuantMSE(
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        vq_block_size=64,
        num_bits=4,
        seed=seed,
        per_head_rotation=True,
        dtype=dtype,
    )


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def main() -> None:
    if not WIKITEXT_PATH.exists():
        sys.exit(
            f"WikiText-2 cache missing at {WIKITEXT_PATH}. Populate via "
            "`uv run python scripts/prepare_wikitext2_cache.py`."
        )

    print(f"Loading {REPO}...")
    adapter, _kv = adapter_for_repo(REPO)
    tokenizer = adapter.tokenizer()
    text = load_wikitext_text(WIKITEXT_PATH)
    tokens = tokenize_for_ppl(
        tokenizer,
        text,
        max_tokens=MAX_TOKENS,
        min_tokens=CHUNK_SIZE,
    )
    n_tokens_hint = int(tokens.shape[1]) - 1
    print(f"Tokens: shape={tuple(tokens.shape)} → scoring {n_tokens_hint} positions")

    # fp16 baseline. fp16 forward is seed-independent — one call,
    # reused across all per-head codec runs.
    print("Computing fp16 baseline (seed-independent)...")
    fp16_nll, fp16_n = teacher_forced_chunked_nll(
        adapter, tokens, chunk_size=CHUNK_SIZE
    )
    fp16_ppl = perplexity_from_nll(fp16_nll, fp16_n)
    print(f"  fp16: nll_sum={fp16_nll:.4f}, n={fp16_n}, ppl={fp16_ppl:.6f}")

    rows: list[dict[str, object]] = []
    silica_deltas_per_head: list[float] = []
    for seed in SEEDS:
        print(f"\nseed={seed}: per-head BlockTQ b64 b4 vqbench-aligned arm")
        nll_sum, n_tokens = teacher_forced_chunked_nll_vqbench_aligned(
            adapter,
            tokens,
            chunk_size=CHUNK_SIZE,
            codec_factory=_per_head_block_tq_factory,
            seed=seed,
            wrap_v=True,
        )
        ppl = perplexity_from_nll(nll_sum, n_tokens)
        delta_ppl = ppl - fp16_ppl
        delta_pct = 100.0 * delta_ppl / fp16_ppl
        silica_deltas_per_head.append(delta_ppl)
        print(
            f"  ppl={ppl:.6f}, ΔPPL={delta_ppl:+.6f}, ΔPPL%={delta_pct:+.4f}"
        )
        rows.append(
            {
                "scenario_id": "qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned",
                "variant": "per_head_rotation",
                "seed": seed,
                "nll_sum": float(nll_sum),
                "n_tokens": int(n_tokens),
                "ppl": float(ppl),
                "ppl_fp16": float(fp16_ppl),
                "delta_ppl": float(delta_ppl),
                "delta_ppl_pct": float(delta_pct),
                "kv_codec": "block_tq_b64_b4",
                "per_head_rotation": True,
                "vq_block_size": 64,
                "num_bits": 4,
                "chunk_size": CHUNK_SIZE,
                "max_tokens": MAX_TOKENS,
                "wikitext_path": str(WIKITEXT_PATH),
            }
        )

    silica_mean, silica_std = _mean_std(silica_deltas_per_head)
    silica_sem = silica_std / math.sqrt(len(silica_deltas_per_head))

    vqbench_mean, vqbench_std = _mean_std(list(VQBENCH_DELTAS))
    vqbench_sem = vqbench_std / math.sqrt(len(VQBENCH_DELTAS))

    mean_gap = silica_mean - vqbench_mean
    sem_diff = math.sqrt(silica_sem ** 2 + vqbench_sem ** 2)
    two_sem_diff = 2.0 * sem_diff

    summary = {
        "scenario_id": "qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned",
        "variant": "per_head_rotation_aggregate",
        "silica_per_seed_delta_ppl": silica_deltas_per_head,
        "silica_mean_delta_ppl": silica_mean,
        "silica_std_delta_ppl": silica_std,
        "silica_sem_delta_ppl": silica_sem,
        "vqbench_per_seed_delta_ppl": list(VQBENCH_DELTAS),
        "vqbench_mean_delta_ppl": vqbench_mean,
        "vqbench_std_delta_ppl": vqbench_std,
        "vqbench_sem_delta_ppl": vqbench_sem,
        "mean_gap": mean_gap,
        "sem_diff": sem_diff,
        "two_sem_diff": two_sem_diff,
        "gate_aggregated_pass": abs(mean_gap) <= two_sem_diff and abs(mean_gap) < 1.0,
        "baseline_shared_rotation_mean_gap": -0.150,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.write(json.dumps(summary) + "\n")

    print("\n=== aggregate ===")
    print(
        f"silica per-head ΔPPL: per-seed={silica_deltas_per_head}, "
        f"mean={silica_mean:+.6f}, std={silica_std:.6f}, "
        f"SEM={silica_sem:.6f}"
    )
    print(
        f"vqbench (locked):     per-seed={list(VQBENCH_DELTAS)}, "
        f"mean={vqbench_mean:+.6f}, std={vqbench_std:.6f}, "
        f"SEM={vqbench_sem:.6f}"
    )
    print(
        f"mean_gap = silica - vqbench = {mean_gap:+.6f} PPL "
        f"(was {summary['baseline_shared_rotation_mean_gap']:+.3f} "
        f"with shared rotation)"
    )
    print(
        f"|mean_gap|={abs(mean_gap):.6f}, 2*SEM_diff={two_sem_diff:.6f}, "
        f"gate (i)={'PASS' if abs(mean_gap) <= two_sem_diff else 'FAIL'} "
        f"({abs(mean_gap):.6f} {'≤' if abs(mean_gap) <= two_sem_diff else '>'} "
        f"{two_sem_diff:.6f})"
    )
    print(
        f"|mean_gap|={abs(mean_gap):.6f}, gate (ii) (<1.0 PPL)="
        f"{'PASS' if abs(mean_gap) < 1.0 else 'FAIL'}"
    )
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
