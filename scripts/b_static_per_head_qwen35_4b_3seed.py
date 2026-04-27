"""(b-static) re-measurement on Qwen3.5-4B with ``per_head_rotation=True``.

Item C₁ follow-up to v1.7.10's caveat. v1.7.10 found per-head
rotation reduces the silica-vs-vqbench `mean_gap` from 0.150 to
0.066 PPL on the **D.2a path** (`vqbench_aligned`, pre-RoPE
projection-patch oracle), but at the cost of degrading silica's
own codec ΔPPL by 0.22 PPL absolutely (from +0.511 to +0.727 PPL).

The unanswered question: does that absolute regression carry over
to the **production path** (`prefix_store_pre_norm`, P-5-F (3b)
projection-output capture), or is it specific to the D.2a
projection-patch semantics? The (b-static) v1.7.7 close measured
silica's production path against vqbench/REPORT.md's static
baseline at mean ΔPPL = +0.0016 PPL across 3 seeds — essentially
zero. This script re-runs the same workload with
``per_head_rotation=True`` to see whether the production-path
ΔPPL stays at zero, regresses (like D.2a), or improves further.

Workload knobs match `qwen35_4b_b_static_close.md` exactly:

  - Model: ``Qwen/Qwen3.5-4B`` (hybrid DeltaNet + GQA).
  - WikiText-2 first 512 tokens, ``chunk_size = 256``.
  - Seeds {42, 43, 44}.
  - Codec ``BlockTurboQuantMSE`` ``vq_block_size=64`` ``num_bits=4``,
    K + V symmetric, this run with ``per_head_rotation=True``.
  - Routing: ``codec_quality_path = "prefix_store_pre_norm"`` —
    the F.3 production default.

Output: writes
``plans/P5_ACCEPTANCE_SWEEP/qwen35_4b_b_static_per_head_3seeds.jsonl``
(per-seed rows + aggregate) and prints comparison vs the v1.7.7
shared-rotation baseline + the vqbench REPORT.md target.

Run: ``uv run python scripts/b_static_per_head_qwen35_4b_3seed.py``.

Dependencies: HF cache for ``Qwen/Qwen3.5-4B`` and the WikiText-2
cache at ``~/.cache/silica/wikitext2-test.txt``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from silica.bench.ppl_oracle import (  # noqa: E402
    perplexity_from_nll,
    teacher_forced_chunked_nll,
    teacher_forced_chunked_nll_with_codec_pre_norm,
)
from silica.bench.wikitext import load_wikitext_text, tokenize_for_ppl  # noqa: E402
from silica.kvcache.prefix import RadixPrefixCache  # noqa: E402
from silica.kvcache.store import SyntheticPrefixBlockStore  # noqa: E402
from silica.models.factory import adapter_for_repo  # noqa: E402
from silica.vq import BlockTurboQuantMSE  # noqa: E402

REPO = "Qwen/Qwen3.5-4B"
WIKITEXT_PATH = Path.home() / ".cache" / "silica" / "wikitext2-test.txt"
CHUNK_SIZE = 256
MAX_TOKENS = 512
PREFIX_BLOCK_SIZE = 16  # matches _maybe_build_prefix_cache
SEEDS = (42, 43, 44)
OUT_PATH = (
    REPO_ROOT
    / "docs"
    / "P5_ACCEPTANCE_SWEEP"
    / "qwen35_4b_b_static_per_head_3seeds.jsonl"
)

# v1.7.7 (b-static) close baseline — shared rotation, same workload.
# From plans/P5_ACCEPTANCE_SWEEP/qwen35_4b_b_static_3seeds.{jsonl,md}.
SHARED_ROTATION_PER_SEED_DELTA = (0.024785, -0.016742, -0.003374)
SHARED_ROTATION_FP16_PPL = 8.855978965984526


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
    layout = adapter.kv_layout()
    tokenizer = adapter.tokenizer()
    text = load_wikitext_text(WIKITEXT_PATH)
    tokens = tokenize_for_ppl(
        tokenizer,
        text,
        max_tokens=MAX_TOKENS,
        min_tokens=CHUNK_SIZE,
    )
    n_tokens_hint = int(tokens.shape[1]) - 1
    print(
        f"Tokens: shape={tuple(tokens.shape)} → scoring {n_tokens_hint} positions; "
        f"layout: n_kv_heads={layout.n_kv_heads}, head_dim={layout.head_dim}, "
        f"dtype={layout.dtype}"
    )

    # fp16 baseline (seed-independent — fp16 forward is deterministic).
    print("Computing fp16 baseline (seed-independent)...")
    fp16_nll, fp16_n = teacher_forced_chunked_nll(
        adapter, tokens, chunk_size=CHUNK_SIZE
    )
    fp16_ppl = perplexity_from_nll(fp16_nll, fp16_n)
    print(f"  fp16: nll_sum={fp16_nll:.4f}, n={fp16_n}, ppl={fp16_ppl:.6f}")

    rows: list[dict[str, object]] = []
    silica_deltas: list[float] = []
    for seed in SEEDS:
        print(
            f"\nseed={seed}: per-head BlockTQ b64 b4 production path "
            f"(prefix_store_pre_norm)"
        )
        # Fresh codec + cache per seed so the rotation seed propagates
        # into the per-head Haar draws (seed * 1000 + head_idx).
        codec = BlockTurboQuantMSE(
            block_size=PREFIX_BLOCK_SIZE,
            n_kv_heads=layout.n_kv_heads,
            head_dim=layout.head_dim,
            vq_block_size=64,
            num_bits=4,
            seed=seed,
            per_head_rotation=True,
            dtype=layout.dtype,
        )
        store = SyntheticPrefixBlockStore(
            block_size=PREFIX_BLOCK_SIZE, codec=codec
        )
        prefix_cache = RadixPrefixCache(
            block_size=PREFIX_BLOCK_SIZE, store=store
        )
        nll_sum, n_tokens = teacher_forced_chunked_nll_with_codec_pre_norm(
            adapter, prefix_cache, tokens, chunk_size=CHUNK_SIZE
        )
        ppl = perplexity_from_nll(nll_sum, n_tokens)
        delta_ppl = ppl - fp16_ppl
        delta_pct = 100.0 * delta_ppl / fp16_ppl
        silica_deltas.append(delta_ppl)
        print(
            f"  ppl={ppl:.6f}, ΔPPL={delta_ppl:+.6f}, "
            f"ΔPPL%={delta_pct:+.4f}"
        )
        rows.append(
            {
                "scenario_id": "qwen3.5-4b-wikitext-ppl-block-tq-b64-b4",
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
                "prefix_block_size": PREFIX_BLOCK_SIZE,
                "codec_quality_path": "prefix_store_pre_norm",
                "wikitext_path": str(WIKITEXT_PATH),
            }
        )

    silica_mean, silica_std = _mean_std(silica_deltas)
    silica_sem = silica_std / math.sqrt(len(silica_deltas))

    shared_mean, shared_std = _mean_std(list(SHARED_ROTATION_PER_SEED_DELTA))
    shared_sem = shared_std / math.sqrt(len(SHARED_ROTATION_PER_SEED_DELTA))

    # vqbench REPORT.md §3.1 row "Block B=64 4-bit K+V" target.
    vqbench_mean = 0.0
    vqbench_sem = 0.0

    # Aggregated gate vs the shared-rotation v1.7.7 baseline.
    vs_shared_gap = silica_mean - shared_mean
    vs_shared_sem_diff = math.sqrt(silica_sem ** 2 + shared_sem ** 2)
    # Aggregated gate vs vqbench REPORT.md target (the (b-static) gate).
    vs_vqbench_gap = silica_mean - vqbench_mean
    vs_vqbench_sem_diff = math.sqrt(silica_sem ** 2 + vqbench_sem ** 2)

    summary = {
        "scenario_id": "qwen3.5-4b-wikitext-ppl-block-tq-b64-b4",
        "variant": "per_head_rotation_aggregate",
        "silica_per_seed_delta_ppl": silica_deltas,
        "silica_mean_delta_ppl": silica_mean,
        "silica_std_delta_ppl": silica_std,
        "silica_sem_delta_ppl": silica_sem,
        "shared_rotation_per_seed_delta_ppl": list(
            SHARED_ROTATION_PER_SEED_DELTA
        ),
        "shared_rotation_mean_delta_ppl": shared_mean,
        "shared_rotation_std_delta_ppl": shared_std,
        "shared_rotation_sem_delta_ppl": shared_sem,
        "vs_shared_mean_gap": vs_shared_gap,
        "vs_shared_sem_diff": vs_shared_sem_diff,
        "vs_shared_two_sem_diff": 2.0 * vs_shared_sem_diff,
        "vqbench_report_md_mean_delta_ppl": vqbench_mean,
        "vs_vqbench_mean_gap": vs_vqbench_gap,
        "vs_vqbench_sem_diff": vs_vqbench_sem_diff,
        "vs_vqbench_two_sem_diff": 2.0 * vs_vqbench_sem_diff,
        "gate_aggregated_pass_vs_vqbench": (
            abs(vs_vqbench_gap) <= 2.0 * vs_vqbench_sem_diff
            and abs(vs_vqbench_gap) < 1.0
        ),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.write(json.dumps(summary) + "\n")

    print("\n=== aggregate ===")
    print(
        f"silica per-head ΔPPL: per-seed={[round(x, 6) for x in silica_deltas]}, "
        f"mean={silica_mean:+.6f}, std={silica_std:.6f}, "
        f"SEM={silica_sem:.6f}"
    )
    print(
        f"silica shared (v1.7.7 baseline): per-seed="
        f"{list(SHARED_ROTATION_PER_SEED_DELTA)}, "
        f"mean={shared_mean:+.6f}, std={shared_std:.6f}, "
        f"SEM={shared_sem:.6f}"
    )
    print(
        f"vqbench REPORT.md target (\"+0.000% ± 0.000%\"): mean={vqbench_mean:.6f}"
    )
    print(
        f"\nvs shared-rotation: gap={vs_shared_gap:+.6f} PPL "
        f"({'regression' if vs_shared_gap > 0 else 'improvement'}); "
        f"2*SEM_diff={2.0 * vs_shared_sem_diff:.6f}"
    )
    print(
        f"vs vqbench target: gap={vs_vqbench_gap:+.6f} PPL; "
        f"2*SEM_diff={2.0 * vs_vqbench_sem_diff:.6f} → "
        f"gate (i)={'PASS' if abs(vs_vqbench_gap) <= 2.0 * vs_vqbench_sem_diff else 'FAIL'}, "
        f"gate (ii) (<1.0)={'PASS' if abs(vs_vqbench_gap) < 1.0 else 'FAIL'}"
    )
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
