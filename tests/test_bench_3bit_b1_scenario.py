"""D-021 step 7 sub-unit (B.1) — 3-bit b1 scenario shape tests.

Pins three contracts on ``qwen3.5-27b-warm-decode-b1-3bit``:

1. **Workload-shape parity with b1 cousin.** Same prompts,
   max_tokens, max_batch_size, oracle. Ratio between the two rows
   reads as the silica-integrated 3-bit speedup against the
   v1.7.13 P-6.0 b1 anchor without normalisation.
2. **Repo differs.** The 3-bit row's ``repo`` is the
   ``NexVeridian/Qwen3.5-27B-3bit`` 3-bit checkpoint, **not**
   the 4-bit cousin's ``mlx-community/Qwen3.5-27B-4bit``.
3. **Gate differs.** The 3-bit row's ``gate_env_var`` is
   ``SILICA_REAL_QWEN3_5_27B_3BIT``, **not** the 4-bit cousin's
   ``SILICA_REAL_QWEN3_5_27B``. Independent gates so a user
   with only the 4-bit cached cannot trigger an unintended
   ~11 GB 3-bit checkpoint load.

Plus: ``--list`` enumerates the new scenario; ``BUILTIN_SCENARIOS``
count rises 67 → 68.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

from silica.bench.scenarios import BUILTIN_SCENARIOS, get_scenario


def _load_bench_cli_module() -> Any:
    here = Path(__file__).resolve().parents[1]
    path = here / "scripts" / "bench.py"
    spec = importlib.util.spec_from_file_location("_bench_cli_b1_3bit", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_b1_3bit_scenario_registered() -> None:
    sc = get_scenario("qwen3.5-27b-warm-decode-b1-3bit")
    assert sc.id == "qwen3.5-27b-warm-decode-b1-3bit"


def test_builtin_scenarios_count_is_68() -> None:
    """v1.7.20 catalog had 67 scenarios; B.1 adds the b1-3bit row.
    Tracked explicitly so an accidental catalog change is caught."""
    assert len(BUILTIN_SCENARIOS) == 68


def test_b1_3bit_workload_shape_matches_b1_cousin() -> None:
    """Mirror invariant: same workload + oracle + max_tokens +
    max_batch_size as ``qwen3.5-27b-warm-decode-b1``. Drift here
    would invalidate the silica-integrated speedup denominator at
    B.3 attestation time."""
    b1 = get_scenario("qwen3.5-27b-warm-decode-b1")
    b1_3bit = get_scenario("qwen3.5-27b-warm-decode-b1-3bit")
    assert b1_3bit.workload.prompts == b1.workload.prompts
    assert b1_3bit.workload.max_tokens == b1.workload.max_tokens
    assert b1_3bit.workload.max_batch_size == b1.workload.max_batch_size
    assert b1_3bit.oracle == b1.oracle


def test_b1_3bit_repo_differs_from_4bit_cousin() -> None:
    """The 3-bit row points at the matched-family native MLX
    checkpoint identified at B.1's HF lookup; the 4-bit cousin
    stays on its production fixture."""
    b1 = get_scenario("qwen3.5-27b-warm-decode-b1")
    b1_3bit = get_scenario("qwen3.5-27b-warm-decode-b1-3bit")
    assert b1.repo == "mlx-community/Qwen3.5-27B-4bit"
    assert b1_3bit.repo == "NexVeridian/Qwen3.5-27B-3bit"
    assert b1_3bit.repo != b1.repo


def test_b1_3bit_gate_env_differs_from_4bit_cousin() -> None:
    """Independent gate envs — a user with only the 4-bit cached
    cannot trigger an unintended 3-bit checkpoint load."""
    b1 = get_scenario("qwen3.5-27b-warm-decode-b1")
    b1_3bit = get_scenario("qwen3.5-27b-warm-decode-b1-3bit")
    assert b1.gate_env_var == "SILICA_REAL_QWEN3_5_27B"
    assert b1_3bit.gate_env_var == "SILICA_REAL_QWEN3_5_27B_3BIT"
    assert b1_3bit.gate_env_var != b1.gate_env_var


def test_b1_3bit_has_no_spec_config() -> None:
    """The 3-bit b1 row is a plain warm-decode scenario, not a
    spec-on row. (B.2) and any speculative composition are
    separate sub-units / future work."""
    b1_3bit = get_scenario("qwen3.5-27b-warm-decode-b1-3bit")
    assert b1_3bit.spec_config is None


def test_cli_list_surfaces_b1_3bit_row() -> None:
    """``python -m scripts.bench --list`` enumerates the new row so
    users find it by id without grepping source."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.bench", "--list"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "qwen3.5-27b-warm-decode-b1-3bit" in result.stdout


def test_cli_list_module_imports_clean() -> None:
    """Sanity: the bench CLI module loads after the b1-3bit
    addition without import-time exceptions."""
    bench = _load_bench_cli_module()
    parser = bench.build_parser()
    ns = parser.parse_args([])
    # Default speculative=none confirms the CLI is intact post-(ζ).
    assert ns.speculative == "none"
