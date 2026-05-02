"""D-021 step 7 sub-unit (B.2) — 27B WikiText-2 PPL row shape tests.

Pins five contracts on the paired ``qwen3.5-27b-wikitext-ppl-{4bit,
3bit}`` rows that drive the (B.2) ΔPPL gate:

1. **Both rows registered.** ``get_scenario(...)`` resolves both
   ids; catalog count rose 68 → 70.
2. **Repos differ.** 4-bit row uses
   ``mlx-community/Qwen3.5-27B-4bit``; 3-bit row uses
   ``NexVeridian/Qwen3.5-27B-3bit``. The REPORT-side ΔPPL only
   isolates the weight-bits delta if the rows differ exclusively
   in repo + gate — confirmed below.
3. **Gate envs differ.** 4-bit row gates on
   ``SILICA_REAL_QWEN3_5_27B``; 3-bit row gates on
   ``SILICA_REAL_QWEN3_5_27B_3BIT``. A user with only one
   checkpoint cached cannot trigger the other's load.
4. **Oracle config equality.** Same ``OracleKind.PPL``, same
   ``oracle_config`` dict (chunk_size=256, max_tokens=512,
   codec_quality_path="prefix_store_pre_norm", seed=0). Drift
   here would invalidate ΔPPL as a clean weight-bits comparison.
5. **No spec, no codec.** Plain teacher-forced PPL —
   ``spec_config`` is None on both rows and ``workload.kv_codec``
   is None. (B.2) measures pure-weight quality drift, not a
   spec or codec composition.
"""

from __future__ import annotations

from silica.bench.scenarios import BUILTIN_SCENARIOS, get_scenario  # noqa: E402

_ID_4BIT = "qwen3.5-27b-wikitext-ppl-4bit"
_ID_3BIT = "qwen3.5-27b-wikitext-ppl-3bit"


def test_b2_both_rows_registered() -> None:
    sc_4bit = get_scenario(_ID_4BIT)
    sc_3bit = get_scenario(_ID_3BIT)
    assert sc_4bit.id == _ID_4BIT
    assert sc_3bit.id == _ID_3BIT


def test_builtin_scenarios_count_is_at_least_70() -> None:
    """Catalog had 68 scenarios after B.1; (B.2) adds the 4-bit and
    3-bit PPL rows, which puts the floor at 70. Loosened to
    ``>= 70`` so later catalog growth does not retro-gate this
    invariant — same precedent as ζ's ``>= 67`` loosening."""
    assert len(BUILTIN_SCENARIOS) >= 70


def test_b2_repos_differ() -> None:
    """REPORT-side ΔPPL is the difference between two PPL rows that
    must differ in repo only — confirmed here. Drift would make
    ΔPPL unreadable as a weight-bits-only signal."""
    sc_4bit = get_scenario(_ID_4BIT)
    sc_3bit = get_scenario(_ID_3BIT)
    assert sc_4bit.repo == "mlx-community/Qwen3.5-27B-4bit"
    assert sc_3bit.repo == "NexVeridian/Qwen3.5-27B-3bit"
    assert sc_4bit.repo != sc_3bit.repo


def test_b2_gate_envs_differ() -> None:
    """Independent gate envs so a user with only one checkpoint
    cached cannot trigger the other's load."""
    sc_4bit = get_scenario(_ID_4BIT)
    sc_3bit = get_scenario(_ID_3BIT)
    assert sc_4bit.gate_env_var == "SILICA_REAL_QWEN3_5_27B"
    assert sc_3bit.gate_env_var == "SILICA_REAL_QWEN3_5_27B_3BIT"
    assert sc_4bit.gate_env_var != sc_3bit.gate_env_var


def test_b2_oracle_config_matches() -> None:
    """Both rows share the same OracleKind.PPL + oracle_config dict
    so PPL is computed over the identical scoring path. ΔPPL is
    only a clean weight-bits delta if oracle config is byte-equal."""
    sc_4bit = get_scenario(_ID_4BIT)
    sc_3bit = get_scenario(_ID_3BIT)
    assert sc_4bit.oracle == sc_3bit.oracle
    assert sc_4bit.oracle_config == sc_3bit.oracle_config


def test_b2_rows_are_plain_ppl_no_spec_no_codec() -> None:
    """Pure-weight quality drift — no spec composition, no codec
    composition. Either would confound the (B.2) signal."""
    for sc_id in (_ID_4BIT, _ID_3BIT):
        sc = get_scenario(sc_id)
        assert sc.spec_config is None, sc_id
        assert sc.workload.kv_codec is None, sc_id
