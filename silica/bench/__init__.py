"""silica.bench — P-4 unified benchmark harness.

Public surface covers the scenario schema, the built-in catalog, and
the runner. Oracle implementations are an internal concern of the
runner and are not re-exported at the package level; authors of new
oracles import from ``silica.bench.oracles`` directly.
"""

from silica.bench.report import render_markdown_report
from silica.bench.runner import (
    BenchRunner,
    DirectBatchedReferenceFn,
    EngineFactory,
    TeacherForcedReferenceFn,
    TeacherForcedSilicaFn,
)
from silica.bench.scenario import (
    OracleFn,
    OracleKind,
    Scenario,
    ScenarioResult,
    VqbenchXcheckSpec,
    Workload,
    hf_cache_path_for_repo,
)
from silica.bench.scenarios import (
    BUILTIN_SCENARIOS,
    get_scenario,
    list_scenario_ids,
)
from silica.bench.vqbench_baseline import (
    VqbenchBaselineResult,
    default_reproduce_script_path,
    parse_headline_row,
    run_vqbench_baseline,
)

__all__ = [
    "BUILTIN_SCENARIOS",
    "BenchRunner",
    "DirectBatchedReferenceFn",
    "EngineFactory",
    "OracleFn",
    "OracleKind",
    "Scenario",
    "ScenarioResult",
    "TeacherForcedReferenceFn",
    "TeacherForcedSilicaFn",
    "VqbenchBaselineResult",
    "VqbenchXcheckSpec",
    "Workload",
    "default_reproduce_script_path",
    "get_scenario",
    "hf_cache_path_for_repo",
    "list_scenario_ids",
    "parse_headline_row",
    "render_markdown_report",
    "run_vqbench_baseline",
]
