"""silica.bench — P-4 unified benchmark harness.

Public surface covers the scenario schema, the built-in catalog, and
the runner. Oracle implementations are an internal concern of the
runner and are not re-exported at the package level; authors of new
oracles import from ``silica.bench.oracles`` directly.
"""

from silica.bench.runner import BenchRunner, EngineFactory
from silica.bench.scenario import (
    OracleFn,
    OracleKind,
    Scenario,
    ScenarioResult,
    Workload,
    hf_cache_path_for_repo,
)
from silica.bench.scenarios import (
    BUILTIN_SCENARIOS,
    get_scenario,
    list_scenario_ids,
)

__all__ = [
    "BUILTIN_SCENARIOS",
    "BenchRunner",
    "EngineFactory",
    "OracleFn",
    "OracleKind",
    "Scenario",
    "ScenarioResult",
    "Workload",
    "get_scenario",
    "hf_cache_path_for_repo",
    "list_scenario_ids",
]
