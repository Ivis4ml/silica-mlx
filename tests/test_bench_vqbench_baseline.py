"""Unit tests for ``silica.bench.vqbench_baseline`` (P-4.4).

Two layers:

  * Pure parser tests against realistic reproduce-script stdouts
    (the stable "Headline table row:" line, plus edge cases like
    missing rows and partial formats).
  * Subprocess-orchestration tests against a monkeypatched
    ``subprocess.run`` so the test suite never needs vqbench's
    torch / transformers / datasets deps installed. The on-device
    validation against the real reproduce-script is gated on
    ``SILICA_REAL_VQBENCH_BASELINE=1`` and a reachable vqbench
    venv path (``SILICA_VQBENCH_PYTHON``) — not a default dev
    run, because a full reproduce takes ~60 s and downloads
    ~8 GB of weights on first use.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from silica.bench import (
    default_reproduce_script_path,
    parse_headline_row,
    run_vqbench_baseline,
)

# ---------- parser --------------------------------------------------


_REALISTIC_HEADLINE = """\
Loading Qwen/Qwen3.5-4B (first run downloads the model) ...
Loading WikiText-2 test split (first 512 tokens) ...

=== fp16 baseline ===
  PPL = 10.3866  (511 tokens scored)

=== TurboQuantMSE 4-bit K ===
  PPL = 10.3866  ΔPPL = +0.0000  (+0.00%)

Headline table row:
  model=Qwen/Qwen3.5-4B  method=TurboQuantMSE  K=4  fp16=10.3866  quant=10.3866  ΔPPL=+0.0000  (+0.00%)
"""


def test_parse_headline_row_happy_path() -> None:
    parsed = parse_headline_row(_REALISTIC_HEADLINE)
    assert parsed is not None
    assert parsed["model"] == "Qwen/Qwen3.5-4B"
    assert parsed["method"] == "TurboQuantMSE"
    assert parsed["bits"] == 4
    assert parsed["ppl_fp16"] == pytest.approx(10.3866)
    assert parsed["ppl_quant"] == pytest.approx(10.3866)
    assert parsed["delta_ppl"] == pytest.approx(0.0000)
    assert parsed["delta_pct"] == pytest.approx(0.00)


def test_parse_headline_row_negative_delta() -> None:
    """Some quantization sweeps produce lower PPL than fp16 (noise
    or lucky regularisation); the regex must accept signed numbers."""
    stdout = (
        "Headline table row:\n"
        "  model=Qwen/Qwen3-4B  method=BlockTurboQuantMSE  K=3  "
        "fp16=10.3866  quant=10.3700  ΔPPL=-0.0166  (-0.16%)\n"
    )
    parsed = parse_headline_row(stdout)
    assert parsed is not None
    assert parsed["delta_ppl"] == pytest.approx(-0.0166)
    assert parsed["delta_pct"] == pytest.approx(-0.16)
    assert parsed["bits"] == 3


def test_parse_headline_row_missing_returns_none() -> None:
    stdout = "no PPL line here\nsome other output\n"
    assert parse_headline_row(stdout) is None


def test_parse_headline_row_ignores_non_matching_lines() -> None:
    """Headline line can be preceded by arbitrary output; only the
    exact format must be matched."""
    stdout = (
        "garbage line\n"
        "model=fake bad format\n"
        "Headline table row:\n"
        "  model=foo/bar  method=X  K=8  fp16=12.0000  quant=12.5000 "
        " ΔPPL=+0.5000  (+4.17%)\n"
    )
    parsed = parse_headline_row(stdout)
    assert parsed is not None
    assert parsed["model"] == "foo/bar"
    assert parsed["method"] == "X"
    assert parsed["bits"] == 8
    assert parsed["ppl_fp16"] == pytest.approx(12.0000)
    assert parsed["ppl_quant"] == pytest.approx(12.5000)


# ---------- subprocess orchestration -------------------------------


def _fake_completed_process(
    stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


def test_script_not_found_skips_before_subprocess(tmp_path: Path) -> None:
    """A bad --script path never invokes subprocess — skip cleanly
    so the bench can report 'environmentally blocked' without
    looking like a hard failure."""
    never_called: dict[str, Any] = {"n": 0}

    def spy_run(*args: Any, **kwargs: Any) -> Any:
        never_called["n"] += 1
        raise AssertionError("subprocess must not run for missing script")

    result = run_vqbench_baseline(
        tmp_path / "does-not-exist.py",
        subprocess_runner=spy_run,
    )
    assert result.status == "skipped"
    assert result.reason is not None
    assert result.reason.startswith("script_not_found:")
    assert never_called["n"] == 0


def test_happy_subprocess_parses_headline(tmp_path: Path) -> None:
    script = tmp_path / "fake_repro.py"
    script.write_text("# not actually run\n")

    def fake_run(
        cmd: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert cmd[1] == str(script.resolve())
        return _fake_completed_process(
            stdout=_REALISTIC_HEADLINE, returncode=0
        )

    result = run_vqbench_baseline(script, subprocess_runner=fake_run)
    assert result.status == "ok"
    assert result.reason is None
    assert result.model == "Qwen/Qwen3.5-4B"
    assert result.method == "TurboQuantMSE"
    assert result.bits == 4
    assert result.ppl_fp16 == pytest.approx(10.3866)
    assert result.ppl_quant == pytest.approx(10.3866)
    assert result.delta_ppl == pytest.approx(0.0000)
    assert result.delta_pct == pytest.approx(0.00)
    assert result.wall_s is not None and result.wall_s >= 0
    assert result.returncode == 0
    assert result.stdout_tail is not None


def test_subprocess_nonzero_exit_is_failed(tmp_path: Path) -> None:
    script = tmp_path / "fake_repro.py"
    script.write_text("# not run\n")

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return _fake_completed_process(
            stdout="some trace\nTraceback ...\n", returncode=7
        )

    result = run_vqbench_baseline(script, subprocess_runner=fake_run)
    assert result.status == "failed"
    assert result.reason == "subprocess_exit:7"
    assert result.returncode == 7
    # Numeric fields stay None — no headline to parse.
    assert result.ppl_fp16 is None


def test_subprocess_timeout_is_failed(tmp_path: Path) -> None:
    script = tmp_path / "fake_repro.py"
    script.write_text("# not run\n")

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1.0)

    result = run_vqbench_baseline(
        script, subprocess_runner=fake_run, timeout_s=1.0
    )
    assert result.status == "failed"
    assert result.reason is not None
    assert result.reason.startswith("subprocess_timeout:")


def test_python_executable_missing_is_failed(tmp_path: Path) -> None:
    """Pointing --python-executable at a non-existent path surfaces
    as ``python_executable_missing:*`` rather than a bare OSError."""
    script = tmp_path / "fake_repro.py"
    script.write_text("# not run\n")

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("No such file or directory: /bogus/python")

    result = run_vqbench_baseline(
        script,
        python_executable="/bogus/python",
        subprocess_runner=fake_run,
    )
    assert result.status == "failed"
    assert result.reason is not None
    assert result.reason.startswith("python_executable_missing:")


def test_headline_not_found_is_failed(tmp_path: Path) -> None:
    """Subprocess exited ok but the headline line was missing — a
    parse failure, still a runner-level ``failed`` so callers know
    the PPL is unavailable."""
    script = tmp_path / "fake_repro.py"
    script.write_text("# not run\n")

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return _fake_completed_process(
            stdout="no headline here\nonly junk\n", returncode=0
        )

    result = run_vqbench_baseline(script, subprocess_runner=fake_run)
    assert result.status == "failed"
    assert result.reason == "headline_row_not_found"
    assert result.ppl_fp16 is None
    assert result.stdout_tail is not None


def test_stdout_tail_truncates_long_output(tmp_path: Path) -> None:
    script = tmp_path / "fake_repro.py"
    script.write_text("# not run\n")

    # Construct a 200-line stdout ending with the headline row
    lines = [f"line {i}" for i in range(200)]
    lines.append(
        "Headline table row:\n  model=foo/bar  method=X  K=4  "
        "fp16=1.0000  quant=1.0000  ΔPPL=+0.0000  (+0.00%)"
    )
    stdout = "\n".join(lines) + "\n"

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return _fake_completed_process(stdout=stdout, returncode=0)

    result = run_vqbench_baseline(
        script, subprocess_runner=fake_run, stdout_tail_lines=10
    )
    assert result.status == "ok"
    assert result.stdout_tail is not None
    # Tail should cap at 10 lines.
    assert len(result.stdout_tail.splitlines()) == 10


def test_relative_script_resolves_against_provided_cwd(tmp_path: Path) -> None:
    """Docstring promises ``script`` is relative to ``cwd`` when
    provided. A caller passing cwd=/a + script="b.py" must resolve
    to /a/b.py (not to <process-cwd>/b.py as the earlier code did)."""
    (tmp_path / "nested").mkdir()
    script = tmp_path / "nested" / "fake_repro.py"
    script.write_text("# placeholder\n")

    seen_cmd: dict[str, list[str]] = {}

    def fake_run(
        cmd: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        seen_cmd["cmd"] = list(cmd)
        return _fake_completed_process(
            stdout=_REALISTIC_HEADLINE, returncode=0
        )

    # Pass script as relative to cwd=tmp_path/"nested".
    result = run_vqbench_baseline(
        "fake_repro.py",
        cwd=tmp_path / "nested",
        subprocess_runner=fake_run,
    )
    assert result.status == "ok"
    # The second argv entry is the resolved script path; it must
    # live under nested/ (i.e. we did NOT fall back to the process
    # cwd where "fake_repro.py" almost certainly does not exist).
    assert seen_cmd["cmd"][1] == str(script.resolve())


def test_default_reproduce_script_path_points_to_checked_in_script() -> None:
    """The checked-in vqbench reproduce script must be where we
    claim it is — otherwise the CLI default silently falls back
    to a skipped status on every run."""
    path = default_reproduce_script_path()
    assert path.name == "reproduce_qwen35_4b_headline.py"
    assert path.exists(), (
        f"expected vqbench reproduce script at {path}; did the "
        "vqbench layout move?"
    )


# ---------- opt-in real run ----------------------------------------


_REAL_VQBENCH_FLAG = os.environ.get("SILICA_REAL_VQBENCH_BASELINE") == "1"
_REAL_VQBENCH_PYTHON = os.environ.get("SILICA_VQBENCH_PYTHON")


@pytest.mark.skipif(
    not _REAL_VQBENCH_FLAG or not _REAL_VQBENCH_PYTHON,
    reason=(
        "Real vqbench baseline run is triple-gated: "
        "(1) SILICA_REAL_VQBENCH_BASELINE=1, "
        "(2) SILICA_VQBENCH_PYTHON=/path/to/vqbench/venv/bin/python, "
        "(3) the script's ~8 GB Qwen3.5-4B download + ~60 s runtime. "
        "Not part of default dev runs."
    ),
)
def test_real_reproduce_qwen35_4b_headline_runs() -> None:
    """Opt-in end-to-end: actually spawn the vqbench script via
    the user-supplied Python and verify the PPL headline parses."""
    assert _REAL_VQBENCH_PYTHON is not None  # narrow for mypy
    result = run_vqbench_baseline(
        default_reproduce_script_path(),
        python_executable=_REAL_VQBENCH_PYTHON,
        timeout_s=600.0,
    )
    assert result.status == "ok", (
        f"real vqbench run failed: reason={result.reason!r}, "
        f"stdout_tail={result.stdout_tail!r}"
    )
    assert isinstance(result.ppl_fp16, float) and result.ppl_fp16 > 0
    assert isinstance(result.ppl_quant, float) and result.ppl_quant > 0
    assert result.model is not None
