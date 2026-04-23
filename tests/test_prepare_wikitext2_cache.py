"""Unit tests for :mod:`scripts.prepare_wikitext2_cache`.

Verifies the WikiText-2 preparer's behaviour without hitting the
real HuggingFace datasets cache or network. ``datasets`` is not an
import-time dependency of the script (the real import lives inside
``main``), so these tests install a stub ``sys.modules["datasets"]``
and assert the generated text file matches the vqbench-compatible
``"\\n\\n".join(row["text"])`` shape.

Invariants pinned:

- Default output path is ``~/.cache/silica/wikitext2-test.txt``
  (matches ``silica.bench.scenarios._WIKITEXT2_DEFAULT_PATH``).
- Text is produced via ``"\\n\\n".join(row["text"] for row in ds)``;
  no strip, no empty-line filter (mirrors vqbench).
- Existing files are NOT rewritten by default; ``--overwrite``
  opts in.
- Write is atomic — a caller-side failure during
  ``_extract_text`` must not corrupt any pre-existing output file.
- Missing ``datasets`` import raises ``SystemExit`` (clean exit
  with install hint), not ``ImportError``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

import scripts.prepare_wikitext2_cache as prep


class _FakeDataset:
    """Fake dataset: iterable of ``{"text": str}`` rows mirroring HF
    datasets' ``Dataset`` object for the three fields the script
    consumes (iteration + ``row["text"]``)."""

    def __init__(self, rows: list[str]) -> None:
        self._rows = [{"text": r} for r in rows]

    def __iter__(self) -> Any:
        return iter(self._rows)


def _install_fake_datasets(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[str],
    *,
    record_calls: list[tuple[Any, ...]] | None = None,
) -> None:
    """Install a fake ``datasets`` module into ``sys.modules`` whose
    ``load_dataset(name, config, split=...)`` returns a fake dataset
    built from ``rows``. Optionally records call arguments into
    ``record_calls`` so tests can assert the script's request
    parameters.
    """

    class _FakeModule:
        pass

    def load_dataset(
        name: str,
        config: str | None = None,
        *,
        split: str = "train",
    ) -> _FakeDataset:
        if record_calls is not None:
            record_calls.append((name, config, split))
        return _FakeDataset(rows)

    fake = _FakeModule()
    fake.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake)


# =============================================================================
# Default path matches silica.bench.scenarios
# =============================================================================


def test_default_out_matches_bench_scenario_path() -> None:
    """Drift-lock: the script's default output path must match the
    ``_WIKITEXT2_DEFAULT_PATH`` constant the bench scenarios point
    at. If one moves, the other must move with it or the preparer
    writes to a file the bench row never reads."""
    from silica.bench.scenarios import _WIKITEXT2_DEFAULT_PATH

    assert str(prep._DEFAULT_OUT) == _WIKITEXT2_DEFAULT_PATH


# =============================================================================
# Write behaviour
# =============================================================================


class TestExtractText:
    def test_joins_rows_with_double_newline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_datasets(monkeypatch, ["alpha", "beta", "gamma"])
        from datasets import load_dataset  # type: ignore[import-not-found]

        text = prep._extract_text(load_dataset)
        assert text == "alpha\n\nbeta\n\ngamma"

    def test_preserves_empty_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """WikiText's article-boundary empty rows must be preserved
        — stripping them here would drift silica's PPL from
        vqbench's REPORT numbers."""
        _install_fake_datasets(
            monkeypatch, ["= Article =", "", " body ", ""]
        )
        from datasets import load_dataset  # type: ignore[import-not-found]

        text = prep._extract_text(load_dataset)
        assert text == "= Article =\n\n\n\n body \n\n"

    def test_requests_correct_dataset_config_split(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded: list[tuple[Any, ...]] = []
        _install_fake_datasets(
            monkeypatch, ["row0"], record_calls=recorded
        )
        from datasets import load_dataset  # type: ignore[import-not-found]

        prep._extract_text(load_dataset)
        assert recorded == [
            ("wikitext", "wikitext-2-raw-v1", "test")
        ]


class TestAtomicWrite:
    def test_round_trip(self, tmp_path: Path) -> None:
        out = tmp_path / "sub" / "out.txt"
        prep._atomic_write_text(out, "hello\n\nworld")
        assert out.read_text(encoding="utf-8") == "hello\n\nworld"

    def test_leaves_no_tempfile_on_success(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.txt"
        prep._atomic_write_text(out, "hello")
        siblings = sorted(
            p.name for p in tmp_path.iterdir() if p.is_file()
        )
        assert siblings == ["out.txt"], (
            f"tempfile leaked alongside the final file: {siblings}"
        )

    def test_interrupt_keeps_existing_file_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pre-existing file must stay untouched when the tempfile
        write itself succeeds but a subsequent replace is forced to
        fail. Simulate by making ``os.replace`` raise; the original
        ``hello`` content must survive."""
        import os

        out = tmp_path / "out.txt"
        out.write_text("hello", encoding="utf-8")

        def _fail_replace(_src: str, _dst: str) -> None:
            raise OSError("simulated interrupt")

        monkeypatch.setattr(os, "replace", _fail_replace)
        with pytest.raises(OSError, match="simulated interrupt"):
            prep._atomic_write_text(out, "goodbye")

        # Original content preserved.
        assert out.read_text(encoding="utf-8") == "hello"


# =============================================================================
# main()
# =============================================================================


class TestMain:
    def test_writes_to_configured_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_datasets(monkeypatch, ["alpha", "beta"])
        out = tmp_path / "wikitext.txt"
        rc = prep.main(["--out", str(out)])
        assert rc == 0
        assert out.read_text(encoding="utf-8") == "alpha\n\nbeta"

    def test_default_args_write_to_default_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default ``--out`` is read from ``prep._DEFAULT_OUT``;
        patch that constant directly for the duration of the test.

        Previously this test reloaded the module under a patched
        ``Path.home()`` to re-evaluate ``_DEFAULT_OUT``; the reloaded
        module was left mutated in ``sys.modules`` after the
        monkeypatch unwound, making later tests in the same process
        see a tmp-path default. Patching the already-imported
        attribute via ``monkeypatch.setattr`` keeps the module state
        intact after the fixture finalizer runs."""
        expected = (
            tmp_path / ".cache" / "silica" / "wikitext2-test.txt"
        )
        monkeypatch.setattr(prep, "_DEFAULT_OUT", expected)
        # ``argparse`` sees the patched default via the module lookup
        # in ``_parse_args``; re-parse to refresh the default for
        # ``args.out``.
        _install_fake_datasets(monkeypatch, ["only-row"])

        rc = prep.main([])
        assert rc == 0
        assert expected.is_file()
        assert expected.read_text(encoding="utf-8") == "only-row"

    def test_existing_file_no_overwrite_is_noop(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        out = tmp_path / "wikitext.txt"
        out.write_text("pre-existing", encoding="utf-8")
        _install_fake_datasets(monkeypatch, ["would-overwrite"])
        rc = prep.main(["--out", str(out)])
        assert rc == 0
        assert out.read_text(encoding="utf-8") == "pre-existing"
        captured = capsys.readouterr()
        assert "already exists" in captured.out

    def test_overwrite_flag_rewrites(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "wikitext.txt"
        out.write_text("old", encoding="utf-8")
        _install_fake_datasets(monkeypatch, ["new-row"])
        rc = prep.main(["--out", str(out), "--overwrite"])
        assert rc == 0
        assert out.read_text(encoding="utf-8") == "new-row"

    def test_missing_datasets_raises_system_exit_with_install_hint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Drop ``datasets`` from sys.modules and block fresh
        imports by inserting a meta path finder that rejects the
        name. ``main()`` must exit via ``SystemExit`` (not
        ``ImportError``) with a message naming the install
        command."""
        import importlib.abc

        monkeypatch.delitem(sys.modules, "datasets", raising=False)

        class _BlockDatasets(importlib.abc.MetaPathFinder):
            def find_spec(
                self,
                fullname: str,
                _path: Any,
                _target: Any = None,
            ) -> None:
                if fullname == "datasets":
                    raise ImportError(
                        "datasets blocked by test fixture"
                    )
                return None

        monkeypatch.setattr(
            sys, "meta_path", [_BlockDatasets(), *sys.meta_path]
        )

        out = tmp_path / "wikitext.txt"
        with pytest.raises(SystemExit) as excinfo:
            prep.main(["--out", str(out)])
        # The SystemExit's ``code`` attribute is the message string
        # (argparse-style) when raised with a string argument.
        assert "datasets" in str(excinfo.value)
        assert "install" in str(excinfo.value).lower()
