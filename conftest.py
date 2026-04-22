"""Repo-root conftest — SWIG teardown suppression + Q-010 marker gating.

**Problem.** The ``sentencepiece`` C extension (Qwen / Gemma tokenizer
backend) emits three ``DeprecationWarning``s about ``SwigPyPacked`` /
``SwigPyObject`` / ``swigvarlink`` builtin types missing ``__module__``.
Under ``pytest -W error`` those promote to exceptions inside
sentencepiece's C finalizer; the finalizer has no exception handling,
and the process aborts with SIGSEGV (exit 139) even though every test
assertion passed. The finalizer can fire at three different moments:

1. During a test, when the last reference to a tokenizer-backed object
   drops and CPython's GC runs. At this moment pytest's
   ``catch_warnings_for_item`` is active and has prepended ``-W error``
   to the filter list (via ``apply_warning_filters``); marker-level
   filters would land AFTER that prepend and beat ``error``.
2. Between tests, outside ``catch_warnings`` blocks. ``warnings.filters``
   holds whatever was there before pytest's session started.
3. During Python interpreter teardown, after pytest has fully exited.
   ``warnings.filters`` is whatever module-level state remained.

No single fix covers all three. This conftest combines three layers:

1. **Monkey-patch** ``_warnings.warn`` and ``warnings.warn_explicit`` —
   the Python and C entry points warning emission ultimately funnels
   through — to silently drop the three known SWIG messages. This sits
   above filter-list resolution, surviving every ``catch_warnings``
   block (``catch_warnings`` saves / restores ``warnings.filters`` and
   ``warnings.showwarning``, never function attributes).
2. **``pytest_collection_modifyitems``** adds three
   ``pytest.mark.filterwarnings`` entries to every collected test.
   Marker filters are applied by ``catch_warnings_for_item`` AFTER
   pytest's ``apply_warning_filters`` processes the ``-W error``
   cmdline flag, so our ignores prepend ahead of ``error`` inside each
   test's catch_warnings block and win matching priority.
3. **Module-level ``warnings.filterwarnings``** for the
   outside-catch_warnings teardown window (belt + braces; layer 1
   already covers this but leaving the filter in place keeps
   filter-based diagnostic tools accurate).

Normal ``-W error`` behaviour on all other DeprecationWarnings is
preserved — layer 1 only short-circuits the three known SWIG messages;
everything else delegates to the original function.

**Q-010 timing-test gating.** ``test_q010_ratio_below_threshold_on_five_runs``
is a wall-clock on-device timing measurement, noise-prone under
machine load. It is out of the default ``pytest tests/`` sweep via the
``q010_timing`` marker; ``pytest_collection_modifyitems`` below skips
marked tests unless either opt-in channel is engaged:

- ``SILICA_Q010_TIMING=1`` environment variable.
- ``pytest -m q010_timing`` explicit marker selection.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

# =============================================================================
# SWIG teardown suppression — layered defence
# =============================================================================

_SWIG_IGNORE_MESSAGES = (
    "builtin type SwigPyPacked has no __module__ attribute",
    "builtin type SwigPyObject has no __module__ attribute",
    "builtin type swigvarlink has no __module__ attribute",
)


def _is_swig_message(message: Any) -> bool:
    try:
        if isinstance(message, BaseException):
            text = str(message.args[0]) if message.args else str(message)
        elif isinstance(message, str):
            text = message
        else:
            text = str(message)
        return any(m in text for m in _SWIG_IGNORE_MESSAGES)
    except Exception:
        return False


# ----- layer 1a: monkey-patch warnings.warn_explicit ------------------------

_original_warn_explicit = warnings.warn_explicit


def _swig_filtered_warn_explicit(  # type: ignore[no-untyped-def]
    message,
    category,
    filename,
    lineno,
    module=None,
    registry=None,
    module_globals=None,
    source=None,
):
    """Replacement for ``warnings.warn_explicit`` that silently drops the
    three sentencepiece SWIG-teardown DeprecationWarnings before any
    filter action can fire. All other warnings delegate to the original.
    """
    if _is_swig_message(message):
        return None
    return _original_warn_explicit(
        message,
        category,
        filename,
        lineno,
        module,
        registry,
        module_globals,
        source,
    )


warnings.warn_explicit = _swig_filtered_warn_explicit  # type: ignore[assignment]


# ----- layer 1b: monkey-patch _warnings.warn + warnings.warn ----------------
#
# warnings.warn is a direct reference to _warnings.warn (C implementation)
# that bypasses Python-level warn_explicit. Patching both the C-module and
# the warnings-module attribute catches both call paths.

import _warnings  # noqa: E402  (deliberately after warnings setup)

_original_c_warn = _warnings.warn


def _swig_filtered_c_warn(  # type: ignore[no-untyped-def]
    message, category=None, stacklevel=1, source=None
):
    if _is_swig_message(message):
        return None
    return _original_c_warn(message, category, stacklevel, source)


_warnings.warn = _swig_filtered_c_warn  # type: ignore[assignment]
warnings.warn = _swig_filtered_c_warn  # type: ignore[assignment]


# ----- layer 3: module-level filterwarnings (teardown residual) -------------

for _msg in _SWIG_IGNORE_MESSAGES:
    warnings.filterwarnings(
        "ignore", message=_msg, category=DeprecationWarning
    )


# =============================================================================
# Q-010 marker-based gating
# =============================================================================


def _q010_env_opted_in() -> bool:
    return os.environ.get("SILICA_Q010_TIMING", "").strip() not in (
        "",
        "0",
        "false",
        "False",
    )


def _q010_marker_selected(config: Any) -> bool:
    """Return True iff ``pytest -m`` expression positively selects
    ``q010_timing`` (distinguishes ``-m q010_timing`` from
    ``-m "not q010_timing"``)."""
    expr = config.getoption("-m", default="") or ""
    if not expr:
        return False
    if "q010_timing" not in expr:
        return False
    return "not q010_timing" not in expr


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Two responsibilities:

    1. Skip ``q010_timing``-marked tests in the default sweep; honour the
       ``SILICA_Q010_TIMING`` env var and ``-m q010_timing`` as
       independent opt-ins.
    2. Add three ``filterwarnings`` markers to every collected test — the
       layer-2 defence against ``-W error``'s SIGSEGV-at-teardown on
       sentencepiece's SWIG DeprecationWarnings. Marker filters apply
       AFTER pytest's ``apply_warning_filters`` processes ``-W error``,
       so our ignores prepend ahead of ``error`` inside each test's
       ``catch_warnings_for_item`` block.
    """
    import pytest

    # (1) Q-010 gating.
    skip_marker = pytest.mark.skip(
        reason=(
            "Q-010 wall-clock timing gate is opted-out by default "
            "(on-device timing measurement, noise-prone). Set "
            "SILICA_Q010_TIMING=1 or invoke via ``pytest -m q010_timing``."
        )
    )
    q010_active = _q010_env_opted_in() or _q010_marker_selected(config)

    # (2) SWIG filter markers — construct once, apply to every test.
    swig_filter_markers = [
        pytest.mark.filterwarnings(
            f"ignore:{m}:DeprecationWarning"
        )
        for m in _SWIG_IGNORE_MESSAGES
    ]

    for item in items:
        # Q-010 skip if not opted in.
        if not q010_active and "q010_timing" in item.keywords:
            item.add_marker(skip_marker)
        # SWIG ignores for every test.
        for marker in swig_filter_markers:
            item.add_marker(marker)
