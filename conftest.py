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

No single fix covers all three. This conftest combines five layers:

1. **Monkey-patch** ``_warnings.warn`` and ``warnings.warn_explicit`` —
   the Python and C entry points warning emission ultimately funnels
   through — to silently drop the three known SWIG messages. This sits
   above filter-list resolution, surviving every ``catch_warnings``
   block (``catch_warnings`` saves / restores ``warnings.filters`` and
   ``warnings.showwarning``, never function attributes). Note: C-level
   ``PyType_Ready`` uses internal ``_warnings.c::warn_explicit`` and
   bypasses these Python-level patches, so layers 4 + 5 cover that
   path.
2. **``pytest_collection_modifyitems``** adds three
   ``pytest.mark.filterwarnings`` entries to every collected test.
   Marker filters are applied by ``catch_warnings_for_item`` AFTER
   pytest's ``apply_warning_filters`` processes the ``-W error``
   cmdline flag, so our ignores prepend ahead of ``error`` inside each
   test's catch_warnings block and win matching priority.
3. **Module-level ``warnings.filterwarnings``** for the
   outside-catch_warnings teardown window (belt + braces).
4. **Pre-fire PyType_Ready via a top-of-conftest
   ``import sentencepiece``.** Sentencepiece's SWIG types emit the
   three warnings exactly once in the process — at the first
   ``PyType_Ready`` call, i.e. the initial import. Forcing that
   import here (AFTER the module-level ``filterwarnings`` calls in
   layer 3, BEFORE pytest enters any ``catch_warnings_for_item``
   context) makes the single emission happen while our ``ignore``
   filters are at position 0 of ``warnings.filters``. Subsequent
   pytest imports hit the ``sys.modules`` cache, so no filter state
   pytest later establishes (including a collection-phase
   ``catch_warnings`` that prepends ``-W error`` ahead of our ini
   filters) can cause re-emission.
5. **``atexit`` re-prepend** of the three ignore filters. At
   interpreter shutdown the ``<sys>:0: swigvarlink`` warning fires
   during SWIG's C-module finalizer, **after** pytest has
   unconfigured its session ``catch_warnings``. Whatever filter
   state pytest leaves behind is what matches at that instant. If
   ``-W error`` survived into teardown (version-sensitive
   ``catch_warnings`` restore, Python-level ``-W error`` passed via
   ``sys.warnoptions``, etc.), the C-level finalizer's raised
   exception aborts the process with SIGSEGV / exit 139 even though
   every test passed. Re-prepending our ignore filters via
   ``atexit`` guarantees they are at position 0 of
   ``warnings.filters`` when the finalizer runs.

Normal ``-W error`` behaviour on all other DeprecationWarnings is
preserved — layer 1 only short-circuits the three known SWIG messages;
everything else delegates to the original function. Layers 3-5 all
use message-specific regexes, so only the three SWIG texts are
affected.

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


# ----- layer 4: pre-fire sentencepiece PyType_Ready -------------------------
#
# Force the single process-wide SWIG type-initialization window to
# land NOW, while layers 1-3 are all active and pytest has not yet
# entered a ``catch_warnings_for_item`` context. After this import
# ``sentencepiece`` is in ``sys.modules`` and subsequent imports
# from test modules are cache hits — ``PyType_Ready`` does not
# re-fire, so the three DeprecationWarnings simply cannot be emitted
# again in this process. If the package is not installed, there is
# nothing to pre-fire and nothing to suppress.

try:
    import sentencepiece as _sentencepiece_prefire  # noqa: F401
except ImportError:
    pass


# ----- layer 5: atexit re-prepend (interpreter teardown) --------------------
#
# Re-register the three ignore filters at ``atexit`` so that whatever
# ``warnings.filters`` state pytest leaves behind on exit, our three
# ignores are at position 0 when the SWIG C-module finalizer fires
# the ``<sys>:0: swigvarlink`` warning during ``Py_Finalize``.
# ``atexit`` handlers run before module finalizers, so by the time
# the SWIG finalizer emits the warning our ignores are already in
# place.

import atexit  # noqa: E402


def _reinstall_swig_filters_at_exit() -> None:
    for msg in _SWIG_IGNORE_MESSAGES:
        warnings.filterwarnings(
            "ignore", message=msg, category=DeprecationWarning
        )


atexit.register(_reinstall_swig_filters_at_exit)


# =============================================================================
# On-device timing marker gates (Q-010 + prefix-hit-decode)
# =============================================================================
#
# Timing-sensitive tests are opted out of the default sweep via per-
# marker gates. Each entry in ``_TIMING_GATES`` declares a marker name,
# its opt-in env var, and a skip reason. Adding a third timing marker
# is a one-line addition to the table rather than a parallel set of
# helpers.

_TIMING_GATES: tuple[tuple[str, str, str], ...] = (
    (
        "q010_timing",
        "SILICA_Q010_TIMING",
        "Q-010 wall-clock timing gate is opted-out by default "
        "(on-device timing measurement, noise-prone). Set "
        "SILICA_Q010_TIMING=1 or invoke via ``pytest -m q010_timing``.",
    ),
    (
        "prefix_hit_decode_timing",
        "SILICA_PREFIX_HIT_DECODE_TIMING",
        "P-5-A.3c prefix-hit decode-speed gate is opted-out by default "
        "(on-device wall-clock timing, noise-prone). Set "
        "SILICA_PREFIX_HIT_DECODE_TIMING=1 or invoke via "
        "``pytest -m prefix_hit_decode_timing``.",
    ),
)


def _env_opted_in(env_var: str) -> bool:
    return os.environ.get(env_var, "").strip() not in (
        "",
        "0",
        "false",
        "False",
    )


def _marker_selected(config: Any, marker: str) -> bool:
    """Return True iff ``pytest -m`` expression positively selects the
    given marker (distinguishes ``-m <marker>`` from
    ``-m "not <marker>"``)."""
    expr = config.getoption("-m", default="") or ""
    if not expr:
        return False
    if marker not in expr:
        return False
    return f"not {marker}" not in expr


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Two responsibilities:

    1. Skip timing-marker-gated tests in the default sweep; honour each
       gate's env var and ``-m <marker>`` expression as independent
       opt-ins. See ``_TIMING_GATES``.
    2. Add three ``filterwarnings`` markers to every collected test — the
       layer-2 defence against ``-W error``'s SIGSEGV-at-teardown on
       sentencepiece's SWIG DeprecationWarnings. Marker filters apply
       AFTER pytest's ``apply_warning_filters`` processes ``-W error``,
       so our ignores prepend ahead of ``error`` inside each test's
       ``catch_warnings_for_item`` block.
    """
    import pytest

    # (1) Timing-gate resolution. Precompute per-gate skip markers +
    # opt-in state so the per-item loop is O(gates × items) not O(...
    # env reads × items).
    gate_plan: list[tuple[str, bool, Any]] = []
    for marker_name, env_var, reason in _TIMING_GATES:
        opted_in = _env_opted_in(env_var) or _marker_selected(
            config, marker_name
        )
        skip_marker = pytest.mark.skip(reason=reason)
        gate_plan.append((marker_name, opted_in, skip_marker))

    # (2) SWIG filter markers — construct once, apply to every test.
    swig_filter_markers = [
        pytest.mark.filterwarnings(
            f"ignore:{m}:DeprecationWarning"
        )
        for m in _SWIG_IGNORE_MESSAGES
    ]

    for item in items:
        # Timing gates: skip marked tests unless their gate is active.
        for marker_name, opted_in, skip_marker in gate_plan:
            if not opted_in and marker_name in item.keywords:
                item.add_marker(skip_marker)
        # SWIG ignores for every test.
        for marker in swig_filter_markers:
            item.add_marker(marker)
