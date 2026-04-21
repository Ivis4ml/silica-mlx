"""silica.chat — minimal multi-turn chat over silica.engine.Engine.

Not a production server (that is P-8). This module provides the
thin session abstraction needed to drive a REPL / notebook chatbot
against the existing single-request engine while surfacing per-
turn TTFT / prefill tok/s / decode tok/s / KV / peak memory so
every chat turn doubles as an optimisation benchmark.
"""

from silica.chat.session import ChatSession, TurnMetrics

__all__ = ["ChatSession", "TurnMetrics"]
