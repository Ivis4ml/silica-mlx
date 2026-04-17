"""``python -m silica`` — dispatches to ``silica.server.cli.main``."""

from __future__ import annotations

import sys

from silica.server.cli import main

if __name__ == "__main__":
    sys.exit(main())
