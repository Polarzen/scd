"""Compatibility wrapper for the P4-C endpoint-sensitivity command."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.cli import main


if __name__ == "__main__":
    argv = sys.argv[1:]
    raise SystemExit(main(["endpoint-sensitivity", *argv]))
