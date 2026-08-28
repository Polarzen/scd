#!/usr/bin/env python3
"""Compatibility entry point for the bounded full MUSIC 5-minute build."""

from __future__ import annotations

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.build_full_5min import build, main  # noqa: E402,F401


if __name__ == "__main__":
    raise SystemExit(main())
