"""Run the project's existing 5-minute feature builder with process workers.

All builder options are passed through unchanged. Use ``--repo PATH`` on the
builder command line to select the project checkout that contains the inputs;
the adapter imports its core builder from this clone's ``scripts`` package.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.build_full_5min as builder


def process_executor(*, max_workers, thread_name_prefix=None):
    return ProcessPoolExecutor(max_workers=max_workers)


def main(argv: Sequence[str] | None = None) -> int:
    builder.ThreadPoolExecutor = process_executor
    return builder.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
