"""Inspect Cycle 2 by default. Add --execute only on the intended T4 session."""

import time

STARTED = time.monotonic()

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm2.research.cycle2 import main


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    main(started=STARTED)
