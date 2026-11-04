#!/usr/bin/env python3
"""Run the 75% hard-probe sweep, or analyze its saved results."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from magpie_1024.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
