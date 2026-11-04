#!/usr/bin/env python3
"""Run or inspect the retained 87.5% parameter sweep."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from magpie_1024.overlap_sweep import main

if __name__ == '__main__':
    raise SystemExit(main())
