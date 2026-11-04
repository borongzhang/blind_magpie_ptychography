#!/usr/bin/env python3
"""Run or inspect the 87.5% overlap batch-size parameter sweeps."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from magpie_1024.batch_sweep import main

if __name__ == '__main__':
    raise SystemExit(main())
