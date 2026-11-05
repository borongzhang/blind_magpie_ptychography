"""Command-line entry point for the synthetic examples."""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("overlap", "noise", "batch"), default="overlap")
    parser.add_argument("--mode", choices=("selected", "sweep"), default="selected")
    parser.add_argument("--overlap", type=float, choices=(.75, .875, .9375))
    parser.add_argument("--eta", type=float, choices=(.1, .2, .4))
    parser.add_argument("--batch-size", type=int, choices=(27, 108, 432))
    parser.add_argument("--method", choices=("rpie", "lsqml", "gm_rpie", "gm_magpie"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output", default="runs")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = vars(parser.parse_args())
    args["device"] = args["device"] or ("cpu" if args["smoke"] else "cuda")
    from .experiments import run_experiments
    frame = run_experiments(**args)
    print(frame.to_string(index=False))
    return 1 if "status" in frame and (frame.status == "failed").any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
