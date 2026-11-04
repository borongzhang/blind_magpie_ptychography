"""Small command line interface for saved results and optional reruns."""
from __future__ import annotations

import argparse
import json

from .api import METHODS, STUDIES, rerun_plan, run_study, select_cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="List saved endpoint records without a GPU")
    rerun = sub.add_parser("rerun", help="Run the original scientific code; archived results remain unchanged")
    for command in (inspect, rerun):
        command.add_argument("--study", required=True, choices=STUDIES)
        command.add_argument("--overlap", type=float, choices=(.75, .875, .9375))
        command.add_argument("--dataset", choices=("test_pattern", "integrated_circuit"))
        command.add_argument("--method", choices=(*METHODS, "all"), default="all")
        command.add_argument("--batch-size", type=int, choices=(27, 108, 432))
        command.add_argument("--root")
    inspect.add_argument("--selected", action="store_true")
    inspect.add_argument("--eta", type=float)
    rerun.add_argument("--smoke", action="store_true", help="Tiny separate-data run; defaults to CPU")
    rerun.add_argument("--device", choices=("cpu", "cuda"))
    rerun.add_argument("--dry-run", action="store_true", help="Print the command without executing a solver")
    args = parser.parse_args(argv)
    options = dict(overlap=args.overlap, dataset=args.dataset, method=args.method,
                   batch_size=args.batch_size, root=args.root)
    if args.command == "inspect":
        rows = select_cases(args.study, eta=args.eta, selected=True if args.selected else None, **options)
        print(json.dumps(rows, indent=2, allow_nan=False))
    else:
        options.update(smoke=args.smoke, device=args.device)
        if args.dry_run:
            print(json.dumps(rerun_plan(args.study, **options), indent=2))
        else:
            run_study(args.study, **options)


if __name__ == "__main__":
    main()
