"""Fresh-process orchestration of unchanged, checksum-checked scientific code."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def _frozen_075(root, *, device, smoke, method):
    frozen = root / "studies/reference_075"
    if not (frozen / "src/magpie_1024/studies.py").is_file():
        raise FileNotFoundError("The frozen 75% scientific source is missing.")
    # This child has only the frozen namespace ahead of site-packages. The
    # active magpie_1024 package must never be reused for the historical sweep.
    program = (
        "import json,sys;from pathlib import Path;"
        "from magpie_1024.studies import run_sweep;"
        "from magpie_1024.protocol import METHODS;"
        "root=Path(sys.argv[1]);options=json.loads(sys.argv[2]);"
        "methods=METHODS if options.pop('method')=='all' else (sys.argv[3],);"
        "[run_sweep(m,root=root,**options) for m in methods]"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(frozen / "src")
    options = dict(device=device, smoke=smoke, method=method, seed=21,
                   error_every=20, print_every=300)
    print(f"75% native outputs: {frozen / 'results'}", flush=True)
    subprocess.run([sys.executable, "-c", program, str(frozen), json.dumps(options), method],
                   cwd=frozen, env=environment, check=True)


def _baseline_data(root, overlap, smoke):
    """Generate exact baseline inputs required by the frozen follow-up checks."""
    from magpie_1024.data import DataConfig, load_or_make_data
    if overlap in (.75, .875):
        load_or_make_data(root, DataConfig(overlap_ratio=overlap, eta=.1, seed=42, smoke=smoke))
    else:
        load_or_make_data(root, DataConfig(overlap_ratio=.875, eta=.1, seed=42, smoke=smoke))
        from magpie_1024.high_overlap_data import HighOverlapDataConfig, load_or_make_data as load_high
        load_high(root / "evaluations/high_overlap_sweep",
                  HighOverlapDataConfig(overlap_ratio=.9375, eta=.1, seed=42, smoke=smoke),
                  source_project_root=root)


def _execute_tasks(study, tasks, root, options):
    failures = 0
    for task in tasks:
        try:
            study.run_task(task["task_id"], root, **options)
        except (FloatingPointError, RuntimeError):
            # A recorded nonfinite candidate must not prevent the remaining
            # sweep pairs from running. Configuration or provenance failures
            # do not have a failed result record and propagate immediately.
            record = study.load_task(task, root, smoke=options["smoke"])
            if record.get("status") != "failed":
                raise
            failures += 1
            print(f"Task {task['task_id']} recorded a failed candidate; continuing the grid.", flush=True)
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--study", choices=("overlap", "noise", "batch", "real"), required=True)
    parser.add_argument("--overlap", type=float, choices=(.75, .875, .9375))
    parser.add_argument("--dataset", choices=("test_pattern", "integrated_circuit"))
    parser.add_argument("--batch-size", type=int, choices=(27, 108, 432))
    parser.add_argument("--method", choices=("all", "rpie", "lsqml", "gm_rpie", "gm_magpie"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    print("Running the original scientific code. The curated index and publication figures are unchanged.", flush=True)
    options = dict(device=args.device, smoke=args.smoke)
    failures = 0
    if args.study == "overlap":
        overlaps = (.75, .875, .9375) if args.overlap is None else (args.overlap,)
        for overlap in overlaps:
            if overlap == .75:
                _frozen_075(root, method=args.method, **options)
                continue
            if overlap == .875:
                from magpie_1024 import overlap_sweep as study
            else:
                from magpie_1024 import high_overlap_sweep as study
                _baseline_data(root, .875, args.smoke)
            tasks = [task for task in study.task_plan() if args.method in ("all", task["method"])]
            failures += _execute_tasks(study, tasks, root, options)
    elif args.study == "noise":
        from magpie_1024 import short_noise as study
        overlaps = (.75, .875, .9375) if args.overlap is None else (args.overlap,)
        for overlap in overlaps:
            _baseline_data(root, overlap, args.smoke)
        tasks = [task for task in study.task_plan(root)
                 if task["overlap_ratio"] in overlaps and args.method in ("all", task["method"])]
        failures += _execute_tasks(study, tasks, root, options)
    elif args.study == "batch":
        _baseline_data(root, .875, args.smoke)
        if args.batch_size == 27:
            from magpie_1024 import overlap_sweep as study
            tasks = study.task_plan()
        else:
            from magpie_1024 import batch_sweep as study
            tasks = study.task_plan(root)
        tasks = [task for task in tasks if args.method in ("all", task["method"])
                 and args.batch_size in (None, task["batch_size"])]
        failures += _execute_tasks(study, tasks, root, options)
    else:
        from magpie_1024 import real as study
        tasks = [task for task in study.task_plan() if args.dataset in (None, task["dataset"])
                 and args.method in ("all", task["method"])]
        failures += _execute_tasks(study, tasks, root, options)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
