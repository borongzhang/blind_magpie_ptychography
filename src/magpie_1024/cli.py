"""Run the twelve fixed-pair overlap cases, or inspect saved results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(argv=None):
    from .overlap import PROJECT_ROOT, OVERLAPS, task_plan, run_case, load_case, report_case
    from .protocol import METHODS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-id', type=int, choices=range(12))
    parser.add_argument('--method', choices=(*METHODS, 'all'), default='all')
    parser.add_argument('--overlap', type=float, choices=OVERLAPS)
    parser.add_argument('--list-tasks', action='store_true')
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--smoke', action='store_true', help='Separate 64/16 arrays, two epochs')
    parser.add_argument('--plot-only', action='store_true', help='Saved arrays only; no reconstruction')
    parser.add_argument('--rerun-failed', action='store_true')
    args = parser.parse_args(argv)
    tasks = task_plan()
    if args.list_tasks:
        print(json.dumps(tasks, indent=2)); return 0
    if args.task_id is not None:
        if args.method != 'all' or args.overlap is not None:
            parser.error('--task-id determines the method and overlap')
        tasks = [tasks[args.task_id]]
    else:
        tasks = [t for t in tasks if (args.method == 'all' or t['method'] == args.method)
                 and (args.overlap is None or t['overlap_ratio'] == args.overlap)]
    if args.progress:
        complete = 0
        print(f'{"Task":>4}  {"Method":<12} {"Overlap":>8} {"Status":<12} {"Epochs":>13} {"Object NMSE":>13}')
        for task in tasks:
            row = load_case(task['method'], task['overlap_ratio'], args.root, smoke=args.smoke)
            score = f"{row['object_nmse']:.6g}" if row.get('object_nmse') is not None else '-'
            print(f"{task['task_id']:>4}  {task['method']:<12} {task['overlap_ratio']:>7.1%} "
                  f"{row['status']:<12} {row['epochs_completed']:>6}/{row['epochs']:<6} {score:>13}")
            complete += row['status'] == 'complete' and row['epochs_completed'] == row['epochs']
        print(f'\n{complete}/{len(tasks)} cases complete. Unfinished includes interrupted or timed-out runs.')
        return 0
    failures = False
    for task in tasks:
        print(f"Task {task['task_id']}: {task['method']}, overlap {task['overlap_ratio']:.1%}, "
              f"{2 if args.smoke else task['epochs']} epochs", flush=True)
        if args.plot_only:
            directory = report_case(task['method'], task['overlap_ratio'], args.root, smoke=args.smoke)
            print(f'Saved report: {directory}', flush=True)
        else:
            try:
                run_case(task['method'], task['overlap_ratio'], args.root, device=args.device,
                         smoke=args.smoke, rerun_failed=args.rerun_failed)
            except Exception as exc:
                print(f"Task {task['task_id']} failed: {type(exc).__name__}: {exc}", flush=True)
                failures = True
    return 1 if failures else 0
