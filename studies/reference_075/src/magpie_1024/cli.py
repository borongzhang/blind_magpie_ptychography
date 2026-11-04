"""Terminal interface for the single 75% hard-probe parameter sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    from .protocol import METHODS, parameter_grid, run_id
    from .studies import retained_parameter_grid

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=(*METHODS, 'all'), default='all')
    parser.add_argument('--tasks',help='Comma/range historical shard IDs')
    parser.add_argument('--submission-tasks',action='store_true')
    parser.add_argument('--task-id', type=int, choices=range(16), help='Slurm array task, 0 through 15')
    parser.add_argument('--shard-index', type=int, help='Zero-based interleaved parameter shard')
    parser.add_argument('--num-shards', type=int, default=4)
    parser.add_argument('--list-tasks', action='store_true', help='Print the approved historical task assignments without running')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--smoke', action='store_true', help='Separate tiny data and two epochs')
    parser.add_argument('--plot-only', action='store_true', help='Read records; requires no Torch/Pty-Chi')
    parser.add_argument('--seed', type=int, default=21, help='Minibatch-order seed')
    parser.add_argument('--error-every', type=int, default=20)
    parser.add_argument('--print-every', type=int, default=300)
    parser.add_argument('--rerun-failed', action='store_true')
    args = parser.parse_args(argv)

    tasks = []
    for task_id in range(16):
        method, shard = METHODS[task_id // 4], task_id % 4
        pairs = [p for i, p in enumerate(parameter_grid(method))
                 if i % 4 == shard and p in retained_parameter_grid(method)]
        if not pairs:
            continue
        tasks.append(dict(task_id=task_id, method=method, shard_index=shard,
                          num_shards=4, parameter_pairs=pairs,
                          run_ids=[run_id(method, p) for p in pairs]))
    if args.submission_tasks:
        import re
        allowed={task['task_id'] for task in tasks}
        requested=set(allowed)
        if args.tasks is not None:
            if not re.fullmatch(r'[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*',args.tasks):
                parser.error('Tasks must be comma-separated integer IDs or ranges')
            requested=set()
            for token in args.tasks.split(','):
                bounds=[int(value) for value in token.split('-')]
                first,last=(bounds[0],bounds[-1])
                if first>last or last>15:
                    parser.error('Invalid historical shard range')
                requested.update(range(first,last+1))
            if not requested or requested-allowed:
                parser.error(f'Excluded historical shard IDs: {sorted(requested-allowed)}')
        print(','.join(map(str,sorted(requested))))
        return 0
    if args.tasks is not None:
        parser.error('--tasks is supported with --submission-tasks')
    if args.list_tasks:
        print(json.dumps(tasks, indent=2))
        return 0
    if args.task_id is not None:
        if args.method != 'all' or args.shard_index is not None or args.num_shards != 4:
            parser.error('--task-id determines method and shard; do not combine it with other shard settings')
        matches = [task for task in tasks if task['task_id'] == args.task_id]
        if not matches:
            parser.error('This historical task ID contains only excluded pairs')
        task = matches[0]
        args.method, args.shard_index = task['method'], task['shard_index']
        print(f"Task {args.task_id}: {args.method}, shard {args.shard_index + 1}/4, {len(task['parameter_pairs'])} retained pairs", flush=True)

    from .studies import assigned_run_ids, load_sweep, method_lock, run_sweep
    from .plots import save_report

    methods = METHODS if args.method == 'all' else (args.method,)
    failures = False
    for method in methods:
        options = dict(root=args.root.resolve(), smoke=args.smoke, seed=args.seed)
        owned = set(assigned_run_ids(method, args.shard_index, args.num_shards))
        if args.plot_only:
            summary = load_sweep(method, **options)
        else:
            summary = run_sweep(method, device=args.device, error_every=args.error_every,
                                print_every=args.print_every, rerun_failed=args.rerun_failed,
                                shard_index=args.shard_index, num_shards=args.num_shards,
                                **options)
        if not args.plot_only:
            failures |= not summary.loc[summary.run_id.isin(owned), 'status'].eq('complete').all()
        with method_lock(args.root.resolve(), method, smoke=args.smoke):
            summary = load_sweep(method, **options)
            terminal = summary.status.isin(('complete', 'failed')).all()
            if args.plot_only or terminal:
                directory = save_report(method, args.root.resolve(), summary, smoke=args.smoke)
                print(f'Saved {method} tables and figures to {directory}', flush=True)
            else:
                count = int(summary.status.isin(('complete', 'failed')).sum())
                print(f'{method}: {count}/{len(retained_parameter_grid(method))} candidates terminal; final report waits for the other shards.', flush=True)
    return 1 if failures else 0
