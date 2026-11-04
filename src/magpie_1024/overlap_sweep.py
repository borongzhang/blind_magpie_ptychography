"""Fixed-budget overlap sweeps using the unchanged native experiment runner.

The completed 75% sweep is preserved. Each new
task runs an approved parameter pair at 87.5% overlap. Selection
uses final object NMSE after every retained pair completes or fails. Historical
task IDs and per-run paths are preserved; excluded pairs leave gaps.
Planning, progress, and selection do not import Torch or Pty-Chi.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .protocol import (BASE_BATCH, BASE_ETA, METHODS, PTYCHI_COMMIT,
                       case_token, float_token, parameter_grid, run_id)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY = 'overlap_parameter_sweep'
OVERLAPS = (0.875,)
SWEEP_EPOCHS = {0.875: 800}
PATTERN_COUNTS = {0.875: 3249}
ERROR_EVERY = 20
PRINT_EVERY = 300


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retained_parameter_grid(method):
    """Sixteen LSQML pairs and nine PIE pairs, preserving original pair IDs."""
    return [dict(p) for p in parameter_grid(method)
            if method == 'lsqml' or (p['object_alpha'] != .5 and p['probe_alpha'] != .5)]


def retained_pair_indices(method):
    allowed = retained_parameter_grid(method)
    return [i for i, parameters in enumerate(parameter_grid(method)) if parameters in allowed]


def task_by_id(tasks, task_id):
    """Look up a historical ID without treating a filtered plan as a dense array."""
    if isinstance(task_id, bool) or not isinstance(task_id, int):
        raise ValueError('task_id must be an integer historical task ID')
    found = [task for task in tasks if task['task_id'] == task_id]
    if len(found) != 1:
        raise ValueError(f'Task {task_id} is excluded or is not part of this study')
    return found[0]


def parse_task_ids(spec, allowed_ids):
    """Parse a Slurm-style comma/range list and reject excluded IDs."""
    import re
    allowed = set(allowed_ids)
    if not isinstance(spec, str) or not re.fullmatch(r'[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*', spec):
        raise ValueError('Tasks must be comma-separated integer IDs or inclusive ranges')
    result = set()
    for token in spec.split(','):
        bounds = [int(value) for value in token.split('-')]
        first, last = (bounds * 2)[:2] if len(bounds) == 1 else bounds
        if first > last or last - first > 10000:
            raise ValueError(f'Invalid task range: {token}')
        result.update(range(first, last + 1))
    if not result or result - allowed:
        raise ValueError(f'Excluded or unknown task IDs: {sorted(result - allowed)}')
    return sorted(result)


def compatible_runner_identity(saved, current, mutable_keys):
    """Only planner metadata may differ; callers must also verify sealed record bytes."""
    return (isinstance(saved, dict) and set(saved) == set(current)
            and all(saved[key] == value for key, value in current.items() if key not in mutable_keys))


def sealed_source_record(directory, manifest):
    """Authenticate exact historical bytes even when records have been mirrored."""
    directory=Path(directory).resolve()
    def run_suffix(path):
        parts=Path(path).parts
        return parts[parts.index('runs'):] if 'runs' in parts else ()
    suffix=run_suffix(directory/'result.json')
    if not suffix:
        return False
    candidates=[c for selection in manifest.get('selections',[]) for c in selection['candidates']]
    candidates+=manifest.get('baseline_records',[])
    matches=[c for c in candidates if run_suffix(c.get('relative_result_path',c.get('source_result_path','')))==suffix]
    if len(matches)!=1:
        return False
    hashes=matches[0].get('source_files_sha256',{})
    named={Path(name).name:digest for name,digest in hashes.items()}
    if not {'result.json','config.json','history.csv'}<=set(named):
        return False
    return all((directory/name).is_file() and _sha(directory/name)==digest
               for name,digest in named.items())


def task_plan():
    """43 approved 87.5% tasks, using their original IDs within 64..127."""
    tasks = []
    for overlap in OVERLAPS:
        for method_index, method in enumerate(METHODS):
            for pair_index in retained_pair_indices(method):
                parameters = parameter_grid(method)[pair_index]
                tasks.append(dict(task_id=64 + method_index * 16 + pair_index, method=method,
                    overlap_ratio=overlap, parameters=parameters, pair_index=pair_index,
                    epochs=SWEEP_EPOCHS[overlap], batch_size=BASE_BATCH, eta=BASE_ETA,
                    num_patterns=PATTERN_COUNTS[overlap],
                    patterns_processed=SWEEP_EPOCHS[overlap] * PATTERN_COUNTS[overlap],
                    data_seed=42, seed=21, error_every=ERROR_EVERY,
                    print_every=PRINT_EVERY))
    return tasks


def _base(root, smoke=False):
    return Path(root).resolve() / 'results' / ('smoke' if smoke else '')


def _relative(task):
    condition = (f"{case_token(task['overlap_ratio'])}_eta_{float_token(BASE_ETA)}"
                 f'_b{BASE_BATCH}')
    return Path('runs') / task['method'] / condition / run_id(task['method'], task['parameters'])


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _source_identity():
    package = Path(__file__).resolve().parent
    names = ('engine.py', 'overlap.py', 'cli.py', 'data.py', 'protocol.py')
    core = {name: _sha(package / name) for name in names}
    core['selected_parameters.json'] = _sha(PROJECT_ROOT / 'provenance/selected_parameters.json')
    return {'core_runner_sha256': core,
            'sweep_runner_sha256': {'overlap_sweep.py': _sha(__file__),
                                    'sweep.py': _sha(PROJECT_ROOT / 'sweep.py')}}


def plan_manifest():
    return dict(schema_version=1, study=STUDY, scientific=True,
        ptychi_commit=PTYCHI_COMMIT, selection_metric='final object_nmse',
        selection_rule='minimum endpoint object_nmse among completed retained pairs after every method-specific grid is terminal; run_id breaks ties',
        original_075_sweep='preserved, 3000 epochs; not rerun by this plan',
        object_roi='fixed 932 x 932 [46:978,46:978], alignment and scoring on the same ROI',
        source_identity=_source_identity(), tasks=task_plan())


def ensure_manifest(root=PROJECT_ROOT, *, smoke=False):
    directory = _base(root, smoke) / 'sweeps' / STUDY
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'plan.json'
    requested = {**plan_manifest(), 'scientific': not smoke, 'smoke': smoke}
    with (directory / '.plan.lock').open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text()) != requested:
                raise ValueError(f'An incompatible sweep plan exists at {path}; preserve it and use a new project.')
        else:
            _atomic_json(path, requested)
    return path


def load_task(task, root=PROJECT_ROOT, *, smoke=False, audit_complete=False):
    """Load one planned run, rejecting mismatched budgets or scientific inputs."""
    epochs = 2 if smoke else task['epochs']
    directory = _base(root, smoke) / _relative(task)
    row = dict(task, run_id=directory.name, epochs=epochs,
               relative_run_dir=_relative(task).as_posix(), status='missing',
               epochs_completed=0, scientific=not smoke)
    result_path = directory / 'result.json'
    if not result_path.exists():
        return row
    saved = json.loads(result_path.read_text())
    expected = dict(method=task['method'], overlap_ratio=task['overlap_ratio'],
        eta=BASE_ETA, batch_size=BASE_BATCH, epochs=epochs, seed=21,
        scientific=not smoke, run_id=directory.name, **task['parameters'])
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Saved result differs from this sweep plan: {result_path}')
    row.update(saved)
    if row['status'] == 'complete' and row.get('epochs_completed') != epochs:
        raise ValueError(f'Completed result has the wrong endpoint: {result_path}')
    if audit_complete and row['status'] == 'complete':
        config_path = directory / 'config.json'
        config = json.loads(config_path.read_text())
        checks = dict(method=task['method'], overlap_ratio=task['overlap_ratio'],
            eta=BASE_ETA, batch_size=BASE_BATCH, epochs=epochs, seed=21,
            scientific=not smoke, parameters=task['parameters'],
            error_every=ERROR_EVERY, smoke=smoke)
        if any(config.get(key) != value for key, value in checks.items()):
            raise ValueError(f'Saved configuration differs from the sweep plan: {config_path}')
        if (config['native_source']['commit'] != PTYCHI_COMMIT or
                config['runner_source_sha256'] != _source_identity()['core_runner_sha256']):
            raise ValueError(f'Saved runner identity differs from this experiment: {config_path}')
        if not (directory / 'reconstruction.npz').is_file():
            raise ValueError(f'Completed result has no saved fields: {directory}')
        for metric in ('object_nmse', 'probe_nmse', 'phase_rmse', 'online_amplitude_mse',
                       'frozen_amplitude_mse', 'frozen_noiseless_amplitude_mse'):
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid final metric {metric}: {result_path}')
        with (directory / 'history.csv').open(newline='') as stream:
            history = list(csv.DictReader(stream))
        if [int(item['epoch']) for item in history] != list(range(epochs + 1)):
            raise ValueError(f'Incomplete history: {directory}')
        for metric in ('object_nmse', 'probe_nmse', 'phase_rmse'):
            if float(history[-1][metric]) != row[metric]:
                raise ValueError(f'Final history/result mismatch for {metric}: {directory}')
        row['result_sha256'] = _sha(result_path)
        row['config_sha256'] = _sha(config_path)
        row['history_sha256'] = _sha(directory / 'history.csv')
        row['dataset_array_sha256'] = config['dataset_array_sha256']
        row['native_source'] = config['native_source']
    return row


def run_task(task_id, root=PROJECT_ROOT, *, device='cuda', smoke=False, rerun_failed=False):
    """Run one pair using unchanged reconstruction and dataset code."""
    task = task_by_id(task_plan(), task_id)
    existing = load_task(task,root,smoke=smoke,audit_complete=True)
    if existing['status']=='complete':
        print(f'Task {task_id}: reusing verified completed reconstruction',flush=True)
        return existing
    manifest_path = ensure_manifest(root, smoke=smoke)
    directory = _base(root, smoke) / _relative(task)
    print(f"Sweep task {task_id}: {task['method']}, overlap {task['overlap_ratio']:.1%}, "
          f"{task['parameters']}, {2 if smoke else task['epochs']} epochs", flush=True)
    from .engine import run_pair
    row = run_pair(task['method'], task['parameters'], root=root,
        overlap_ratio=task['overlap_ratio'], eta=BASE_ETA, batch_size=BASE_BATCH,
        epochs=task['epochs'], device=device, smoke=smoke, seed=21,
        error_every=ERROR_EVERY, print_every=PRINT_EVERY, rerun_failed=rerun_failed)
    # Kept outside config.json/result.json so native identities and safe reuse
    # remain unchanged. The sweep planner and its exact manifest are still traced.
    _atomic_json(directory / 'sweep_task.json', dict(study=STUDY, task=task,
        scientific=not smoke, smoke=smoke, plan_sha256=_sha(manifest_path),
        plan_relative_path=os.path.relpath(manifest_path, Path(root).resolve()),
        sweep_runner_sha256=_source_identity()['sweep_runner_sha256']))
    if row['status'] != 'complete':
        raise RuntimeError(f"Sweep task {task_id}: {row.get('error', row['status'])}")
    return row


def summarize(root=PROJECT_ROOT, *, smoke=False, write=True):
    """Select an endpoint winner after every compatible pair is terminal."""
    rows = [load_task(task, root, smoke=smoke, audit_complete=True) for task in task_plan()]
    groups = []
    for overlap in OVERLAPS:
        for method in METHODS:
            candidates = [row for row in rows if row['method'] == method
                          and row['overlap_ratio'] == overlap]
            complete = [row for row in candidates if row['status'] == 'complete']
            if complete:
                first = complete[0]
                if any(row['dataset_array_sha256'] != first['dataset_array_sha256']
                       or row['native_source'] != first['native_source'] for row in complete):
                    raise ValueError(f'Candidate data/native identities differ: {method}, {overlap}')
            selected = None
            if complete and all(row['status'] in ('complete', 'failed') for row in candidates):
                winner = min(complete, key=lambda row: (row['object_nmse'], row['run_id']))
                selected = dict(parameters=winner['parameters'], run_id=winner['run_id'],
                    object_nmse=winner['object_nmse'], probe_nmse=winner['probe_nmse'],
                    task_id=winner['task_id'], relative_run_dir=winner['relative_run_dir'],
                    result_sha256=winner['result_sha256'], config_sha256=winner['config_sha256'])
            groups.append(dict(study=STUDY, method=method, overlap_ratio=overlap,
                epochs=2 if smoke else SWEEP_EPOCHS[overlap], scientific=not smoke,
                completed_count=len(complete), failed_count=sum(row['status'] == 'failed' for row in candidates),
                missing_count=sum(row['status'] == 'missing' for row in candidates),
                unfinished_count=sum(row['status'] not in ('complete', 'failed', 'missing') for row in candidates),
                selection_metric='final object_nmse', selection=selected,
                candidates=[{key: row[key] for key in ('task_id', 'run_id', 'status', 'epochs_completed')}
                            | {key: row[key] for key in ('object_nmse', 'probe_nmse', 'result_sha256', 'config_sha256') if key in row}
                            for row in candidates]))
    if write:
        manifest_path = ensure_manifest(root, smoke=smoke)
        directory = manifest_path.parent
        with (directory / '.summary.lock').open('a+') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            for group in groups:
                _atomic_json(directory / case_token(group['overlap_ratio']) / group['method'] / 'selection.json', group)
            _atomic_json(directory / 'summary.json', dict(study=STUDY, scientific=not smoke,
                plan_sha256=_sha(manifest_path), groups=groups))
    return groups


def print_progress(root=PROJECT_ROOT, *, smoke=False):
    rows = [load_task(task, root, smoke=smoke) for task in task_plan()]
    print(f'{"Overlap":>8} {"Method":<12} {"Complete":>8} {"Failed":>7} {"Unfinished":>11} {"Missing":>8}')
    for overlap in OVERLAPS:
        for method in METHODS:
            candidates = [row for row in rows if row['method'] == method and row['overlap_ratio'] == overlap]
            statuses = [row['status'] for row in candidates]
            print(f'{overlap:>7.1%} {method:<12} {statuses.count("complete"):>5}/{len(candidates)} '
                  f'{statuses.count("failed"):>7} '
                  f'{sum(s not in ("complete", "failed", "missing") for s in statuses):>11} '
                  f'{statuses.count("missing"):>8}')
    print(f'\n{sum(row["status"] == "complete" for row in rows)}/{len(rows)} sweep runs complete; 75% results are separate and preserved.')
    for row in rows:
        if row['status'] == 'failed':
            print(f"Task {row['task_id']} FAILED: {row.get('error', 'inspect result.json')}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-id', type=int, choices=[task['task_id'] for task in task_plan()])
    parser.add_argument('--tasks',help='Comma/range historical task IDs')
    parser.add_argument('--submission-tasks',action='store_true',help='Print or validate retained IDs for Slurm')
    parser.add_argument('--method', choices=(*METHODS, 'all'), default='all')
    parser.add_argument('--overlap', type=float, choices=OVERLAPS)
    parser.add_argument('--list-tasks', action='store_true')
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--write-plan', action='store_true')
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--smoke', action='store_true', help='Tiny arrays, two epochs, separate results/smoke')
    parser.add_argument('--rerun-failed', action='store_true')
    args = parser.parse_args(argv)
    if args.task_id is not None and args.tasks is not None:
        parser.error('Use --task-id or --tasks, not both')
    planned_tasks = task_plan()
    try:
        chosen_ids = parse_task_ids(args.tasks, [t['task_id'] for t in planned_tasks]) if args.tasks is not None else [t['task_id'] for t in planned_tasks]
    except ValueError as exc:
        parser.error(str(exc))
    planned_tasks = [t for t in planned_tasks if t['task_id'] in chosen_ids]
    if args.submission_tasks:
        print(','.join(str(t['task_id']) for t in planned_tasks))
        return 0

    tasks = planned_tasks
    if args.task_id is not None:
        if args.method != 'all' or args.overlap is not None:
            parser.error('--task-id determines the method and overlap')
        tasks = [task_by_id(tasks, args.task_id)]
    else:
        tasks = [task for task in tasks if (args.method == 'all' or task['method'] == args.method)
                 and (args.overlap is None or task['overlap_ratio'] == args.overlap)]
    if args.list_tasks:
        print(json.dumps(tasks, indent=2)); return 0
    if args.progress:
        print_progress(args.root, smoke=args.smoke); return 0
    if args.write_plan:
        print(ensure_manifest(args.root, smoke=args.smoke)); return 0
    if args.summarize:
        print(json.dumps(summarize(args.root, smoke=args.smoke), indent=2)); return 0
    failures = False
    for task in tasks:
        try:
            run_task(task['task_id'], args.root, device=args.device, smoke=args.smoke,
                     rerun_failed=args.rerun_failed)
        except Exception as exc:
            print(f"Task {task['task_id']} failed: {type(exc).__name__}: {exc}", flush=True)
            failures = True
    return 1 if failures else 0
