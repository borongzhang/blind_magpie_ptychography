"""Two larger-batch parameter sweeps at the retained 87.5% overlap budget.

The original batch-27 sweep is preserved. Each new task crosses one of
the approved parameter pairs with batch 108 or 432, at eta=.1 and 800 epochs.
Planning and progress do not import Torch or load reconstruction arrays.
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

from .overlap_sweep import (_atomic_json, retained_parameter_grid, retained_pair_indices,
                            task_by_id, parse_task_ids, compatible_runner_identity)
from .protocol import METHODS, PTYCHI_COMMIT, parameter_grid, run_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY = 'batch_0875_parameter_sweep'
MANIFEST = Path('provenance/batch_0875_protocol.json')
BATCHES = (108, 432)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_manifest_path():
    path = Path(os.environ.get('MAGPIE_BATCH_MANIFEST', str(MANIFEST)))
    path = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError('Batch protocol manifest must remain inside this project')
    return path


def _frozen():
    frozen = json.loads(protocol_manifest_path().read_text())
    expected = dict(schema_version=1, study=STUDY, overlap_ratio=.875, eta=.1,
                    batch_sizes=list(BATCHES), baseline_batch_size=27, epochs=800,
                    num_patterns=3249, data_seed=42, seed=21, error_every=20,
                    print_every=300, roi_bounds=[46, 978, 46, 978],
                    object_shape=[1024, 1024], probe_shape=[128, 128])
    if any(frozen.get(k) != v for k, v in expected.items()) or len(frozen['baseline_records']) not in (36,43,64):
        raise ValueError('The frozen batch-sweep protocol differs from the approved experiment.')
    for name, digest in frozen['frozen_core_source_sha256'].items():
        path = (PROJECT_ROOT / 'provenance/selected_parameters.json' if name == 'selected_parameters.json'
                else PROJECT_ROOT / 'src/magpie_1024' / name)
        if _sha(path) != digest:
            raise ValueError(f'The retained scientific source changed: {name}')
    if frozen['native_source']['commit'] != PTYCHI_COMMIT:
        raise ValueError('The baseline native source changed.')
    return frozen


def task_plan(root=PROJECT_ROOT):
    _frozen()
    tasks = []
    for batch_index, batch in enumerate(BATCHES):
        for method_index, method in enumerate(METHODS):
            for pair_index in retained_pair_indices(method):
                parameters = parameter_grid(method)[pair_index]
                tasks.append(dict(task_id=batch_index*64 + method_index*16 + pair_index, method=method, parameters=parameters,
                    pair_index=pair_index, overlap_ratio=.875, eta=.1, batch_size=batch,
                    epochs=800, num_patterns=3249, patterns_processed=3249 * 800,
                    minibatches_per_epoch=math.ceil(3249 / batch),
                    minibatch_updates=800 * math.ceil(3249 / batch),
                    final_minibatch_size=3249 % batch or batch,
                    data_seed=42, seed=21, error_every=20, print_every=300))
    return tasks


def _source_file(relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Baseline source paths must remain inside the project.')
    return PROJECT_ROOT / relative


def verify_baseline(root=PROJECT_ROOT):
    """Verify the method-specific baseline records and their frozen numerical inputs.

    Baseline records belong to this code project's frozen provenance; root
    selects output/data fixtures and never redirects the baseline evidence.
    """
    frozen = _frozen()
    seen = {method: [] for method in METHODS}
    completed = failed = 0
    for record in frozen['baseline_records']:
        if record['parameters'] not in retained_parameter_grid(record['method']):
            continue
        for relative, digest in record['source_files_sha256'].items():
            if _sha(_source_file(relative)) != digest:
                raise ValueError(f'The preserved batch-27 source record changed: {relative}')
        path = _source_file(record['source_result_path'])
        result = json.loads(path.read_text())
        config = json.loads(path.with_name('config.json').read_text())
        expected = dict(method=record['method'], overlap_ratio=.875, eta=.1,
                        epochs=800, batch_size=27, seed=21, scientific=True)
        if any(result.get(k) != v or config.get(k) != v for k, v in expected.items()):
            raise ValueError('Preserved batch-27 settings differ from the approved baseline.')
        if (record['method'] not in METHODS or config['parameters'] != record['parameters']
                or result['run_id'] != record['run_id'] or config['smoke']
                or result['status'] != record['status']
                or config['error_every'] != 20 or config['print_every'] != 300
                or config['dataset_fingerprint'] != frozen['baseline_dataset_fingerprint']
                or config['dataset_array_sha256'] != frozen['baseline_array_sha256']
                or config['native_source'] != frozen['native_source']
                or config['runner_source_sha256'] != frozen['frozen_core_source_sha256']):
            raise ValueError('Preserved batch-27 provenance or parameter grid changed.')
        seen[record['method']].append(record['parameters'])
        if result['status'] == 'complete':
            completed += 1
            with path.with_name('history.csv').open(newline='') as stream:
                history = list(csv.DictReader(stream))
            if result['epochs_completed'] != 800 or [int(r['epoch']) for r in history] != list(range(801)):
                raise ValueError('A completed baseline history is incomplete.')
            for metric in ('object_nmse', 'probe_nmse', 'phase_rmse'):
                value = result[metric]
                if not math.isfinite(value) or value < 0 or float(history[-1][metric]) != value:
                    raise ValueError('Baseline endpoint metrics differ from the saved history.')
        elif result['status'] == 'failed':
            failed += 1
        else:
            raise ValueError('The frozen baseline contains an unfinished candidate.')
    def key(parameters):
        return tuple(sorted(parameters.items()))
    for method in METHODS:
        if (len(seen[method]) != len(retained_parameter_grid(method)) or
                {key(p) for p in seen[method]} != {key(p) for p in retained_parameter_grid(method)}):
            raise ValueError(f'The approved parameter grid changed for {method}.')
    if len(frozen['baseline_records']) == sum(len(retained_parameter_grid(method)) for method in METHODS) and (completed != frozen['baseline_completed_count'] or failed != frozen['baseline_failed_count']):
        raise ValueError('Baseline completion counts changed.')
    return dict(records=sum(len(seen[method]) for method in METHODS), complete=completed, failed=failed,
                baseline_dataset_fingerprint=frozen['baseline_dataset_fingerprint'],
                baseline_array_sha256=frozen['baseline_array_sha256'])


def execution_root(root=PROJECT_ROOT):
    return Path(root).resolve() / 'evaluations/batch_0875_sweep'


def _base(root=PROJECT_ROOT, smoke=False):
    return execution_root(root) / 'results' / ('smoke' if smoke else '')


def _relative(task):
    return (Path('runs') / task['method'] /
            f"overlap_0875_eta_0p1_b{task['batch_size']}" /
            run_id(task['method'], task['parameters']))


def run_directory(task, root=PROJECT_ROOT, *, smoke=False):
    return _base(root, smoke) / _relative(task)


def ensure_plan(root=PROJECT_ROOT, *, smoke=False):
    from .batch_sweep_engine import source_identity
    requested = dict(schema_version=1, study=STUDY, scientific=not smoke, smoke=smoke,
        ptychi_commit=PTYCHI_COMMIT, protocol_manifest_sha256=_sha(protocol_manifest_path()),
        source_identity=source_identity(), starts_from='original initialization',
        baseline='Preserved batch-27 87.5% eta=.1 800-epoch parameter sweep; never rerun here',
        selection='minimum final object_nmse among completed method-specific grids; report failures',
        budget='equal epochs and complete data passes; differing minibatch-update counts',
        tasks=task_plan(root))
    directory = _base(root, smoke) / 'studies' / STUDY
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'plan.json'
    with (directory / '.plan.lock').open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text()) != requested:
                raise ValueError(f'Incompatible existing batch-sweep plan: {path}')
        else:
            _atomic_json(path, requested)
    return path


def prepare_data(root=PROJECT_ROOT, *, smoke=False):
    from .batch_sweep_engine import data_config, load_batch_data, validate_metadata
    from .engine import check_native_source
    baseline = verify_baseline(root)
    native = check_native_source()
    if native != _frozen()['native_source']:
        raise ValueError('Installed native numerical code differs from the preserved baseline.')
    plan = ensure_plan(root, smoke=smoke)
    config = data_config(smoke=smoke)
    dataset = load_batch_data(execution_root(root), config, source_project_root=root)
    check = validate_metadata(config, dataset.metadata)
    record = dict(study=STUDY, scientific=not smoke, smoke=smoke, baseline_verification=baseline,
                  dataset_path=str(dataset.path), metadata=dataset.metadata,
                  invariant_check=check, native_source=native, plan_sha256=_sha(plan))
    _atomic_json(plan.parent / 'prepared_data.json', record)
    print(f"Verified batch baseline: {check['num_patterns']} measurements; batches 108 and 432 share all data arrays.", flush=True)
    return record


def run_task(task_id, root=PROJECT_ROOT, *, device='cuda', smoke=False, rerun_failed=False):
    task = task_by_id(task_plan(root), task_id)
    existing = load_task(task,root,smoke=smoke)
    if existing['status']=='complete':
        print(f'Task {task_id}: reusing verified completed reconstruction',flush=True)
        return existing
    from .batch_sweep_engine import run_pair, data_config, cache_directory, validate_metadata
    baseline = verify_baseline(root)
    plan = ensure_plan(root, smoke=smoke)
    print(f"Batch task {task_id}: {task['method']}, batch={task['batch_size']}, "
          f"{2 if smoke else 800} epochs, {task['parameters']}", flush=True)
    row = run_pair(task['method'], task['parameters'], root=execution_root(root),
        source_project_root=Path(root).resolve(), overlap_ratio=.875, eta=.1,
        batch_size=task['batch_size'], epochs=800, device=device, smoke=smoke, seed=21,
        error_every=20, print_every=300, rerun_failed=rerun_failed)
    directory = run_directory(task, root, smoke=smoke)
    config = json.loads((directory / 'config.json').read_text())
    data_cfg = data_config(smoke=smoke)
    metadata = json.loads((cache_directory(root, data_cfg) / 'metadata.json').read_text())
    check = validate_metadata(data_cfg, metadata)
    if (config['dataset_fingerprint'] != check['dataset_fingerprint']
            or config['dataset_array_sha256'] != check['dataset_array_sha256']
            or config['native_source'] != _frozen()['native_source']):
        raise ValueError('Batch reconstruction inputs or native source differ from the verified baseline.')
    _atomic_json(directory / 'batch_case.json', dict(study=STUDY, task=task,
        protocol_manifest_sha256=_sha(protocol_manifest_path()), plan_sha256=_sha(plan),
        scientific=not smoke, smoke=smoke, starts_from='original initialization',
        dataset_invariant_check=check, baseline_verification=baseline))
    if row['status'] != 'complete':
        raise RuntimeError(f"Task {task_id}: {row.get('error', row['status'])}")
    return row


def _retained_artifact(task):
    matches = [entry for entry in _frozen().get('retained_batch_artifacts', [])
               if entry['method']==task['method'] and entry['batch_size']==task['batch_size']
               and entry['parameters']==task['parameters']]
    if len(matches)>1:
        raise ValueError('Duplicate retained batch artifact proof')
    return matches[0] if matches else None


def _verify_retained_batch(task,directory,config,result,entry):
    from .batch_sweep_engine import source_identity
    settings=('method','overlap_ratio','eta','parameters','epochs','batch_size',
              'data_seed','seed','error_every','print_every')
    if (entry is None or any(entry.get(key,entry.get('source_task',{}).get(key))!=task.get(key) for key in settings)
            or result.get('status') not in ('complete','failed')
            or entry.get('status','complete')!=result['status']):
        return False
    completed=result.get('epochs_completed')
    if (not isinstance(completed,int) or isinstance(completed,bool)
            or not 0<=completed<=task['epochs']):
        raise ValueError('Retained batch record has an invalid completed epoch')
    if result['status']=='complete':
        if completed!=task['epochs']:
            raise ValueError('Retained completed batch record has the wrong endpoint')
    elif (completed!=entry.get('epochs_completed') or not result.get('error')
            or not isinstance(result.get('failure_epoch'),int)
            or not completed<=result['failure_epoch']<=task['epochs']):
        raise ValueError('Retained failed batch record lacks its exact failure endpoint')
    canonical=entry['relative_run_dir']
    expected=(Path('evaluations/batch_0875_sweep/results')/_relative(task)).as_posix()
    if canonical!=expected:
        raise ValueError('Retained batch run path differs from its historical parameters')
    hashes=entry['source_files_sha256']
    required_names=('config.json','result.json','history.csv')
    if result['status']=='complete':
        required_names+=('reconstruction.npz',)
    if not {f'{canonical}/{name}' for name in required_names}<=set(hashes):
        raise ValueError('Retained batch proof omits required artifacts')
    for name,digest in hashes.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe retained batch artifact path')
        path=directory/relative.relative_to(canonical)
        if not path.is_file() or _sha(path)!=digest:
            raise ValueError(f'The sealed batch artifact changed: {path}')
    saved=config['runner_source_sha256']
    if (saved!=entry['runner_source_sha256'] or not compatible_runner_identity(saved,source_identity(),
            {'overlap_sweep.py','batch_sweep.py','batch_0875_protocol.json','batch_sweep_entrypoint.py'})
            or config['native_source']!=entry['native_source']
            or config['dataset_array_sha256']!=entry['dataset_array_sha256']
            or config['dataset_fingerprint']!=entry['dataset_fingerprint']):
        raise ValueError('Retained batch numerical/native/data identity changed')
    with (directory/'history.csv').open(newline='') as stream:
        history=list(csv.DictReader(stream))
    if [int(item['epoch']) for item in history]!=list(range(completed+1)):
        raise ValueError('Retained batch history is incomplete')
    if result['status']=='complete':
        for metric in ('object_nmse','probe_nmse','phase_rmse'):
            value=result.get(metric)
            if (not isinstance(value,(int,float)) or not math.isfinite(value) or value<0
                    or value!=entry['recorded_metrics'][metric] or float(history[-1][metric])!=value):
                raise ValueError('Retained batch endpoint metrics changed')
    case=json.loads((directory/'batch_case.json').read_text())
    if case['task']!=entry['source_task']:
        raise ValueError('Retained batch planner proof changed')
    return True


def load_task(task, root=PROJECT_ROOT, *, smoke=False):
    from .batch_sweep_engine import source_identity
    row = dict(task, epochs=2 if smoke else 800, status='missing', epochs_completed=0,
               run_id=run_id(task['method'], task['parameters']), relative_run_dir=_relative(task).as_posix())
    directory = run_directory(task, root, smoke=smoke)
    path = directory / 'result.json'
    entry = None if smoke else _retained_artifact(task)
    if not path.is_file() and entry is not None and Path(root).resolve()==PROJECT_ROOT:
        mirror = Path(entry['source_mirror_relative_run_dir'])
        if mirror.is_absolute() or '..' in mirror.parts:
            raise ValueError('Unsafe batch mirror path')
        directory = PROJECT_ROOT / mirror
        path = directory / 'result.json'
    if not path.is_file():
        return row
    result = json.loads(path.read_text())
    config = json.loads(path.with_name('config.json').read_text())
    expected = dict(method=task['method'], overlap_ratio=.875, eta=.1, epochs=row['epochs'],
                    batch_size=task['batch_size'], seed=21, scientific=not smoke)
    identity_ok=config['runner_source_sha256']==source_identity()
    reused=False
    if not smoke and entry is not None:
        reused=_verify_retained_batch(task,directory,config,result,entry)
        identity_ok=reused
    if (any(result.get(k) != v or config.get(k) != v for k, v in expected.items())
            or config['parameters'] != task['parameters'] or config['smoke'] != smoke
            or config['error_every'] != 20 or config['print_every'] != 300
            or not identity_ok
            or config['native_source'] != _frozen()['native_source']):
        raise ValueError(f'Saved batch-sweep settings differ from the plan: {path}')
    if not smoke and (config['dataset_fingerprint'] != _frozen()['baseline_dataset_fingerprint']
            or config['dataset_array_sha256'] != _frozen()['baseline_array_sha256']):
        raise ValueError('Saved batch-sweep data identity differs from the retained baseline.')
    row.update(result)
    row['reused_sealed_record']=reused
    if row['status']=='complete' and row.get('epochs_completed')!=row['epochs']:
        raise ValueError(f'Completed batch result has wrong endpoint: {path}')
    if row['status'] != 'complete' and path.with_name('history.csv').is_file():
        with path.with_name('history.csv').open(newline='') as stream:
            history = list(csv.DictReader(stream))
        if history:
            row['epochs_completed'] = int(history[-1]['epoch'])
    return row


def print_progress(root=PROJECT_ROOT, *, smoke=False):
    rows = [load_task(t, root, smoke=smoke) for t in task_plan(root)]
    print(f"{'Task':>4} {'Method':<12} {'Batch':>6} {'Status':<12} {'Epochs':>12} {'Object NMSE':>14}")
    for row in rows:
        metric = f"{row['object_nmse']:.6g}" if 'object_nmse' in row else '-'
        print(f"{row['task_id']:>4} {row['method']:<12} {row['batch_size']:>6} {row['status']:<12} "
              f"{row['epochs_completed']:>4}/{row['epochs']:<7} {metric:>14}")
    print(f"\n{sum(r['status'] == 'complete' for r in rows)}/{len(rows)} new batch-sweep cases complete.")
    print('The original batch-27 sweep is preserved and is not resubmitted.')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-id', type=int, choices=[task['task_id'] for task in task_plan()])
    parser.add_argument('--protocol-manifest',type=Path,help='Read an approved alternative batch manifest')
    parser.add_argument('--tasks',help='Comma/range historical task IDs')
    parser.add_argument('--submission-tasks',action='store_true',help='Print or validate retained IDs for Slurm')
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    for flag in ('list-tasks', 'progress', 'write-plan', 'verify-baseline', 'prepare-data', 'smoke', 'rerun-failed'):
        parser.add_argument('--' + flag, action='store_true')
    args = parser.parse_args(argv)
    if args.protocol_manifest is not None:
        os.environ['MAGPIE_BATCH_MANIFEST']=str(args.protocol_manifest)
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

    if args.list_tasks:
        print(json.dumps(planned_tasks, indent=2)); return 0
    if args.progress:
        print_progress(args.root, smoke=args.smoke); return 0
    if args.write_plan:
        print(ensure_plan(args.root, smoke=args.smoke)); return 0
    if args.verify_baseline:
        print(json.dumps(verify_baseline(args.root), indent=2)); return 0
    if args.prepare_data:
        prepare_data(args.root, smoke=args.smoke); return 0
    if args.task_id is None:
        parser.error('Specify --task-id or a planning/preparation/progress action.')
    run_task(args.task_id, args.root, device=args.device, smoke=args.smoke, rerun_failed=args.rerun_failed)
    return 0
