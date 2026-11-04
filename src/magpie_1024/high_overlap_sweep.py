"""A 200-epoch, 43-task parameter sweep at 93.75% nominal overlap.

The original experiment sources and cached datasets are immutable. This
extension has its own data cache, source identity, plans, and outputs.
Planning and progress do not import Torch or load measurement arrays.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path

from .overlap_sweep import (_atomic_json, retained_pair_indices, task_by_id,
                            compatible_runner_identity, sealed_source_record, parse_task_ids)
from .protocol import METHODS, PTYCHI_COMMIT, parameter_grid, run_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY = 'overlap_09375_parameter_sweep'
OVERLAP = 0.9375
EPOCHS = 200
BATCH = 27
ETA = 0.1
PATTERNS = 12769
ERROR_EVERY = 20
PRINT_EVERY = 300


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execution_root(root=PROJECT_ROOT):
    return Path(root).resolve() / 'evaluations/high_overlap_sweep'


def _base(root, smoke=False):
    return execution_root(root) / 'results' / ('smoke' if smoke else '')


def task_plan():
    return [dict(task_id=i*16+j, method=method, parameters=dict(parameters),
                 pair_index=j, overlap_ratio=OVERLAP, eta=ETA, batch_size=BATCH,
                 epochs=EPOCHS, num_patterns=PATTERNS, data_seed=42, seed=21,
                 error_every=ERROR_EVERY, print_every=PRINT_EVERY,
                 patterns_processed=PATTERNS*EPOCHS)
            for i, method in enumerate(METHODS)
            for j in retained_pair_indices(method)
            for parameters in [parameter_grid(method)[j]]]


def _relative(task):
    return (Path('runs') / task['method'] / 'overlap_09375_eta_0p1_b27' /
            run_id(task['method'], task['parameters']))


def plan_manifest(*, smoke=False):
    from .high_overlap_engine import source_identity
    return dict(schema_version=1, study=STUDY, scientific=not smoke, smoke=smoke,
        ptychi_commit=PTYCHI_COMMIT, source_identity=source_identity(),
        selection_metric='final object_nmse',
        selection_rule='minimum endpoint object_nmse among completed pairs after the complete method-specific grid is terminal; run_id breaks ties',
        scan_rule='step 8 over 0..896; preserve 87.5% sites; new sites have integer perturbations within +/-3',
        measurement_rule='copy shared 87.5% measurements exactly; independent recorded RNG stream for added sites',
        object_roi='fixed 932 x 932 [46:978,46:978]; same alignment and scoring as existing studies',
        earlier_studies='unchanged; 62.5% excluded from final comparisons',
        tasks=task_plan())


def ensure_manifest(root=PROJECT_ROOT, *, smoke=False):
    directory=_base(root,smoke)/'sweeps'/STUDY
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'plan.json'
    requested=plan_manifest(smoke=smoke)
    with (directory/'.plan.lock').open('a+') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text())!=requested:
                raise ValueError(f'Incompatible high-overlap plan; preserve existing outputs: {path}')
        else:
            _atomic_json(path,requested)
    return path


def prepare_data(root=PROJECT_ROOT, *, smoke=False):
    from .high_overlap_data import HighOverlapDataConfig, load_or_make_data
    ensure_manifest(root,smoke=smoke)
    data=load_or_make_data(execution_root(root),
        HighOverlapDataConfig(overlap_ratio=OVERLAP,eta=ETA,seed=42,smoke=smoke),
        source_project_root=Path(root).resolve())
    if not smoke and (len(data.intensities)!=PATTERNS or data.roi_mask.sum()!=932**2):
        raise ValueError('Prepared data have the wrong position count or common ROI.')
    payload=dict(study=STUDY,scientific=not smoke,smoke=smoke,
                 fingerprint=data.fingerprint,array_sha256=data.metadata['array_sha256'],
                 num_patterns=len(data.intensities),dataset_path=str(data.path),
                 metadata=data.metadata)
    _atomic_json(_base(root,smoke)/'sweeps'/STUDY/'prepared_data.json',payload)
    print(f"Verified {'diagnostic' if smoke else 'scientific'} data: "
          f"{len(data.intensities)} positions; fingerprint {data.fingerprint}",flush=True)
    return payload


def load_task(task, root=PROJECT_ROOT, *, smoke=False, audit_complete=False):
    epochs=2 if smoke else EPOCHS
    directory=_base(root,smoke)/_relative(task)
    row=dict(task,run_id=directory.name,epochs=epochs,scientific=not smoke,
             relative_run_dir=_relative(task).as_posix(),status='missing',epochs_completed=0)
    result_path=directory/'result.json'
    if not result_path.is_file():
        return row
    saved=json.loads(result_path.read_text())
    expected=dict(method=task['method'],overlap_ratio=OVERLAP,eta=ETA,
        batch_size=BATCH,epochs=epochs,seed=21,scientific=not smoke,
        run_id=directory.name,**task['parameters'])
    if any(saved.get(k)!=v for k,v in expected.items()):
        raise ValueError(f'Incompatible high-overlap result: {result_path}')
    row.update(saved)
    if row['status']=='complete' and row.get('epochs_completed')!=epochs:
        raise ValueError(f'Wrong completed epoch: {result_path}')
    history_path=directory/'history.csv'
    if row['status']!='complete' and history_path.is_file():
        with history_path.open(newline='') as stream:
            recorded=list(csv.DictReader(stream))
        if recorded:
            row['epochs_completed']=int(recorded[-1]['epoch'])
    if audit_complete and row['status']=='complete':
        from .high_overlap_engine import source_identity
        config_path=directory/'config.json'
        config=json.loads(config_path.read_text())
        expected_config=dict(method=task['method'],parameters=task['parameters'],
            overlap_ratio=OVERLAP,eta=ETA,batch_size=BATCH,epochs=epochs,seed=21,
            scientific=not smoke,smoke=smoke,error_every=ERROR_EVERY)
        current_identity = source_identity()
        saved_identity = config['runner_source_sha256']
        identity_ok = saved_identity == current_identity
        if not identity_ok and compatible_runner_identity(saved_identity, current_identity, {'high_overlap_sweep.py', 'high_overlap.py'}):
            from .short_noise import _frozen
            identity_ok = sealed_source_record(directory, _frozen())
        if (any(config.get(k)!=v for k,v in expected_config.items())
            or config['native_source']['commit']!=PTYCHI_COMMIT
            or not identity_ok):
            raise ValueError(f'Incompatible high-overlap configuration: {config_path}')
        if not (directory/'reconstruction.npz').is_file():
            raise ValueError(f'Missing final reconstruction: {directory}')
        with history_path.open(newline='') as stream:
            history=list(csv.DictReader(stream))
        if [int(r['epoch']) for r in history]!=list(range(epochs+1)):
            raise ValueError(f'Incomplete high-overlap history: {directory}')
        metrics=('object_nmse','probe_nmse','phase_rmse')
        for item in history:
            e=int(item['epoch'])
            for metric in metrics:
                if e%ERROR_EVERY==0 or e==epochs:
                    value=float(item[metric])
                    if not math.isfinite(value) or value<0:
                        raise ValueError(f'Invalid checkpoint metric: {directory}')
                elif item[metric]!='':
                    raise ValueError(f'Unexpected scoring checkpoint: {directory}')
        for metric in (*metrics,'online_amplitude_mse','frozen_amplitude_mse',
                       'frozen_noiseless_amplitude_mse'):
            value=row.get(metric)
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError(f'Invalid final metric {metric}: {directory}')
        if any(float(history[-1][m])!=row[m] for m in metrics):
            raise ValueError(f'History endpoint mismatch: {directory}')
        row.update(result_sha256=_sha(result_path),config_sha256=_sha(config_path),
                   history_sha256=_sha(history_path),
                   dataset_array_sha256=config['dataset_array_sha256'],
                   dataset_fingerprint=config['dataset_fingerprint'],native_source=config['native_source'])
    return row


def run_task(task_id, root=PROJECT_ROOT, *, device='cuda',smoke=False,rerun_failed=False):
    task=task_by_id(task_plan(), task_id)
    existing=load_task(task,root,smoke=smoke,audit_complete=True)
    if existing['status']=='complete':
        print(f'Task {task_id}: reusing verified completed reconstruction',flush=True)
        return existing
    from .high_overlap_engine import run_pair,source_identity
    path=ensure_manifest(root,smoke=smoke)
    directory=_base(root,smoke)/_relative(task)
    print(f"Task {task_id}: {task['method']}, 93.75% overlap, {task['parameters']}, "
          f"{2 if smoke else EPOCHS} epochs",flush=True)
    row=run_pair(task['method'],task['parameters'],root=execution_root(root),
        source_project_root=Path(root).resolve(),overlap_ratio=OVERLAP,eta=ETA,
        batch_size=BATCH,epochs=EPOCHS,device=device,smoke=smoke,seed=21,
        error_every=ERROR_EVERY,print_every=PRINT_EVERY,rerun_failed=rerun_failed)
    _atomic_json(directory/'sweep_task.json',dict(study=STUDY,task=task,
        scientific=not smoke,smoke=smoke,plan_sha256=_sha(path),
        runner_source_sha256=source_identity()))
    if row['status']!='complete':
        raise RuntimeError(f"Task {task_id}: {row.get('error',row['status'])}")
    load_task(task,root,smoke=smoke,audit_complete=True)
    return row


def summarize(root=PROJECT_ROOT, *, smoke=False,write=True):
    rows=[load_task(t,root,smoke=smoke,audit_complete=True) for t in task_plan()]
    groups=[]
    for method in METHODS:
        candidates=[r for r in rows if r['method']==method]
        complete=[r for r in candidates if r['status']=='complete']
        if complete and any(r['dataset_array_sha256']!=complete[0]['dataset_array_sha256']
                            or r['native_source']!=complete[0]['native_source'] for r in complete):
            raise ValueError(f'Mismatched candidate data or native code: {method}')
        selection=None
        if complete and all(r['status'] in ('complete','failed') for r in candidates):
            winner=min(complete,key=lambda r:(r['object_nmse'],r['run_id']))
            selection={k:winner[k] for k in ('task_id','parameters','run_id','object_nmse',
                'probe_nmse','relative_run_dir','result_sha256','config_sha256')}
        groups.append(dict(study=STUDY,method=method,overlap_ratio=OVERLAP,
            epochs=2 if smoke else EPOCHS,scientific=not smoke,selection=selection,
            completed_count=len(complete),failed_count=sum(r['status']=='failed' for r in candidates),
            missing_count=sum(r['status']=='missing' for r in candidates),
            unfinished_count=sum(r['status'] not in ('complete','failed','missing') for r in candidates),
            candidates=candidates))
    all_complete=[r for r in rows if r['status']=='complete']
    if all_complete and any(r['dataset_array_sha256']!=all_complete[0]['dataset_array_sha256']
                           or r['native_source']!=all_complete[0]['native_source'] for r in all_complete):
        raise ValueError('Methods did not share the same data and native code.')
    if write:
        path=ensure_manifest(root,smoke=smoke)
        with (path.parent/'.summary.lock').open('a+') as stream:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
            for group in groups:
                _atomic_json(path.parent/group['method']/'selection.json',group)
            _atomic_json(path.parent/'summary.json',dict(study=STUDY,scientific=not smoke,
                                                       plan_sha256=_sha(path),groups=groups))
    return groups


def print_progress(root=PROJECT_ROOT, *, smoke=False):
    rows=[load_task(t,root,smoke=smoke) for t in task_plan()]
    print(f'{"Method":<12} {"Complete":>8} {"Failed":>7} {"Unfinished":>11} {"Missing":>8}')
    for method in METHODS:
        statuses=[r['status'] for r in rows if r['method']==method]
        print(f'{method:<12} {statuses.count("complete"):>5}/{len(statuses)} {statuses.count("failed"):>7} '
              f'{sum(s not in ("complete","failed","missing") for s in statuses):>11} '
              f'{statuses.count("missing"):>8}')
    print(f'\n93.75% overlap, 200 epochs: {sum(r["status"]=="complete" for r in rows)}/{len(rows)} complete.')
    for row in rows:
        if row['status']=='failed':
            print(f"Task {row['task_id']} FAILED: {row.get('error','inspect result.json')}")
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-id',type=int,choices=[task['task_id'] for task in task_plan()])
    parser.add_argument('--tasks',help='Comma/range historical task IDs')
    parser.add_argument('--submission-tasks',action='store_true',help='Print or validate retained IDs for Slurm')
    parser.add_argument('--method',choices=(*METHODS,'all'),default='all')
    parser.add_argument('--root',type=Path,default=PROJECT_ROOT)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--rerun-failed',action='store_true')
    for flag in ('list-tasks','write-plan','prepare-data','progress','summarize'):
        parser.add_argument('--'+flag,action='store_true')
    args=parser.parse_args(argv)
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
        print(json.dumps(planned_tasks,indent=2));return 0
    if args.write_plan:
        print(ensure_manifest(args.root,smoke=args.smoke));return 0
    if args.prepare_data:
        prepare_data(args.root,smoke=args.smoke);return 0
    if args.progress:
        print_progress(args.root,smoke=args.smoke);return 0
    if args.summarize:
        print(json.dumps(summarize(args.root,smoke=args.smoke),indent=2));return 0
    if args.task_id is None:
        parser.error('Specify --task-id, --list-tasks, --write-plan, --prepare-data, --progress, or --summarize.')
    task=task_by_id(task_plan(), args.task_id)
    if args.method!='all' and args.method!=task['method']:
        parser.error('The task ID belongs to a different method.')
    run_task(args.task_id,args.root,device=args.device,smoke=args.smoke,rerun_failed=args.rerun_failed)
    return 0
