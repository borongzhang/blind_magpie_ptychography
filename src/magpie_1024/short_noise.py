"""Noise tests at the three retained parameter-sweep epoch budgets.

The eta=0.1 sweeps fix one pair for each method and overlap. Twenty-four
new cases vary eta only, with the original data, initialization, seeds,
native updates, and common ROI. Planning/progress never import Torch.
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

from .overlap_sweep import (_atomic_json, retained_parameter_grid, parse_task_ids,
                            compatible_runner_identity)
from .protocol import METHODS, PTYCHI_COMMIT, parameter_grid, run_id

PROJECT_ROOT=Path(__file__).resolve().parents[2]
STUDY='short_noise_selected_pairs'
MANIFEST=Path('provenance/short_noise_parameters.json')
OVERLAPS=(.75,.875,.9375)
ETAS=(.2,.4)
EPOCHS={.75:3000,.875:800,.9375:200}
PATTERNS={.75:841,.875:3249,.9375:12769}
TOKENS={.75:'overlap_075',.875:'overlap_0875',.9375:'overlap_09375'}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selection_manifest_path():
    path = Path(os.environ.get('MAGPIE_SHORT_NOISE_MANIFEST', str(MANIFEST)))
    path = path if path.is_absolute() else PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError('Selection manifest must remain inside this project')
    return path


def _frozen():
    frozen=json.loads(selection_manifest_path().read_text())
    expected=dict(study=STUDY,overlaps=list(OVERLAPS),etas=list(ETAS),batch_size=27,
        epochs={str(k):v for k,v in EPOCHS.items()},data_seed=42,seed=21,
        error_every=20,print_every=300,roi_bounds=[46,978,46,978])
    if any(frozen.get(k)!=v for k,v in expected.items()) or len(frozen['selections'])!=12:
        raise ValueError('Frozen short-noise settings differ from the requested experiment.')
    keys={(s['overlap_ratio'],s['method']) for s in frozen['selections']}
    if keys!={(o,m) for o in OVERLAPS for m in METHODS}:
        raise ValueError('The parameter manifest must cover each method and overlap once.')
    for name,digest in frozen['frozen_core_source_sha256'].items():
        path=PROJECT_ROOT/'provenance/selected_parameters.json' if name=='selected_parameters.json' else PROJECT_ROOT/'src/magpie_1024'/name
        if _sha(path)!=digest:
            raise ValueError(f'The retained scientific source changed: {name}')
    return frozen


def task_plan(root=PROJECT_ROOT):
    frozen=_frozen()
    tasks=[]
    for eta in ETAS:
        for overlap in OVERLAPS:
            for method in METHODS:
                selected=next(s for s in frozen['selections'] if s['method']==method and s['overlap_ratio']==overlap)
                tasks.append(dict(task_id=len(tasks),method=method,overlap_ratio=overlap,
                    eta=eta,parameters=selected['parameters'],epochs=EPOCHS[overlap],
                    num_patterns=PATTERNS[overlap],patterns_processed=PATTERNS[overlap]*EPOCHS[overlap],
                    batch_size=27,data_seed=42,seed=21,error_every=20,print_every=300,
                    selection_id=selected['selection_id'],source_completed_count=selected['completed_count'],
                    source_failed_count=selected['failed_count'],source_selection_provisional=selected['provisional']))
    return tasks


def _source_file(relative):
    relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Selection source paths must stay inside the project.')
    return PROJECT_ROOT/relative


def verify_selection(task, root=PROJECT_ROOT):
    """Verify every method-specific immutable candidate and the endpoint winner."""
    frozen=_frozen()
    selected=next(s for s in frozen['selections'] if s['selection_id']==task['selection_id'])
    if (selected['parameters']!=task['parameters'] or selected['method']!=task['method']
            or selected['overlap_ratio']!=task['overlap_ratio']):
        raise ValueError('Task parameters differ from the retained sweep selection.')
    complete=[]
    seen=[]
    baseline=None
    native=None
    runner=None
    for candidate in selected['candidates']:
        for relative,digest in candidate['source_files_sha256'].items():
            path=_source_file(relative)
            if _sha(path)!=digest:
                raise ValueError(f'The original source sweep record changed: {path}')
        path=_source_file(candidate['relative_result_path'])
        result=json.loads(path.read_text())
        config=json.loads(path.with_name('config.json').read_text())
        expected=dict(method=task['method'],overlap_ratio=task['overlap_ratio'],eta=.1,
            epochs=EPOCHS[task['overlap_ratio']],batch_size=27,seed=21,scientific=True)
        if any(config.get(k)!=v or result.get(k)!=v for k,v in expected.items()):
            raise ValueError('Source sweep settings differ from the short-budget experiment.')
        if (config['parameters']!=candidate['parameters'] or config['smoke']
                or config['error_every']!=20 or config['native_source']['commit']!=PTYCHI_COMMIT
                or result['status']!=candidate['status']):
            raise ValueError('Source sweep configuration or status changed.')
        if baseline is None:
            baseline=config['dataset_array_sha256'];native=config['native_source'];runner=config['runner_source_sha256']
        if (config['dataset_array_sha256']!=baseline or config['native_source']!=native
                or not compatible_runner_identity(config['runner_source_sha256'],runner,
                    {'studies.py','high_overlap_sweep.py','high_overlap.py'})):
            raise ValueError('Source candidates have different data or numerical implementations.')
        seen.append(config['parameters'])
        if result['status']=='complete':
            if result['epochs_completed']!=expected['epochs']:
                raise ValueError('Source candidate completed at the wrong epoch.')
            for metric in ('object_nmse','probe_nmse','phase_rmse'):
                value=result[metric]
                if not math.isfinite(value) or value<0:
                    raise ValueError('Source candidate has an invalid final score.')
            with path.with_name('history.csv').open(newline='') as stream:
                history=list(csv.DictReader(stream))
            if [int(r['epoch']) for r in history]!=list(range(expected['epochs']+1)):
                raise ValueError('The source history is incomplete.')
            if any(float(history[-1][metric])!=result[metric] for metric in ('object_nmse','probe_nmse','phase_rmse')):
                raise ValueError('Source endpoint metrics differ from the saved history.')
            if result['object_nmse']!=candidate['object_nmse']:
                raise ValueError('Source endpoint selection score changed.')
            complete.append(candidate)
    def key(parameters):return tuple(sorted(parameters.items()))
    expected_grid=retained_parameter_grid(task['method'])
    if len(seen)!=len(expected_grid) or {key(p) for p in seen}!={key(p) for p in expected_grid}:
        raise ValueError('The source comparison must contain the complete approved method-specific grid.')
    if not complete:
        raise ValueError('The retained source grid contains no completed candidate')
    if any(c['status'] not in ('complete', 'failed') for c in selected['candidates']):
        raise ValueError('The retained source grid contains unfinished candidates')
    best=min(complete,key=lambda c:(c['object_nmse'],c['run_id']))
    if (best['run_id']!=selected['source_run_id'] or best['parameters']!=selected['parameters']
            or len(complete)!=selected['completed_count'] or len(expected_grid)-len(complete)!=selected['failed_count']
            or selected['baseline_array_sha256']!=baseline or selected['native_source']!=native):
        raise ValueError('The frozen pair is not the lowest final-error completed source pair.')
    return selected


def execution_root(root=PROJECT_ROOT):
    return Path(root).resolve()/'evaluations/noise_short'


def _base(root=PROJECT_ROOT,smoke=False):
    return execution_root(root)/'results'/('smoke' if smoke else '')


def _relative(task):
    from .protocol import float_token
    return (Path('runs')/task['method']/f"{TOKENS[task['overlap_ratio']]}_eta_{float_token(task['eta'])}_b27"/
            run_id(task['method'],task['parameters']))


def run_directory(task,root=PROJECT_ROOT,*,smoke=False):
    return _base(root,smoke)/_relative(task)


def ensure_plan(root=PROJECT_ROOT,*,smoke=False):
    from .short_noise_engine import source_identity
    requested=dict(schema_version=1,study=STUDY,scientific=not smoke,smoke=smoke,
        ptychi_commit=PTYCHI_COMMIT,selection_manifest_sha256=_sha(selection_manifest_path()),
        source_identity=source_identity(),starts_from='original initialization',
        baseline='Retained eta=0.1 short parameter sweeps; no baseline resubmission',tasks=task_plan(root))
    directory=_base(root,smoke)/'studies'/STUDY
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'plan.json'
    with (directory/'.plan.lock').open('a+') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text())!=requested:
                raise ValueError(f'Incompatible short-noise plan; preserve existing outputs: {path}')
        else:
            _atomic_json(path,requested)
    return path


def prepare_data(root=PROJECT_ROOT,*,overlap=None,smoke=False):
    from .short_noise_engine import data_config,load_noise_data,baseline_metadata,verify_metadata
    overlaps=OVERLAPS if overlap is None else (overlap,)
    if any(o not in OVERLAPS for o in overlaps):
        raise ValueError('Unsupported preparation overlap.')
    plan=ensure_plan(root,smoke=smoke)
    tasks=task_plan(root)
    prepared=[]
    for o in overlaps:
        for task in tasks:
            if task['overlap_ratio']==o and task['eta']==ETAS[0]:verify_selection(task,root)
        for eta in ETAS:
            config=data_config(overlap_ratio=o,eta=eta,seed=42,smoke=smoke)
            dataset=load_noise_data(execution_root(root),config,source_project_root=Path(root).resolve())
            baseline=baseline_metadata(execution_root(root),root,config)
            check=verify_metadata(config,baseline,dataset.metadata)
            expected=PATTERNS[o] if not smoke else { .75:169,.875:625,.9375:49 }[o]
            if check['num_patterns']!=expected:
                raise ValueError('Noise dataset has the wrong number of scan positions.')
            record=dict(overlap_ratio=o,eta=eta,scientific=not smoke,
                dataset_path=str(dataset.path),metadata=dataset.metadata,invariant_check=check,
                plan_sha256=_sha(plan))
            destination=plan.parent/f"prepared_{TOKENS[o]}_eta_{str(eta).replace('.','p')}.json"
            _atomic_json(destination,record);prepared.append(record)
            print(f'Verified {o:.2%}, eta={eta:g}: {check["num_patterns"]} measurements',flush=True)
            del dataset
    return prepared


def run_task(task_id,root=PROJECT_ROOT,*,device='cuda',smoke=False,rerun_failed=False):
    if isinstance(task_id,bool) or not isinstance(task_id,int) or task_id not in range(24):
        raise ValueError('task_id must be an integer from 0 through 23')
    from .short_noise_engine import run_pair,data_config,cache_directory,baseline_metadata,verify_metadata
    task=task_plan(root)[task_id]
    existing=load_task(task,root,smoke=smoke)
    if existing['status']=='complete':
        print(f'Task {task_id}: reusing verified completed reconstruction',flush=True)
        return existing
    selected=verify_selection(task,root)
    plan=ensure_plan(root,smoke=smoke)
    print(f"Short noise task {task_id}: {task['method']}, {task['overlap_ratio']:.2%}, "
          f"eta={task['eta']:g}, {2 if smoke else task['epochs']} epochs",flush=True)
    row=run_pair(task['method'],task['parameters'],root=execution_root(root),
        source_project_root=Path(root).resolve(),overlap_ratio=task['overlap_ratio'],eta=task['eta'],
        batch_size=27,epochs=task['epochs'],device=device,smoke=smoke,seed=21,
        error_every=20,print_every=300,rerun_failed=rerun_failed)
    directory=_base(root,smoke)/_relative(task)
    config=json.loads((directory/'config.json').read_text())
    data_cfg=data_config(overlap_ratio=task['overlap_ratio'],eta=task['eta'],smoke=smoke)
    metadata=json.loads((cache_directory(execution_root(root),root,data_cfg)/'metadata.json').read_text())
    check=verify_metadata(data_cfg,baseline_metadata(execution_root(root),root,data_cfg),metadata)
    if (config['dataset_fingerprint']!=check['noise_fingerprint']
            or config['dataset_array_sha256']!=metadata['array_sha256']):
        raise ValueError('Reconstruction inputs differ from the verified noise dataset.')
    _atomic_json(directory/'noise_case.json',dict(study=STUDY,task=task,
        selection_id=selected['selection_id'],selection_manifest_sha256=_sha(selection_manifest_path()),
        plan_sha256=_sha(plan),scientific=not smoke,starts_from='original initialization',
        dataset_invariant_check=check))
    if row['status']!='complete':
        raise RuntimeError(f"Task {task_id}: {row.get('error',row['status'])}")
    return row


def _verify_retained_noise(task, directory, config, result):
    """Reuse only byte-sealed old completions; never rewrite their saved identities."""
    from .short_noise_engine import source_identity
    matches = [entry for entry in _frozen().get('retained_noise_artifacts', [])
               if entry['task_id'] == task['task_id']]
    if len(matches) != 1:
        return False
    entry = matches[0]
    try:
        relative = directory.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return False
    settings = ('method','overlap_ratio','eta','parameters','epochs','batch_size',
                'data_seed','seed','error_every','print_every')
    if (entry['relative_run_dir'] != relative or
            any(entry.get(key) != task.get(key) for key in settings) or
            entry.get('selection_id') != task['selection_id'] or
            result.get('status') != 'complete' or result.get('epochs_completed') != task['epochs']):
        raise ValueError('Sealed noise record does not match the retained numerical task')
    hashes = entry['source_files_sha256']
    required = {f'{relative}/{name}' for name in ('config.json','result.json','history.csv','reconstruction.npz')}
    if not required <= set(hashes):
        raise ValueError('Retained noise proof omits required scientific artifacts')
    for name, digest in hashes.items():
        path = _source_file(name)
        if not path.is_file() or _sha(path) != digest:
            raise ValueError(f'The sealed retained noise artifact changed: {path}')
    saved = config['runner_source_sha256']
    if (saved != entry['runner_source_sha256'] or
            not compatible_runner_identity(saved, source_identity(),
                {'short_noise.py','short_noise_entrypoint.py','short_noise_parameters.json'}) or
            config['native_source'] != entry['native_source'] or
            config['dataset_array_sha256'] != entry['dataset_array_sha256'] or
            config['dataset_fingerprint'] != entry['dataset_fingerprint']):
        raise ValueError('The retained noise numerical, native, or data identity changed')
    with (directory/'history.csv').open(newline='') as stream:
        history=list(csv.DictReader(stream))
    if [int(item['epoch']) for item in history] != list(range(task['epochs']+1)):
        raise ValueError('Retained noise history is incomplete')
    for metric in ('object_nmse','probe_nmse','phase_rmse'):
        value=result.get(metric)
        if (not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or
                value != entry['recorded_metrics'][metric] or float(history[-1][metric]) != value):
            raise ValueError('Retained noise endpoint metrics changed')
    case_path=directory/'noise_case.json'
    if case_path.is_file():
        case=json.loads(case_path.read_text())
        if (case['task'] != entry['source_task'] or
                case['selection_id'] != entry['source_selection_id'] or
                case['selection_manifest_sha256'] != entry['source_selection_manifest_sha256'] or
                case['plan_sha256'] != entry['source_plan_sha256']):
            raise ValueError('Retained noise planner proof changed')
    return True


def load_task(task,root=PROJECT_ROOT,*,smoke=False):
    from .short_noise_engine import source_identity
    row=dict(task,epochs=2 if smoke else task['epochs'],status='missing',epochs_completed=0,
             run_id=run_id(task['method'],task['parameters']),relative_run_dir=_relative(task).as_posix())
    directory=_base(root,smoke)/_relative(task)
    path=directory/'result.json'
    if not path.is_file():return row
    result=json.loads(path.read_text());config=json.loads(path.with_name('config.json').read_text())
    expected=dict(method=task['method'],overlap_ratio=task['overlap_ratio'],eta=task['eta'],
        epochs=row['epochs'],batch_size=27,seed=21,scientific=not smoke)
    identity_ok = config['runner_source_sha256']==source_identity()
    reused = False
    sealed = not smoke and any(entry['task_id']==task['task_id']
                               for entry in _frozen().get('retained_noise_artifacts',[]))
    if sealed:
        reused = _verify_retained_noise(task,directory,config,result)
        identity_ok = reused
    if (any(result.get(k)!=v or config.get(k)!=v for k,v in expected.items())
            or config['parameters']!=task['parameters'] or config['smoke']!=smoke
            or not identity_ok
            or config['error_every']!=20 or config['print_every']!=300
            or config['native_source']['commit']!=PTYCHI_COMMIT):
        raise ValueError(f'Saved short-noise settings differ from the plan: {path}')
    row.update(result)
    row['reused_sealed_record']=reused
    if row['status']=='complete' and row.get('epochs_completed')!=row['epochs']:
        raise ValueError(f'Completed noise result has wrong endpoint: {path}')
    if row['status']!='complete' and path.with_name('history.csv').is_file():
        with path.with_name('history.csv').open(newline='') as stream:
            recorded=list(csv.DictReader(stream))
        if recorded:row['epochs_completed']=int(recorded[-1]['epoch'])
    return row


def print_progress(root=PROJECT_ROOT,*,smoke=False,tasks=None):
    rows=[load_task(t,root,smoke=smoke) for t in (task_plan(root) if tasks is None else tasks)]
    print(f'{"Task":>4} {"Method":<12} {"Overlap":>8} {"Eta":>5} {"Status":<12} {"Epochs":>13} {"Object NMSE":>13}')
    for row in rows:
        metric=f"{row['object_nmse']:.6g}" if 'object_nmse' in row else '-'
        print(f"{row['task_id']:>4} {row['method']:<12} {row['overlap_ratio']:>7.2%} "
              f"{row['eta']:>5g} {row['status']:<12} "
              f"{row['epochs_completed']:>5}/{row['epochs']:<7} {metric:>13}")
    print(f"\n{sum(r['status']=='complete' for r in rows)}/{len(rows)} short-noise evaluations complete.")
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-id',type=int,choices=range(24))
    parser.add_argument('--tasks',help='Comma/range historical task IDs; no reindexing')
    parser.add_argument('--selection-manifest',type=Path,help='Read an approved alternative provenance manifest')
    parser.add_argument('--submission-tasks',action='store_true',help='Print only default rerun IDs, or validate --tasks')
    parser.add_argument('--overlap',type=float,choices=OVERLAPS)
    parser.add_argument('--root',type=Path,default=PROJECT_ROOT)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    for flag in ('list-tasks','progress','write-plan','verify-selections','prepare-data','smoke','rerun-failed'):
        parser.add_argument('--'+flag,action='store_true')
    args=parser.parse_args(argv)
    if args.selection_manifest is not None:
        os.environ['MAGPIE_SHORT_NOISE_MANIFEST']=str(args.selection_manifest)
    all_tasks=task_plan(args.root)
    if args.task_id is not None and args.tasks is not None:
        parser.error('Use --task-id or --tasks, not both')
    try:
        ids = parse_task_ids(args.tasks, range(24)) if args.tasks is not None else list(range(24))
    except ValueError as exc:
        parser.error(str(exc))
    tasks=[t for t in all_tasks if t['task_id'] in ids
           and (args.overlap is None or t['overlap_ratio']==args.overlap)]
    if args.submission_tasks:
        if args.tasks is None:
            ids=_frozen().get('resubmit_task_ids')
            if ids is None:
                parser.error('The manifest must explicitly specify resubmit_task_ids before submission')
        ids=parse_task_ids(','.join(map(str,ids)),range(24))
        print(','.join(str(t['task_id']) for t in all_tasks if t['task_id'] in ids
                       and (args.overlap is None or t['overlap_ratio']==args.overlap)))
        return 0
    if args.list_tasks:
        print(json.dumps(tasks,indent=2));return 0
    if args.progress:
        print_progress(args.root,smoke=args.smoke,tasks=tasks);return 0
    if args.write_plan:
        print(ensure_plan(args.root,smoke=args.smoke));return 0
    if args.verify_selections:
        selected=list({t['selection_id']:t for t in tasks}.values())
        for task in selected:verify_selection(task,args.root)
        print(f'Verified {len(selected)} selected pairs against their complete method-specific source grids.');return 0
    if args.prepare_data:
        prepare_data(args.root,overlap=args.overlap,smoke=args.smoke);return 0
    if args.task_id is None and args.tasks is None:
        parser.error('Specify --task-id, --tasks, or a planning/preparation/progress action.')
    if args.task_id is not None and args.overlap is not None and all_tasks[args.task_id]['overlap_ratio']!=args.overlap:
        parser.error('The task belongs to a different overlap.')
    for task in ([all_tasks[args.task_id]] if args.task_id is not None else tasks):
        run_task(task['task_id'],args.root,device=args.device,smoke=args.smoke,rerun_failed=args.rerun_failed)
    return 0
