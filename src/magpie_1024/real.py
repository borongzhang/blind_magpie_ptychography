"""Paper-matched full-data reruns using the updated native implementations."""
from __future__ import annotations

import argparse
import csv
import fcntl
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
import math
from pathlib import Path
import time

from .protocol import METHODS, PTYCHI_COMMIT
from .overlap_sweep import _atomic_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY = 'real_paper_rerun'
DATASETS = ('test_pattern', 'integrated_circuit')
PARAMETERS = {
    'test_pattern': {'rpie': dict(object_alpha=.2, probe_alpha=1.),
        'lsqml': dict(beta=.5, sigma=.5, damping=.1),
        'gm_rpie': dict(object_alpha=.2, probe_alpha=1.),
        'gm_magpie': dict(object_alpha=2., probe_alpha=1.)},
    'integrated_circuit': {'rpie': dict(object_alpha=1., probe_alpha=.5),
        'lsqml': dict(beta=.5, sigma=.5, damping=.1),
        'gm_rpie': dict(object_alpha=1., probe_alpha=.1),
        'gm_magpie': dict(object_alpha=3., probe_alpha=.1)},
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_plan():
    return [dict(task_id=i*4+j, dataset=dataset, method=method,
        parameters=PARAMETERS[dataset][method], epochs=100,
        batch_size=244 if dataset=='test_pattern' else 29,
        num_patterns=14640 if dataset=='test_pattern' else 812,
        probe_size=256 if dataset=='test_pattern' else 512,
        seed=42, score_every=20, print_every=10)
        for i,dataset in enumerate(DATASETS) for j,method in enumerate(METHODS)]


def _base(root, smoke=False):
    return Path(root).resolve()/'evaluations/real/results'/('smoke' if smoke else '')


def _directory(task, root, smoke=False):
    return _base(root,smoke)/'runs'/task['dataset']/task['method']


def _identity(root):
    package=Path(__file__).resolve().parent
    return {**{name:_sha(package/name) for name in ('real.py','real_data.py','engine.py','data.py','protocol.py')},
        'entrypoint':_sha(PROJECT_ROOT/'real.py'),
        'paper_protocol':_sha(Path(root).resolve()/'provenance/real_protocol.json')}


def ensure_plan(root=PROJECT_ROOT, *, smoke=False):
    directory=_base(root,smoke)/'study'
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'plan.json'
    requested=dict(study=STUDY,schema_version=1,scientific=not smoke,smoke=smoke,
        native_commit=PTYCHI_COMMIT,runner_sha256=_identity(root),tasks=task_plan(),
        initialization='shared published-notebook initialization, seed42',
        real_metrics='frozen amplitude MSE and qualitative native fields; no truth NMSE')
    with (directory/'.plan.lock').open('a+') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text())!=requested:
                raise ValueError(f'Incompatible real-study plan retained at {path}')
        else:_atomic_json(path,requested)
    return path


def _json_options(value):
    if isinstance(value,dict):return {str(k):_json_options(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_json_options(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    return value


def frozen_amplitude_mse(task, data):
    import torch
    total=torch.zeros((),dtype=torch.float64,device=task.get_data('object').device)
    count=0
    batch=64 if data.probe_init.shape[-1]<=256 else 16
    with torch.no_grad():
        for start in range(0,len(data.data),batch):
            stop=min(start+batch,len(data.data))
            index=torch.arange(start,stop,device=total.device)
            prediction=task.reconstructor.forward_model.forward(index)
            prediction=torch.fft.fftshift(prediction,dim=(-2,-1))
            measured=torch.as_tensor(data.data[start:stop],device=total.device)
            delta=torch.sqrt(prediction.clamp_min(0)+1e-7)-torch.sqrt(measured.clamp_min(0)+1e-7)
            total+=delta.square().sum(dtype=torch.float64)
            count+=delta.numel()
    value=float((total/count).item())
    if not math.isfinite(value) or value<0:raise FloatingPointError('Non-finite frozen amplitude MSE')
    return value


def _fields(task):
    import numpy as np
    fields={name:task.get_data_to_cpu(name,as_numpy=True).squeeze()
            for name in ('object','probe','probe_positions')}
    if not all(np.isfinite(array).all() for array in fields.values()):
        raise FloatingPointError('Non-finite real-data fields or scan coordinates')
    return fields


def _save_figure(directory, fields):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,2,figsize=(9,8))
    for axis,name,phase in zip(axes.flat,('object','object','probe','probe'),(False,True,False,True)):
        values=np.angle(fields[name]) if phase else abs(fields[name])
        image=axis.imshow(values,cmap='twilight' if phase else 'gray')
        axis.set_title(f'{name.capitalize()} {"phase" if phase else "magnitude"}')
        axis.set_xlabel('Pixel');axis.set_ylabel('Pixel')
        fig.colorbar(image,ax=axis,fraction=.046,pad=.04)
    fig.tight_layout()
    fig.savefig(directory/'fields.png',dpi=160)
    plt.close(fig)


def _completed(directory, row, epochs):
    import numpy as np
    if row.get('status')!='complete' or row.get('epochs_completed')!=epochs:return False
    if not (directory/'reconstruction.npz').is_file():return False
    try:
        with (directory/'history.csv').open(newline='') as stream:history=list(csv.DictReader(stream))
        if [int(r['epoch']) for r in history]!=list(range(epochs+1)):return False
        with np.load(directory/'reconstruction.npz',allow_pickle=False) as fields:
            if fields['object'].ndim!=2 or fields['probe'].ndim!=2:return False
            if fields['positions_px'].shape!=(row['num_patterns'],2):return False
            if fields['object_origin_coords'].shape!=(2,):return False
            if not all(np.isfinite(fields[name]).all() for name in ('object','probe','positions_px','object_origin_coords')):return False
        return float(history[-1]['frozen_amplitude_mse'])==row['frozen_amplitude_mse']
    except (OSError,ValueError,KeyError):return False


def run_task(task_id, root=PROJECT_ROOT, *, device='cuda', smoke=False, rerun_failed=False):
    import numpy as np
    import torch
    from ptychi.api.task import PtychographyTask
    from ptychi.timing import timer_utils
    from .engine import (check_native_source,_run_lock,_environment,OnlineAmplitudeMSE,_sync,write_json)
    from .real_data import load_or_make_real_data,make_real_options
    if isinstance(task_id,bool) or not isinstance(task_id,int) or task_id not in range(8):
        raise ValueError('Real task ID must be from zero through seven.')
    if device not in ('cuda','cpu'):raise ValueError('Device must be cuda or cpu.')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('A CUDA GPU allocation is required.')
    root=Path(root).resolve();planned=task_plan()[task_id]
    epochs=2 if smoke else 100
    native=check_native_source()
    plan=ensure_plan(root,smoke=smoke)
    directory=_directory(planned,root,smoke)
    logging.getLogger('ptychi').setLevel(logging.WARNING)
    torch.set_num_threads(1)
    previous_timing=timer_utils.ENABLE_TIMING
    timer_utils.toggle_timer(False)
    task=None
    try:
        with _run_lock(directory.parent/f'.{planned["method"]}.lock'):
            data=load_or_make_real_data(root,planned['dataset'],smoke=smoke)
            options=make_real_options(planned['method'],data,device=device,smoke=smoke)
            assert options.reconstructor_options.num_epochs==epochs
            assert options.reconstructor_options.batch_size==(8 if smoke else planned['batch_size'])
            config=dict(schema_version=1,study=STUDY,scientific=not smoke,smoke=smoke,
                task_id=task_id,dataset=planned['dataset'],method=planned['method'],
                parameters=planned['parameters'],epochs=epochs,seed=42,device=device,
                batch_size=options.reconstructor_options.batch_size,
                num_patterns=len(data.data),dataset_fingerprint=data.fingerprint,
                dataset_array_sha256=data.metadata['array_sha256'],
                dataset_reference=str(data.path.relative_to(root)),
                native_source=native,runner_sha256=_identity(root),plan_sha256=_sha(plan),
                options=_json_options(options.get_dict()),
                score_every=20,print_every=10,
                metric='frozen raw amplitude MSE before any display registration; no ground truth',
                online_metric='pixel-weighted pre-update minibatch amplitude MSE')
            config_path=directory/'config.json';result_path=directory/'result.json'
            if config_path.exists() and json.loads(config_path.read_text())!=config:
                raise ValueError(f'Incompatible saved real settings retained at {directory}')
            if result_path.exists():
                saved=json.loads(result_path.read_text())
                if not config_path.exists():raise ValueError('Saved real result has no configuration.')
                if saved.get('status')=='failed' and not rerun_failed:
                    raise RuntimeError(f'Retaining recorded real-data failure: {saved.get("error")}')
                if _completed(directory,saved,epochs):
                    print(f'Real task {task_id}: using completed result',flush=True)
                    return saved
            if directory.exists() and any(directory.iterdir()):
                archive=directory.parent/'_previous_runs';archive.mkdir(exist_ok=True)
                stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                directory.rename(archive/f'{directory.name}_{stamp}')
            directory.mkdir(parents=True,exist_ok=True)
            write_json(config_path,config);write_json(directory/'environment.json',_environment(device))
            row=dict(study=STUDY,task_id=task_id,dataset=planned['dataset'],method=planned['method'],
                parameters=planned['parameters'],epochs=epochs,epochs_completed=0,status='running',
                scientific=not smoke,smoke=smoke,batch_size=options.reconstructor_options.batch_size,
                num_patterns=len(data.data),dataset_fingerprint=data.fingerprint,
                seed=42,relative_run_dir=str(directory.relative_to(_base(root,smoke))))
            write_json(result_path,row)
            print(f'Real task {task_id}: {planned["dataset"]}, {planned["method"]}, '
                  f'{len(data.data)} patterns, batch {row["batch_size"]}, {epochs} epochs',flush=True)
            epoch=0;started=time.perf_counter();training_seconds=0.;scoring_seconds=0.
            try:
                task=PtychographyTask(options,diffraction_data=data.data,
                    object_data=data.object_init[None].copy(),probe_data=data.probe_init[None,None].copy(),
                    probe_position_x_px=data.positions_px[:,1].copy(),
                    probe_position_y_px=data.positions_px[:,0].copy())
                task.reconstructor.dataloader.generator.manual_seed(42)
                collector=OnlineAmplitudeMSE();task.reconstructor.loss_tracker.metric_function=collector
                columns=('epoch','online_amplitude_mse','frozen_amplitude_mse',
                         'position_update_rms_px','position_update_max_px','position_mean_y_px',
                         'position_mean_x_px','reconstruction_seconds','scoring_seconds','patterns_processed')
                with (directory/'history.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
                    start=time.perf_counter();initial=frozen_amplitude_mse(task,data)
                    scoring_seconds+=time.perf_counter()-start
                    writer.writerow(dict(epoch=0,online_amplitude_mse='',frozen_amplitude_mse=initial,
                        position_update_rms_px=0.,position_update_max_px=0.,position_mean_y_px=float(data.positions_px[:,0].mean()),
                        position_mean_x_px=float(data.positions_px[:,1].mean()),reconstruction_seconds=0.,
                        scoring_seconds=scoring_seconds,patterns_processed=0));stream.flush()
                    for epoch in range(1,epochs+1):
                        collector.reset();_sync(device);start=time.perf_counter()
                        task.run(n_epochs=1);_sync(device)
                        elapsed=time.perf_counter()-start;training_seconds+=elapsed
                        online=collector.value()
                        if not math.isfinite(online):raise FloatingPointError(f'Non-finite online amplitude MSE at epoch {epoch}')
                        metrics={name:'' for name in columns[2:7]}
                        score_time=0.
                        if epoch%20==0 or epoch==epochs:
                            start=time.perf_counter();fields=_fields(task)
                            current=fields['probe_positions'];delta=current-data.positions_px
                            metrics=dict(frozen_amplitude_mse=frozen_amplitude_mse(task,data),
                                position_update_rms_px=float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))),
                                position_update_max_px=float(np.sqrt(np.sum(delta**2,axis=1)).max()),
                                position_mean_y_px=float(current[:,0].mean()),position_mean_x_px=float(current[:,1].mean()))
                            score_time=time.perf_counter()-start;scoring_seconds+=score_time
                        writer.writerow(dict(epoch=epoch,online_amplitude_mse=online,**metrics,
                            reconstruction_seconds=elapsed,scoring_seconds=score_time,patterns_processed=epoch*len(data.data)))
                        stream.flush();row['epochs_completed']=epoch
                        if epoch%10==0 or epoch==epochs:
                            row.update(online_amplitude_mse=online,reconstruction_seconds=training_seconds,
                                       scoring_seconds=scoring_seconds)
                            if metrics['frozen_amplitude_mse']!='':row.update(metrics)
                            write_json(result_path,row)
                            print(f'Real task {task_id}: epoch {epoch}/{epochs}; online MSE={online:.6g}',flush=True)
                origin=task.object.pos_origin_coords
                origin=origin.detach().cpu().numpy() if torch.is_tensor(origin) else np.asarray(origin)
                np.savez_compressed(directory/'reconstruction.npz',object=fields['object'],probe=fields['probe'],
                    positions_px=fields['probe_positions'],initial_positions_px=data.positions_px,
                    object_origin_coords=origin,initial_probe=data.probe_init,
                    selected_pattern_indices=data.selected_indices,dx_m=data.dx_m)
                _save_figure(directory,fields)
                row.update(status='complete',epochs_completed=epochs,**metrics,
                    total_seconds=time.perf_counter()-started,patterns_processed=epochs*len(data.data))
            except Exception as exc:
                row.update(status='failed',error=f'{type(exc).__name__}: {exc}',failure_epoch=epoch,
                    reconstruction_seconds=training_seconds,scoring_seconds=scoring_seconds,
                    total_seconds=time.perf_counter()-started)
                print(f'Real task {task_id} FAILED: {row["error"]}',flush=True)
            write_json(result_path,row)
            if row['status']!='complete':raise RuntimeError(row['error'])
            return row
    finally:
        if task is not None:task.reconstructor.pbar.close()
        del task;gc.collect()
        timer_utils.toggle_timer(previous_timing)
        if device=='cuda':torch.cuda.empty_cache()


def load_task(task,root=PROJECT_ROOT,*,smoke=False):
    row={**task,'epochs':2 if smoke else 100,'epochs_completed':0,'status':'missing'}
    path=_directory(task,root,smoke)/'result.json'
    if path.is_file():
        saved=json.loads(path.read_text())
        expected=dict(dataset=task['dataset'],method=task['method'],parameters=task['parameters'],
                      epochs=row['epochs'],seed=42,scientific=not smoke,smoke=smoke)
        if any(saved.get(k)!=v for k,v in expected.items()):raise ValueError(f'Saved real metadata differs: {path}')
        row.update(saved)
    return row


def print_progress(root=PROJECT_ROOT,*,smoke=False):
    print(f'{"Task":>4} {"Dataset":<20} {"Method":<12} {"Status":<10} {"Epochs":>9} {"Frozen amplitude MSE":>21}')
    rows=[load_task(t,root,smoke=smoke) for t in task_plan()]
    for row in rows:
        metric=f'{row["frozen_amplitude_mse"]:.6g}' if 'frozen_amplitude_mse' in row else '-'
        print(f'{row["task_id"]:>4} {row["dataset"]:<20} {row["method"]:<12} {row["status"]:<10} '
              f'{row["epochs_completed"]:>4}/{row["epochs"]:<4} {metric:>21}')
    print(f'\n{sum(r["status"]=="complete" for r in rows)}/8 real-data runs complete.')
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-id',type=int,choices=range(8))
    parser.add_argument('--list-tasks',action='store_true')
    parser.add_argument('--progress',action='store_true')
    parser.add_argument('--write-plan',action='store_true')
    parser.add_argument('--root',type=Path,default=PROJECT_ROOT)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--rerun-failed',action='store_true')
    args=parser.parse_args(argv);tasks=task_plan()
    if args.list_tasks:print(json.dumps(tasks,indent=2));return 0
    if args.progress:print_progress(args.root,smoke=args.smoke);return 0
    if args.write_plan:print(ensure_plan(args.root,smoke=args.smoke));return 0
    if args.task_id is not None:tasks=[tasks[args.task_id]]
    failed=False
    for task in tasks:
        try:run_task(task['task_id'],args.root,device=args.device,smoke=args.smoke,rerun_failed=args.rerun_failed)
        except Exception as exc:
            print(f'Real task {task["task_id"]} failed: {type(exc).__name__}: {exc}',flush=True);failed=True
    return int(failed)
