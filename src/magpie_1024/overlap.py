"""Twelve fixed-parameter overlap runs, isolated from the original sweep."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .protocol import (BASE_BATCH, BASE_ETA, EPOCHS, METHODS, METHOD_NAMES,
                       case_token, float_token, run_id)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OVERLAPS = (0.625, 0.75, 0.875)
PATTERN_COUNTS = {0.625: 400, 0.75: 841, 0.875: 3249}
PARAMETERS = {
    'rpie': {'object_alpha': .02, 'probe_alpha': .5},
    'lsqml': {'damping': .04, 'sigma': .4},
    'gm_rpie': {'object_alpha': .02, 'probe_alpha': .1},
    'gm_magpie': {'object_alpha': .1, 'probe_alpha': .02},
}
STUDY = 'overlap_fixed_075_parameters'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_plan():
    """One method/overlap per task; importing this module needs no Torch."""
    return [dict(task_id=i * 3 + j, method=method, overlap_ratio=overlap,
                 parameters=dict(PARAMETERS[method]), epochs=EPOCHS[overlap],
                 num_patterns=PATTERN_COUNTS[overlap], batch_size=BASE_BATCH,
                 eta=BASE_ETA, patterns_processed=EPOCHS[overlap] * PATTERN_COUNTS[overlap])
            for i, method in enumerate(METHODS) for j, overlap in enumerate(OVERLAPS)]


def verify_selection(method, root=PROJECT_ROOT):
    """Verify the frozen winner against all sixteen original JSON records."""
    root = Path(root).resolve()
    manifest_path = root / 'provenance/selected_parameters.json'
    frozen = json.loads(manifest_path.read_text())['methods'][method]
    if frozen['parameters'] != PARAMETERS[method]:
        raise ValueError('The frozen parameter pair differs from the study.')
    path = root / frozen['selection_file']
    if _sha(path) != frozen['selection_sha256']:
        raise ValueError(f'Original selection failed its hash check: {path}')
    selection = json.loads(path.read_text())
    if selection['completed_count'] != 16 or selection['failed_count'] != 0:
        raise ValueError('The source selection must have sixteen completed runs.')
    rows = []
    source = root / 'provenance/source_sweep/results'
    for candidate in selection['candidates']:
        result = source / candidate['relative_result_path']
        if (_sha(result) != candidate['result_sha256'] or
                _sha(result.with_name('config.json')) != candidate['config_sha256']):
            raise ValueError(f'Original candidate failed its hash check: {result}')
        row = json.loads(result.read_text())
        if row['status'] != 'complete' or row['epochs_completed'] != 3000:
            raise ValueError('A frozen source candidate is incomplete.')
        rows.append(row)
    winner = min(rows, key=lambda row: (row['object_nmse'], row['run_id']))
    if (len(rows) != 16 or winner['run_id'] != frozen['run_id'] or
            selection['parameters'] != PARAMETERS[method]):
        raise ValueError('The frozen pair is not the lowest final source error.')
    winning_path = source / next(c['relative_result_path'] for c in selection['candidates']
                                if c['run_id'] == frozen['run_id'])
    if _sha(winning_path) != frozen['source_result_sha256']:
        raise ValueError('Selected source result failed its hash check.')
    return {**frozen, 'manifest_sha256': _sha(manifest_path)}


def _plan_case(method, overlap_ratio, smoke=False):
    if method not in METHODS or overlap_ratio not in OVERLAPS:
        raise ValueError(f'Choose a method in {METHODS} and overlap in {OVERLAPS}.')
    overlap_ratio = float(overlap_ratio)
    parameters = dict(PARAMETERS[method])
    relative = (Path('runs') / method /
                f'{case_token(overlap_ratio)}_eta_{float_token(BASE_ETA)}_b{BASE_BATCH}' /
                run_id(method, parameters))
    count = ({.625: 81, .75: 169, .875: 625} if smoke else PATTERN_COUNTS)[overlap_ratio]
    return dict(study=STUDY, method=method, algorithm=method,
                overlap_ratio=overlap_ratio, eta=BASE_ETA, batch_size=BASE_BATCH,
                **parameters, run_id=run_id(method, parameters),
                epochs=2 if smoke else EPOCHS[overlap_ratio], epochs_completed=0,
                status='missing', seed=21, data_seed=42, scientific=not smoke,
                num_patterns=count, relative_run_dir=relative.as_posix())


def _base(root, smoke=False):
    return Path(root).resolve() / 'results' / ('smoke' if smoke else '')


def load_case(method, overlap_ratio, root=PROJECT_ROOT, *, smoke=False):
    """Read one authoritative result, including missing/failed states."""
    planned = _plan_case(method, overlap_ratio, smoke)
    path = _base(root, smoke) / planned['relative_run_dir'] / 'result.json'
    if path.exists():
        saved = json.loads(path.read_text())
        keys = ('method', 'overlap_ratio', 'eta', 'batch_size', 'run_id', 'epochs', 'seed', 'scientific')
        keys += ('damping', 'sigma') if method == 'lsqml' else ('object_alpha', 'probe_alpha')
        if any(saved.get(key) != planned[key] for key in keys):
            raise ValueError(f'Saved metadata differs from the overlap plan: {path}')
        planned.update(saved)
        planned['study'] = STUDY
    return planned


def run_case(method, overlap_ratio, root=PROJECT_ROOT, *, device='cuda', smoke=False,
             rerun_failed=False):
    """Start from the original fields, or reuse an already completed case."""
    root = Path(root).resolve()
    selected = verify_selection(method, root)
    planned = _plan_case(method, overlap_ratio, smoke)
    from .engine import run_pair, write_json
    row = run_pair(method, dict(PARAMETERS[method]), root=root,
                   overlap_ratio=float(overlap_ratio), eta=BASE_ETA,
                   batch_size=BASE_BATCH, epochs=planned['epochs'], device=device,
                   smoke=smoke, seed=21, error_every=20, print_every=300,
                   rerun_failed=rerun_failed)
    row.update(study=STUDY, data_seed=42, num_patterns=planned['num_patterns'],
               parameter_selection=selected, equivalent_075_epochs=(
                   row['epochs_completed'] * planned['num_patterns'] / 841),
               planned_pattern_budget=planned['epochs'] * planned['num_patterns'])
    write_json(_base(root, smoke) / row['relative_run_dir'] / 'result.json', row)
    report_case(method, overlap_ratio, root, smoke=smoke)
    if row['status'] != 'complete':
        raise RuntimeError(f"{method}, overlap {overlap_ratio}: {row.get('error', row['status'])}")
    return row


def _atomic_json(path, value):
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def report_case(method, overlap_ratio, root=PROJECT_ROOT, *, smoke=False):
    """Render one case from saved arrays; never import reconstruction code."""
    row = load_case(method, overlap_ratio, root, smoke=smoke)
    base = _base(root, smoke)
    out = base / 'studies' / 'overlap' / case_token(overlap_ratio) / method
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / 'summary.json', row)
    run = base / row['relative_run_dir']
    if row['status'] != 'complete' or not (run / 'reconstruction.npz').exists():
        return out
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (run / 'history.csv').open(newline='') as stream:
        history = list(csv.DictReader(stream))
    seconds = np.cumsum([float(r['reconstruction_seconds']) for r in history])
    scored = [(i, r) for i, r in enumerate(history) if r.get('object_nmse', '') != '']
    epochs = np.array([int(r['epoch']) for _, r in scored])
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    for i, (key, label) in enumerate((('object_nmse', 'Object NMSE (fixed ROI)'),
                                     ('probe_nmse', 'Probe NMSE'))):
        values = [float(r[key]) for _, r in scored]
        for axis, x, xlabel in zip(axes[i],
                (epochs, epochs * row['num_patterns'] / 1e6,
                 np.array([seconds[j] for j, _ in scored]) / 60),
                ('Epoch', 'Processed patterns (million)', 'Reconstruction time (min)')):
            axis.semilogy(x, values)
            axis.set_xlabel(xlabel); axis.set_ylabel(label); axis.grid(alpha=.25)
    fig.suptitle(f'{METHOD_NAMES[method]}, nominal overlap {overlap_ratio:.1%}; fixed 75% selected pair')
    for suffix in ('png', 'svg'):
        fig.savefig(out / f'convergence.{suffix}', dpi=160)
    plt.close(fig)
    config = json.loads((run / 'config.json').read_text())
    with np.load(Path(root).resolve() / config['data_reference'], allow_pickle=False) as archive:
        truth = archive['truth']; probe_truth = archive['probe_truth']
    with np.load(run / 'reconstruction.npz', allow_pickle=False) as archive:
        obj = archive['object_aligned']; probe = archive['probe_aligned']; mask = archive['roi_mask']
    y, x = np.where(mask)
    cut = np.s_[y.min():y.max()+1, x.min():x.max()+1]
    fields = (truth[cut], obj[cut], probe_truth, probe)
    labels = ('True object (ROI)', 'Recovered object (ROI)', 'True probe', 'Recovered probe')
    fig, axes = plt.subplots(2, 4, figsize=(11, 6), constrained_layout=True)
    for i, (field, label) in enumerate(zip(fields, labels)):
        group = fields[:2] if i < 2 else fields[2:]
        mag = axes[0, i].imshow(abs(field), cmap='gray', vmin=0,
                                vmax=max(float(abs(f).max()) for f in group))
        phase = axes[1, i].imshow(np.angle(field), cmap='twilight', vmin=-np.pi, vmax=np.pi)
        axes[0, i].set_title(label)
        for axis in axes[:, i]: axis.set_xticks([]); axis.set_yticks([])
        if i in (1, 3):
            fig.colorbar(mag, ax=axes[0, i-1:i+1].tolist(), shrink=.7)
            fig.colorbar(phase, ax=axes[1, i-1:i+1].tolist(), shrink=.7, ticks=[-np.pi, 0, np.pi])
    axes[0, 0].set_ylabel('Magnitude'); axes[1, 0].set_ylabel('Phase (rad)')
    fig.suptitle(f'{METHOD_NAMES[method]}, nominal overlap {overlap_ratio:.1%}')
    for suffix in ('png', 'svg'):
        fig.savefig(out / f'fields.{suffix}', dpi=160)
    plt.close(fig)
    return out
