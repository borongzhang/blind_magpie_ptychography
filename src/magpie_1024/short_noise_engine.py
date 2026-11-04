"""Use the frozen numerical loop for the three short noise comparisons."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import FunctionType

from .protocol import float_token, run_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OVERLAPS = (.75, .875, .9375)
TOKENS = {.75: 'overlap_075', .875: 'overlap_0875', .9375: 'overlap_09375'}
INVARIANT_ARRAYS = ('noiseless_intensities', 'truth', 'probe_truth', 'probe_init',
                    'object_init', 'positions_px', 'top_left_px', 'roi_mask')


def source_identity():
    """Hash numerical dependencies and the complete new experiment definition."""
    package=Path(__file__).resolve().parent
    paths={name:package/name for name in ('engine.py','overlap.py','cli.py','data.py',
        'protocol.py','high_overlap_data.py','short_noise_engine.py','short_noise.py')}
    paths['selected_parameters.json']=PROJECT_ROOT/'provenance/selected_parameters.json'
    paths['short_noise_parameters.json']=PROJECT_ROOT/'provenance/short_noise_parameters.json'
    paths['short_noise_entrypoint.py']=PROJECT_ROOT/'short_noise.py'
    return {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}


def data_config(*, overlap_ratio, eta, seed=42, smoke=False):
    if overlap_ratio not in OVERLAPS:
        raise ValueError('Choose overlap 0.75, 0.875, or 0.9375.')
    if overlap_ratio==.9375:
        from .high_overlap_data import HighOverlapDataConfig
        return HighOverlapDataConfig(overlap_ratio=overlap_ratio,eta=eta,seed=seed,smoke=smoke)
    from .data import DataConfig
    return DataConfig(overlap_ratio=overlap_ratio,eta=eta,seed=seed,smoke=smoke)


def cache_directory(execution_root, source_project_root, config):
    root=Path(execution_root).resolve() if config.overlap_ratio==.9375 else Path(source_project_root).resolve()
    return (root/'results'/('smoke' if config.smoke else '')/'datasets'/
            f'{TOKENS[config.overlap_ratio]}_eta_{float_token(config.eta)}_seed{config.seed}')


def baseline_metadata(execution_root, source_project_root, config):
    source_project_root=Path(source_project_root).resolve()
    baseline=data_config(overlap_ratio=config.overlap_ratio,eta=.1,seed=config.seed,smoke=config.smoke)
    if config.overlap_ratio==.9375:
        baseline_root=source_project_root/'evaluations/high_overlap_sweep'
        directory=cache_directory(baseline_root,source_project_root,baseline)
    else:
        baseline_root=source_project_root
        directory=cache_directory(baseline_root,source_project_root,baseline)
    path=directory/'metadata.json'
    if config.smoke and not path.is_file():
        if config.overlap_ratio==.9375:
            from .high_overlap_data import load_or_make_data
            data=load_or_make_data(baseline_root,baseline,source_project_root=source_project_root)
        else:
            from .data import load_or_make_data
            data=load_or_make_data(source_project_root,baseline)
        metadata=data.metadata
        del data
    else:
        if not path.is_file() or not (directory/'dataset.npz').is_file():
            raise FileNotFoundError(f'The retained eta=0.1 baseline is required: {directory}')
        metadata=json.loads(path.read_text())
    if metadata['config']!=asdict(baseline):
        raise ValueError('Baseline cache settings differ from the requested comparison.')
    if not config.smoke:
        frozen=json.loads((PROJECT_ROOT/'provenance/short_noise_parameters.json').read_text())
        selections=[s for s in frozen['selections'] if s['overlap_ratio']==config.overlap_ratio]
        if len(selections)!=4 or any(s['baseline_array_sha256']!=metadata['array_sha256'] for s in selections):
            raise ValueError('The retained baseline arrays differ from the source parameter sweep.')
    return metadata


def verify_metadata(config, baseline, noisy):
    if noisy['config']!=asdict(config):
        raise ValueError('Noise dataset settings differ from the planned comparison.')
    if any(noisy['array_sha256'][name]!=baseline['array_sha256'][name] for name in INVARIANT_ARRAYS):
        raise ValueError('Changing eta altered a truth field, initialization, scan, ROI, or exact measurements.')
    bounds=[4,60,4,60] if config.smoke else [46,978,46,978]
    if baseline['evaluation_roi_bounds']!=bounds or noisy['evaluation_roi_bounds']!=bounds:
        raise ValueError('The common evaluation ROI changed.')
    return dict(baseline_fingerprint=baseline['fingerprint'],noise_fingerprint=noisy['fingerprint'],
        shared_array_sha256={name:noisy['array_sha256'][name] for name in INVARIANT_ARRAYS},
        measured_intensities_sha256=noisy['array_sha256']['intensities'],
        roi_bounds=bounds,num_patterns=noisy['num_patterns'])


def load_noise_data(execution_root, config, *, source_project_root=PROJECT_ROOT):
    """Load the shared cache and verify all noise-independent arrays before updates."""
    baseline=baseline_metadata(execution_root,source_project_root,config)
    if config.overlap_ratio==.9375:
        from .high_overlap_data import load_or_make_data
        dataset=load_or_make_data(execution_root,config,source_project_root=source_project_root)
    else:
        from .data import load_or_make_data
        dataset=load_or_make_data(source_project_root,config)
    verify_metadata(config,baseline,dataset.metadata)
    return dataset


def relative_run_directory(method, overlap_ratio, eta, batch_size, parameters):
    if overlap_ratio not in OVERLAPS:
        raise ValueError('Unsupported short-noise overlap.')
    return (Path('runs')/method/
            f'{TOKENS[overlap_ratio]}_eta_{float_token(eta)}_b{batch_size}'/
            run_id(method,parameters))


def run_pair(method, parameters, *, root, source_project_root=PROJECT_ROOT,
             overlap_ratio, epochs, **kwargs):
    """Call the original reconstruction bytecode with isolated data/source bindings."""
    from . import engine
    def loader(execution_root,config):
        return load_noise_data(execution_root,config,source_project_root=source_project_root)
    bindings=dict(engine.run_pair.__globals__)
    bindings.update(DataConfig=data_config,load_or_make_data=loader,
                    relative_run_directory=relative_run_directory,_runner_source=source_identity)
    runner=FunctionType(engine.run_pair.__code__,bindings,name='run_short_noise_pair',
        argdefs=engine.run_pair.__defaults__,closure=engine.run_pair.__closure__)
    runner.__kwdefaults__=dict(engine.run_pair.__kwdefaults__ or {})
    return runner(method,parameters,root=root,overlap_ratio=overlap_ratio,epochs=epochs,**kwargs)
