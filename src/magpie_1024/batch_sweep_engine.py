"""Reuse the frozen numerical loop with an immutable baseline dataset."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import FunctionType

from .protocol import float_token, run_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def source_identity():
    package = Path(__file__).resolve().parent
    paths = {name: package / name for name in (
        'engine.py', 'overlap.py', 'cli.py', 'data.py', 'protocol.py',
        'overlap_sweep.py', 'batch_sweep.py', 'batch_sweep_engine.py')}
    paths['selected_parameters.json'] = PROJECT_ROOT / 'provenance/selected_parameters.json'
    paths['batch_0875_protocol.json'] = PROJECT_ROOT / 'provenance/batch_0875_protocol.json'
    paths['batch_sweep_entrypoint.py'] = PROJECT_ROOT / 'batch_sweep.py'
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def data_config(*, overlap_ratio=.875, eta=.1, seed=42, smoke=False):
    if overlap_ratio != .875 or eta != .1 or seed != 42:
        raise ValueError('This batch study requires overlap=.875, eta=.1 and data seed=42.')
    from .data import DataConfig
    return DataConfig(overlap_ratio=.875, eta=.1, seed=42, smoke=smoke)


def cache_directory(source_project_root, config):
    return (Path(source_project_root).resolve() / 'results' /
            ('smoke' if config.smoke else '') / 'datasets' /
            f'overlap_0875_eta_0p1_seed{config.seed}')


def validate_metadata(config, metadata):
    """Require the same measured data, truth, initialization and common ROI."""
    from .batch_sweep import _frozen
    frozen = _frozen()
    bounds = [4, 60, 4, 60] if config.smoke else [46, 978, 46, 978]
    patterns = 625 if config.smoke else 3249
    if (metadata['config'] != asdict(config) or metadata['evaluation_roi_bounds'] != bounds
            or metadata['num_patterns'] != patterns):
        raise ValueError('Cached data settings, common ROI, or scan count changed.')
    if not config.smoke and (metadata['fingerprint'] != frozen['baseline_dataset_fingerprint']
            or metadata['array_sha256'] != frozen['baseline_array_sha256']):
        raise ValueError('The batch study inputs differ from the retained batch-27 sweep.')
    return dict(dataset_fingerprint=metadata['fingerprint'],
                dataset_array_sha256=metadata['array_sha256'],
                roi_bounds=bounds, num_patterns=patterns,
                same_measured_data_and_initialization_as_baseline=not config.smoke)


def load_batch_data(execution_root, config, *, source_project_root=PROJECT_ROOT):
    """Read the original cache without regenerating or rewriting scientific inputs."""
    from . import data
    directory = cache_directory(source_project_root, config)
    path = directory / 'dataset.npz'
    if config.smoke and not path.is_file():
        # Diagnostic fixtures may generate their own tiny original-format cache.
        data.load_or_make_data(source_project_root, config)
    metadata_path = directory / 'metadata.json'
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f'The retained baseline dataset is required: {directory}')
    metadata = json.loads(metadata_path.read_text())
    validate_metadata(config, metadata)
    _, expected_fingerprint = data._identity(config)
    dataset = data._read_cache(path, config, expected_fingerprint)
    if dataset.metadata != metadata:
        raise ValueError('External and embedded baseline cache metadata differ.')
    validate_metadata(config, dataset.metadata)
    if (dataset.truth.shape != (config.object_size, config.object_size)
            or dataset.intensities.shape != (len(dataset.positions_px), config.probe_size, config.probe_size)
            or not (dataset.object_init == 1).all()
            or int(dataset.roi_mask.sum()) != (config.roi_bounds[1] - config.roi_bounds[0])**2):
        raise ValueError('Cached field shapes, original initialization, or common ROI changed.')
    return dataset


def relative_run_directory(method, overlap_ratio, eta, batch_size, parameters):
    if overlap_ratio != .875 or eta != .1 or batch_size not in (108, 432):
        raise ValueError('Unsupported batch-study condition.')
    return (Path('runs') / method /
            f'overlap_0875_eta_{float_token(eta)}_b{batch_size}' /
            run_id(method, parameters))


def run_pair(method, parameters, *, root, source_project_root=PROJECT_ROOT, **kwargs):
    """Bind paths and provenance around the unchanged reconstruction bytecode."""
    from . import engine
    def loader(execution_root, config):
        return load_batch_data(execution_root, config, source_project_root=source_project_root)
    bindings = dict(engine.run_pair.__globals__)
    bindings.update(DataConfig=data_config, load_or_make_data=loader,
                    relative_run_directory=relative_run_directory, _runner_source=source_identity)
    runner = FunctionType(engine.run_pair.__code__, bindings, name='run_batch_sweep_pair',
                          argdefs=engine.run_pair.__defaults__, closure=engine.run_pair.__closure__)
    runner.__kwdefaults__ = dict(engine.run_pair.__kwdefaults__ or {})
    return runner(method, parameters, root=root, **kwargs)
