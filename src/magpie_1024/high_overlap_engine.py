"""Use the frozen reconstruction loop with an isolated high-overlap adapter.

No original module globals or source files are changed. The native update
rules and scoring loop are reused verbatim; the dataset, output directory,
and recorded source identity are supplied by this extension.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import FunctionType

from .high_overlap_data import HighOverlapDataConfig, load_or_make_data
from .protocol import float_token, run_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def source_identity():
    """Flat hashes of frozen runner plus extension, without importing Torch."""
    package = Path(__file__).resolve().parent
    paths = {name:package/name for name in
             ('engine.py','overlap.py','cli.py','data.py','protocol.py',
              'high_overlap_data.py','high_overlap_engine.py','high_overlap_sweep.py')}
    paths['selected_parameters.json'] = PROJECT_ROOT/'provenance/selected_parameters.json'
    paths['high_overlap.py'] = PROJECT_ROOT/'high_overlap.py'
    return {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}


def relative_run_directory(method, overlap_ratio, eta, batch_size, parameters):
    if float(overlap_ratio)!=.9375:
        raise ValueError('This runner only supports 93.75% overlap.')
    return (Path('runs')/method/
            f'overlap_09375_eta_{float_token(eta)}_b{batch_size}'/run_id(method,parameters))


def run_pair(method, parameters, *, root, source_project_root=None,
             overlap_ratio=.9375, epochs=200, **kwargs):
    """Run the unchanged native loop with process-local dependency bindings.

    A new function object receives its own globals mapping. This avoids
    monkeypatching the frozen engine or affecting other studies in a notebook.
    Every scientific configuration includes all extension source hashes.
    """
    from . import engine
    if float(overlap_ratio)!=.9375:
        raise ValueError('This runner only supports 93.75% overlap.')
    def data_loader(execution_root, config):
        return load_or_make_data(execution_root, config,
                                 source_project_root=source_project_root)
    bindings = dict(engine.run_pair.__globals__)
    bindings.update(DataConfig=HighOverlapDataConfig,
        load_or_make_data=data_loader, relative_run_directory=relative_run_directory,
        _runner_source=source_identity)
    runner = FunctionType(engine.run_pair.__code__, bindings,
                          name='run_high_overlap_pair',
                          argdefs=engine.run_pair.__defaults__, closure=engine.run_pair.__closure__)
    runner.__kwdefaults__ = dict(engine.run_pair.__kwdefaults__ or {})
    return runner(method, parameters, root=root, overlap_ratio=overlap_ratio,
                  epochs=epochs, **kwargs)
