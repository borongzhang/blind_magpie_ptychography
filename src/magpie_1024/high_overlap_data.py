"""Nested 93.75% data without changing the frozen lower-overlap generator.

The 87.5% sites and both of their measurement arrays are copied exactly.
New step-eight sites use bounded integer jitter and an independent noise
stream. Importing this module requires NumPy, but never Torch or Pty-Chi.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from . import data as reference_data
from .protocol import float_token

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OVERLAP = .9375
CASE_TOKEN = 'overlap_09375'
NEW_JITTER_SEED_OFFSET = 9375
NEW_NOISE_SEED_OFFSET = 142000


@dataclass(frozen=True)
class HighOverlapDataConfig:
    overlap_ratio: float = OVERLAP
    eta: float = .1
    seed: int = 42
    smoke: bool = False

    def __post_init__(self):
        if float(self.overlap_ratio) != OVERLAP:
            raise ValueError('This extension only supports 93.75% overlap.')
        if not np.isfinite(self.eta) or self.eta < 0:
            raise ValueError('eta must be finite and nonnegative.')
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer.')
        object.__setattr__(self, 'overlap_ratio', OVERLAP)
        object.__setattr__(self, 'eta', float(self.eta))
        object.__setattr__(self, 'seed', int(self.seed))
        object.__setattr__(self, 'smoke', bool(self.smoke))

    @property
    def object_size(self):
        return 64 if self.smoke else 1024

    @property
    def probe_size(self):
        return 16 if self.smoke else 128

    @property
    def roi_bounds(self):
        return (4, 60, 4, 60) if self.smoke else (46, 978, 46, 978)


def scan_geometry(config):
    """Return positions, dense indices of retained sites, retained positions.

    Full geometry contains 113 by 113 nominal step-eight sites. The old
    57 by 57 perturbed sites occupy even rows and columns. Their jitter is
    unchanged (at most four); new sites have jitter at most three. Adjacent
    nominal coordinates therefore remain separated by at least one pixel.
    Boundary coordinate components remain fixed, and every patch is in bounds.

    Smoke geometry is a separate 7 by 7 diagnostic grid, not a scientific
    overlap experiment. It has no inherited measurement sites.
    """
    if config.smoke:
        axis = np.arange(0, 49, 8, dtype=np.int32)
        yy, xx = np.meshgrid(axis, axis, indexing='ij')
        positions = np.stack((yy, xx), axis=-1).reshape(-1, 2)
        return positions, np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.int32)
    source_config = reference_data.DataConfig(overlap_ratio=.875, eta=config.eta,
                                               seed=config.seed, smoke=False)
    retained, _, _ = reference_data.scan_geometry(source_config)
    axis = np.arange(0, 897, 8, dtype=np.int32)
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    nominal = np.stack((yy, xx), axis=-1)
    rng = np.random.default_rng(config.seed + NEW_JITTER_SEED_OFFSET)
    jitter = rng.integers(-3, 4, size=nominal.shape, dtype=np.int32)
    jitter[(nominal == 0) | (nominal == 896)] = 0
    dense = nominal + jitter
    dense[::2, ::2] = retained.reshape(57, 57, 2)
    shared = np.arange(12769).reshape(113, 113)[::2, ::2].ravel()
    positions = dense.reshape(-1, 2)
    if (len(np.unique(positions, axis=0)) != 12769 or
            positions.min() < 0 or positions.max() > 896 or
            not np.array_equal(positions[shared], retained)):
        raise ValueError('The nested high-overlap scan failed its geometry checks.')
    return positions, shared, retained


def _source(config, source_project_root):
    source_config = reference_data.DataConfig(overlap_ratio=.875, eta=config.eta,
                                               seed=config.seed, smoke=config.smoke)
    return reference_data.load_or_make_data(source_project_root, source_config)


def _identity(config, source):
    identity = dict(schema_version=1, study='nested_high_overlap_extension',
        config=asdict(config), data_source_sha256=reference_data._sha256(__file__),
        reference_data_source_sha256=reference_data._sha256(reference_data.__file__),
        assets_sha256=source.metadata['assets_sha256'],
        source_0875_fingerprint=source.fingerprint,
        source_0875_array_sha256=source.metadata['array_sha256'],
        data_protocol=dict(object_size=config.object_size, probe_size=config.probe_size,
            roi_bounds=list(config.roi_bounds), nominal_step=8,
            legacy_sites='copy existing 87.5% positions and measurement arrays exactly',
            new_site_jitter_bound=3, retained_site_jitter_bound=4,
            jitter_seed=config.seed+NEW_JITTER_SEED_OFFSET,
            noise_seed=config.seed+NEW_NOISE_SEED_OFFSET,
            new_noise_order='new sites only, dense row-major, blocks of 64',
            smoke_geometry='49 unjittered diagnostic sites; no inherited measurements'
                           if config.smoke else None))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    return identity, fingerprint


def _generate(config, source, identity, fingerprint, path):
    positions, shared, retained = scan_geometry(config)
    if not config.smoke and not np.array_equal(retained, source.top_left_px):
        raise ValueError('Cached 87.5% positions differ from the frozen geometry.')
    shape = (len(positions), config.probe_size, config.probe_size)
    exact = np.empty(shape, dtype=np.float32)
    measured = np.empty(shape, dtype=np.float32) if config.eta else exact
    if len(shared):
        exact[shared] = source.noiseless_intensities
        measured[shared] = source.intensities
    new = np.setdiff1d(np.arange(len(positions)), shared, assume_unique=True)
    rng = np.random.default_rng(config.seed + NEW_NOISE_SEED_OFFSET)
    for start in range(0, len(new), 64):
        indices = new[start:start+64]
        values = reference_data.forward_intensities(source.truth, source.probe_truth,
                                                    positions[indices])
        exact[indices] = values
        measured[indices] = (config.eta * rng.poisson(
            np.maximum(values.astype(np.float64)/config.eta, 0.))) if config.eta else values
    arrays = dict(intensities=measured, noiseless_intensities=exact,
        truth=source.truth, probe_truth=source.probe_truth, probe_init=source.probe_init,
        object_init=source.object_init, roi_mask=source.roi_mask,
        positions_px=np.asarray(positions+(config.probe_size-config.object_size)/2,
                                dtype=np.float32), top_left_px=positions)
    weights = reference_data.illumination_sum(source.probe_truth, positions, config.object_size)
    middle = config.object_size//4
    central_mean = float(weights[middle:-middle, middle:-middle].mean())
    roi = source.roi_mask
    if np.any(weights[roi] <= 0):
        raise ValueError('The fixed common ROI contains unobserved pixels.')
    metadata = dict(identity, fingerprint=fingerprint,
        num_patterns=len(positions), object_shape=list(source.truth.shape),
        probe_shape=list(source.probe_truth.shape), nominal_overlap=OVERLAP,
        nominal_step=8 if not config.smoke else None,
        positions='nested integer step-eight grid, old jitter retained, new jitter bounded by three',
        shared_0875_count=len(shared), new_pattern_count=len(new),
        shared_dense_indices=shared.tolist(), scan_positions_unique=True,
        measurements='d = eta * Poisson(d_tilde / eta); eta=0 returns d_tilde',
        noise_coupling='exact cached 87.5% values at retained sites; independent stream at new sites',
        forward_transform=source.metadata['forward_transform'],
        evaluation_roi_bounds=list(config.roi_bounds), evaluation_roi_pixels=int(roi.sum()),
        roi_coverage_fraction=float((weights[roi]>0).mean()),
        roi_min_illumination_to_central_mean=float(weights[roi].min()/central_mean),
        probe_truth_squared_norm=source.metadata['probe_truth_squared_norm'],
        probe_initial_squared_norm=source.metadata['probe_initial_squared_norm'],
        probe_initial_shape_nmse=source.metadata['probe_initial_shape_nmse'],
        scientific=not config.smoke,
        smoke_note='Separate 49-site diagnostic geometry; not a scientific overlap experiment.'
                   if config.smoke else None,
        array_sha256={name:reference_data._array_sha256(array) for name,array in arrays.items()})
    return reference_data.SyntheticData(config=config, metadata=metadata, path=path, **arrays)


def load_or_make_data(root, config=None, *, source_project_root=None):
    """Load/create this extension's independent, locked, integrity-checked cache.

    ``root`` is the new sweep execution root. The immutable source dataset
    lives at the project root, optionally supplied for an isolated diagnostic.
    Both complete arrays at shared scientific sites are retained bit for bit.
    """
    config = config or HighOverlapDataConfig()
    if not isinstance(config, HighOverlapDataConfig):
        raise TypeError('Use HighOverlapDataConfig for the high-overlap extension.')
    source_project_root = Path(source_project_root or PROJECT_ROOT).resolve()
    base = Path(root).resolve()/'results'/('smoke' if config.smoke else '')
    directory = base/'datasets'/f'{CASE_TOKEN}_eta_{float_token(config.eta)}_seed{config.seed}'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'dataset.npz'
    with (directory/'.dataset.lock').open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        source = _source(config, source_project_root)
        identity, fingerprint = _identity(config, source)
        if path.exists():
            dataset = reference_data._read_cache(path, config, fingerprint)
        else:
            dataset = _generate(config, source, identity, fingerprint, path)
            fd, temporary = tempfile.mkstemp(prefix='.dataset-', suffix='.npz', dir=directory)
            os.close(fd)
            try:
                np.savez_compressed(temporary,
                    metadata_json=json.dumps(dataset.metadata, sort_keys=True),
                    **{name:getattr(dataset,name) for name in reference_data._ARRAY_FIELDS})
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        if (dataset.truth.shape!=(config.object_size,config.object_size) or
                dataset.intensities.shape!=(len(dataset.positions_px),config.probe_size,config.probe_size) or
                not np.all(dataset.object_init==1) or
                not np.array_equal(dataset.roi_mask,source.roi_mask)):
            raise ValueError('The high-overlap cache has inconsistent dimensions or initialization.')
        positions, shared, retained = scan_geometry(config)
        if not np.array_equal(dataset.top_left_px, positions):
            raise ValueError('The high-overlap cache has inconsistent scan positions.')
        for name in ('truth','probe_truth','probe_init','object_init','roi_mask'):
            if dataset.metadata['array_sha256'][name]!=source.metadata['array_sha256'][name]:
                raise ValueError(f'Changing overlap changed the frozen {name} array.')
        if len(shared) and (not np.array_equal(dataset.intensities[shared],source.intensities)
                            or not np.array_equal(dataset.noiseless_intensities[shared],source.noiseless_intensities)):
            raise ValueError('The inherited 87.5% measurement arrays were changed.')
        reference_data._atomic_json(directory/'metadata.json',dataset.metadata)
        return dataset
