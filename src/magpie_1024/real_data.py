"""Paper-matched measured-data inputs and native reconstruction options.

Only data loading/initialization lives here; all numerical updates use the
pinned upstream Pty-Chi implementations. Large measured arrays are cached as
float32 NPY files and opened read-only through memory maps.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from .data import generate_probe
from .protocol import METHODS

PAPER_SHA256 = '030cf0c5fd460b731095a006804ae47b1b8c05a7068a842b53b38124b1391f7d'
ORIGINAL_REPOSITORY = 'https://github.com/borongzhang/blind_magpie_ptychography'
ORIGINAL_COMMIT = '9baaf92c57f0ad067edf5005b9050a1bccf0dffa'
DATASETS = ('test_pattern', 'integrated_circuit')
SPECS = {
    'test_pattern': dict(filename='Velo_18c3_comm_TP_scan119_data_roi0_Ndp256_dp.hdf5',
        source_count=14641, retained_count=14640, side=256, batch_size=244,
        defocus_m=-.3e-3,
        source_sha256='8fdb75e4cd96b20ef7bab7c99a7b46be2e9b98c037b6c08f89f9ca52f3db2f0d',
        notebook='examples/real_data/test_pattern_full.ipynb',
        notebook_sha256='e0168c5401d8765f0c3d12803d72969975cca06c7828fe78f8a8fcc2163811e1'),
    'integrated_circuit': dict(filename='Velo_18c3_comm_chip65nm_scan054_data_roi0_Ndp512_us2.hdf5',
        source_count=812, retained_count=812, side=512, batch_size=29,
        defocus_m=-1e-3,
        source_sha256='c889cfef624b49ef0b5da930b94707ceb26aed6beed0ee3e39ac9d572f6f487b',
        notebook='examples/real_data/chip_full.ipynb',
        notebook_sha256='1dbc9d669b6845906474b6185dc4fafb800394f1b366311f8bafa6eb21953093'),
}
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
ARRAYS = ('data', 'positions_px', 'probe_init', 'object_init', 'selected_indices')


@dataclass(frozen=True)
class RealData:
    dataset: str
    data: np.ndarray
    positions_px: np.ndarray
    probe_init: np.ndarray
    object_init: np.ndarray
    selected_indices: np.ndarray
    dx_m: float
    wavelength_m: float
    metadata: dict
    path: Path

    @property
    def fingerprint(self):
        return self.metadata['fingerprint']


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array):
    array = np.asarray(array)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    raw = memoryview(array).cast('B')
    digest = hashlib.sha256()
    for start in range(0, len(raw), 8 * 1024**2):
        digest.update(raw[start:start + 8 * 1024**2])
    return digest.hexdigest()


def _atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix='.metadata-', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _save_npy(path, value):
    fd, temporary = tempfile.mkstemp(prefix='.array-', suffix='.npy', dir=path.parent)
    os.close(fd)
    try:
        np.save(temporary, value, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def selected_indices(dataset, *, smoke=False):
    spec = SPECS[dataset]
    if smoke:
        return np.sort(np.random.default_rng(42).choice(spec['source_count'], 16, replace=False)).astype(np.int64)
    if dataset == 'test_pattern':
        # This exactly reproduces random_pattern_fraction=1.0, complete
        # minibatches, seed42 in the published full-data notebook.
        indices = np.sort(np.random.default_rng(42).choice(
            spec['source_count'], spec['retained_count'], replace=False)).astype(np.int64)
        if np.setdiff1d(np.arange(spec['source_count']), indices).tolist() != [1772]:
            raise RuntimeError('The published test-pattern selection was not reproduced.')
        return indices
    return np.arange(spec['source_count'], dtype=np.int64)


def _identity(dataset, source, smoke):
    spec = SPECS[dataset]
    source_hash = file_sha256(source)
    if source_hash != spec['source_sha256']:
        raise ValueError(f'Real source file differs from the preserved input: {source}')
    identity = dict(schema_version=1, dataset=dataset, smoke=smoke, scientific=not smoke,
        source_filename=spec['filename'], source_hdf5_sha256=source_hash,
        loader_sha256=file_sha256(__file__),
        probe_generator_sha256=file_sha256(Path(__file__).with_name('data.py')),
        paper_sha256=PAPER_SHA256,
        original_repository=ORIGINAL_REPOSITORY, original_commit=ORIGINAL_COMMIT,
        original_notebook=spec['notebook'], original_notebook_sha256=spec['notebook_sha256'],
        source_count=spec['source_count'], retained_count=16 if smoke else spec['retained_count'],
        side=16 if smoke else spec['side'], seed=42, object_extra_pixels=20,
        defocus_m=spec['defocus_m'],
        selection='seed42 sorted choice, sixteen positions' if smoke else (
            'seed42 sorted all-but-one choice; omitted source index1772' if dataset=='test_pattern'
            else 'all source positions, original order'),
        preprocessing='supplied centered intensity arrays; float32; no additional correction or mask',
        object_initialization='-1 + .01/sqrt(2)*(normal_real + 1j*normal_imag), TorchCPU seed42',
        position_convention='[ppY,ppX]/dx then float32 mean centering',
        probe_normalization='float32 mean intensity summed / side**2 equals squared probe norm',
        smoke_geometry='center16crop; scale positions to span32px' if smoke else None)
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()
    return identity, hashlib.sha256(encoded).hexdigest()


def _generate(directory, source, dataset, identity, fingerprint, smoke):
    import torch
    spec = SPECS[dataset]
    indices = selected_indices(dataset, smoke=smoke)
    side = 16 if smoke else spec['side']
    fd, temporary = tempfile.mkstemp(prefix='.measured-', suffix='.npy', dir=directory)
    os.close(fd)
    try:
        measured = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                             shape=(len(indices), side, side))
        with h5py.File(source, 'r') as handle:
            if tuple(handle['dp'].shape) != (spec['source_count'], spec['side'], spec['side']):
                raise ValueError('Unexpected measured array shape.')
            if smoke:
                lo=(spec['side']-side)//2
                measured[:] = handle['dp'][indices, lo:lo+side, lo:lo+side]
            elif dataset == 'integrated_circuit':
                handle['dp'].read_direct(measured)
            else:
                # Two contiguous spans reproduce the all-but-one selection
                # without slow HDF5 advanced indexing or a float64 full copy.
                handle['dp'].read_direct(measured, np.s_[:1772], np.s_[:1772])
                handle['dp'].read_direct(measured, np.s_[1773:], np.s_[1772:])
            for start in range(0, len(measured), 32):
                chunk=measured[start:start+32]
                if not np.isfinite(chunk).all() or (chunk<0).any():
                    raise ValueError('Measured intensities must be finite and nonnegative.')
            dx=float(np.asarray(handle['dx'][()]).squeeze())
            wavelength=float(np.asarray(handle['lambda'][()]).squeeze())
            x=np.asarray(handle['ppX'][indices], dtype=np.float32)
            y=np.asarray(handle['ppY'][indices], dtype=np.float32)
        positions=np.stack([y/dx, x/dx], axis=1).astype(np.float32)
        positions-=positions.mean(axis=0, keepdims=True)
        if smoke:
            positions *= np.float32(32 / max(float(np.ptp(positions, axis=0).max()), 1.))
        probe=generate_probe(side,wavelength,dx,spec['defocus_m'])
        mean_data=np.mean(measured,axis=0,dtype=np.float32)
        probe_norm=float(np.sqrt(np.sum(abs(probe)**2,dtype=np.float32)))
        if not np.isfinite(probe_norm) or probe_norm<=0:raise ValueError('Invalid generated initial probe.')
        scale=np.sqrt(np.sum(mean_data))/float(side)
        probe=(probe*(scale/probe_norm)).astype(np.complex64)
        shape=tuple(int(np.ceil(np.ptp(positions[:,axis])))+side+20 for axis in (0,1))
        generator=torch.Generator(device='cpu').manual_seed(42)
        real=torch.randn((1,*shape),generator=generator,dtype=torch.float32,device='cpu')
        imag=torch.randn((1,*shape),generator=generator,dtype=torch.float32,device='cpu')
        initial=(-1+torch.complex(real,imag)*(.01*2**-.5)).numpy()[0]
        values=dict(data=measured,positions_px=positions,probe_init=probe,
                    object_init=initial,selected_indices=indices)
        hashes={name:array_sha256(value) for name,value in values.items()}
        measured.flush();del values['data'];del measured
        os.replace(temporary,directory/'data.npy')
        for name,value in values.items():_save_npy(directory/f'{name}.npy',value)
        metadata=dict(identity=identity,fingerprint=fingerprint,array_sha256=hashes,
            dataset=dataset,smoke=smoke,scientific=not smoke,dx_m=dx,wavelength_m=wavelength,
            object_shape=list(shape),probe_shape=[side,side],num_patterns=len(indices),
            position_refinement=dict(step_size=1.,update_cap_px=50.,clip_by_mad=False,
                                     start_epoch=0,stride=1,constrain_mean=True),
            native_option_defaults=dict(lsqml_preconditioning_damping_factor=.1,
                                       lsqml_optimal_step_size_scaler=.5,lsqml_sigma=.5))
        _atomic_json(directory/'metadata.json',metadata)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def load_or_make_real_data(root, dataset, *, smoke=False):
    if dataset not in DATASETS:raise ValueError(f'Choose dataset from {DATASETS}.')
    root=Path(root).resolve()
    source=root/'assets/ptycho_real_data'/SPECS[dataset]['filename']
    if not source.is_file():raise FileNotFoundError(source)
    identity,fingerprint=_identity(dataset,source,bool(smoke))
    base=root/'evaluations/real/results'
    if smoke:base/= 'smoke'
    directory=base/'datasets'/dataset
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.dataset.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        path=directory/'metadata.json'
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['fingerprint']!=fingerprint or saved['identity']!=identity:
                raise ValueError(f'Incompatible real-data cache at {directory}; existing cache retained.')
        else:_generate(directory,source,dataset,identity,fingerprint,bool(smoke))
        metadata=json.loads(path.read_text())
        arrays={name:np.load(directory/f'{name}.npy',mmap_mode='r',allow_pickle=False) for name in ARRAYS}
        for name,value in arrays.items():
            if array_sha256(value)!=metadata['array_sha256'][name]:
                raise ValueError(f'Cached real array failed its hash check: {name}')
    return RealData(dataset=dataset,metadata=metadata,path=directory,
        dx_m=metadata['dx_m'],wavelength_m=metadata['wavelength_m'],**arrays)


def make_real_options(method, data, *, device='cuda', smoke=False):
    import ptychi.api as api
    if method not in METHODS:raise ValueError(f'Unknown method {method}.')
    if device not in ('cuda','cpu'):raise ValueError('Device must be cuda or cpu.')
    if bool(smoke)!=data.metadata['smoke']:raise ValueError('Data/option smoke settings differ.')
    factories={'rpie':api.RPIEOptions,'lsqml':api.LSQMLOptions,
               'gm_rpie':api.MAGPIEOptions,'gm_magpie':api.MAGPIEOptions}
    options=factories[method]()
    options.data_options.fft_shift=True
    # Native save_data_on_device selects the accelerator independently of the
    # reconstructor default device; CPU diagnostics must retain CPU data.
    options.data_options.save_data_on_device=device=='cuda'
    options.data_options.wavelength_m=data.wavelength_m
    options.object_options.pixel_size_m=data.dx_m
    options.probe_options.pixel_size_m=data.dx_m
    options.object_options.determine_position_origin_coords_by=api.ObjectPosOriginCoordsMethods.POSITIONS
    options.object_options.patch_interpolation_method=api.PatchInterpolationMethods.FOURIER
    options.object_options.remove_object_probe_ambiguity.enabled=True
    options.object_options.remove_object_probe_ambiguity.optimization_plan.stride=1
    for field in (options.object_options,options.probe_options,options.probe_position_options):
        field.optimizable=True
        field.optimizer=api.Optimizers.SGD
        field.optimizer_params={}
        field.step_size=1.
        field.optimization_plan.start=0
        field.optimization_plan.stop=None
        field.optimization_plan.stride=1
        field.optimization_plan.step_size_scheduler_class=None
        field.optimization_plan.step_size_scheduler_options={}
    probe=options.probe_options
    for feature in (probe.power_constraint,probe.center_constraint,probe.support_constraint,
                    probe.orthogonalize_incoherent_modes,probe.orthogonalize_opr_modes):feature.enabled=False
    options.opr_mode_weight_options.optimizable=False
    position=options.probe_position_options
    position.constrain_position_mean=True
    position.affine_transform_constraint.enabled=False
    correction=position.correction_options
    correction.correction_type=api.PositionCorrectionTypes.GRADIENT
    correction.differentiation_method=api.ImageGradientMethods.FOURIER_DIFFERENTIATION
    correction.clip_update_magnitude_by_mad=False
    correction.update_magnitude_limit=50.
    rec=options.reconstructor_options
    rec.default_device=api.Devices.GPU if device=='cuda' else api.Devices.CPU
    rec.default_dtype=api.Dtypes.FLOAT32
    rec.allow_nondeterministic_algorithms=False
    rec.use_double_precision_for_fft=False
    rec.num_epochs=2 if smoke else 100
    rec.batch_size=8 if smoke else SPECS[data.dataset]['batch_size']
    rec.batching_mode=api.BatchingModes.RANDOM
    rec.random_seed=42
    rec.displayed_loss_function=api.LossFunctions.MSE_SQRT
    rec.forward_model_options.pad_for_shift=1
    parameters=PARAMETERS[data.dataset][method]
    if method=='lsqml':
        rec.noise_model=api.NoiseModels.GAUSSIAN
        rec.gaussian_noise_std=.5
        rec.preconditioning_damping_factor=.1
        rec.rescale_probe_intensity_in_first_epoch=False
        options.object_options.optimal_step_size_scaler=.5
        options.probe_options.optimal_step_size_scaler=.5
    else:
        options.object_options.alpha=parameters['object_alpha']
        options.probe_options.alpha=parameters['probe_alpha']
        if method.startswith('gm_'):
            rec.multigrid_levels=1 if method=='gm_rpie' else int(np.log2(data.probe_init.shape[-1]))+1
    return options
