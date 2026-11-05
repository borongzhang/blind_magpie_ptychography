"""Run the four synthetic methods through the pinned Pty-Chi implementation.

Importing this module does not import Torch or Pty-Chi. Reconstruction dependencies
are loaded only when a run or native option construction is requested.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import fcntl
import gc
import hashlib
from importlib import metadata
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
from zipfile import BadZipFile

import numpy as np

from .protocol import EPOCHS, METHODS, PTYCHI_COMMIT, case_token, float_token, run_id

NATIVE_HASHES = {'forward_models.py': '3672327ee6034bfa304fd01ad06f61a9f347d617fb8f8aacaa96074ae711a7e8',
 'image_proc.py': '2fe83441985601fe3a88d8b436bd7a32c26c0c444f207cf7b1c0fc0be412085f',
 'maths.py': '660b44ef32f480f0d33e67377cc6906d68fc75df6f32f52bea2f979dc9aca749',
 'reconstructors/base.py': 'c2126a42568fc2d97c7ea8ea26ce803a82092685bc11a21fd7dd33d65e7ab408',
 'reconstructors/lsqml.py': '1296ae6c9300344e47b73e4e89a7fea5fd573f89e375758807135dd0219d5b14',
 'reconstructors/magpie.py': '366d090d1b4ce8a55cdb0110c5f07c5623b578c461b4ad6814095d3041da7dab',
 'reconstructors/pie.py': 'd5c068d87bfb3faebda5d222efc90a2d0a287a7f81386fe523e1b9f59b256ad7'}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_native_source():
    """Require the recorded upstream commit and unmodified numerical sources."""
    import ptychi
    package = Path(ptychi.__file__).resolve().parent
    checkout = package.parent.parent
    if (checkout / ".git").exists():
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    else:
        direct_url = json.loads(metadata.distribution("ptychi").read_text("direct_url.json") or "{}")
        commit = direct_url.get("vcs_info", {}).get("commit_id")
    actual = {name: _sha(package / name) for name in NATIVE_HASHES}
    if commit != PTYCHI_COMMIT or actual != NATIVE_HASHES:
        raise RuntimeError(f"Install the unmodified Pty-Chi commit {PTYCHI_COMMIT} using the reconstruction extra.")
    return {"commit": commit, "source_sha256": actual}


def make_options(method, parameters, data, *, epochs, batch_size, device, seed=21):
    import ptychi.api as api
    factories = {"rpie": api.RPIEOptions, "lsqml": api.LSQMLOptions,
                 "gm_rpie": api.MAGPIEOptions, "gm_magpie": api.MAGPIEOptions}
    options = factories[method]()
    options.data_options.fft_shift = True
    options.data_options.save_data_on_device = False
    pixel_size = getattr(data.config, "sample_pixel_size_m", 40e-9)
    options.object_options.pixel_size_m = pixel_size
    options.probe_options.pixel_size_m = pixel_size
    options.object_options.determine_position_origin_coords_by = api.ObjectPosOriginCoordsMethods.SUPPORT
    options.object_options.patch_interpolation_method = api.PatchInterpolationMethods.FOURIER
    options.object_options.remove_object_probe_ambiguity.enabled = True
    options.object_options.remove_object_probe_ambiguity.optimization_plan.stride = 1
    options.probe_position_options.optimizable = False
    for field in (options.object_options, options.probe_options):
        field.optimizable = True
        field.optimizer = api.Optimizers.SGD
        field.step_size = 1.0
        field.optimization_plan.start = 0
        field.optimization_plan.stop = None
        field.optimization_plan.stride = 1
        field.optimization_plan.step_size_scheduler_class = None
        field.optimization_plan.step_size_scheduler_options = {}
    options.probe_options.power_constraint.enabled = False
    options.probe_options.center_constraint.enabled = False
    options.probe_options.support_constraint.enabled = False
    rec = options.reconstructor_options
    rec.default_device = api.Devices.GPU if device == "cuda" else api.Devices.CPU
    rec.default_dtype = api.Dtypes.FLOAT32
    rec.use_double_precision_for_fft = False
    rec.allow_nondeterministic_algorithms = False
    rec.num_epochs = epochs
    rec.batch_size = batch_size
    rec.batching_mode = api.BatchingModes.RANDOM
    rec.random_seed = seed
    rec.displayed_loss_function = api.LossFunctions.MSE_SQRT
    rec.forward_model_options.pad_for_shift = 1
    if method == "lsqml":
        rec.noise_model = api.NoiseModels.GAUSSIAN
        rec.preconditioning_damping_factor = parameters["damping"]
        rec.gaussian_noise_std = parameters["sigma"]
        rec.rescale_probe_intensity_in_first_epoch = False
        rec.solve_step_sizes_only_using_first_probe_mode = False
        options.object_options.optimal_step_size_scaler = 1.0
        options.probe_options.optimal_step_size_scaler = 1.0
    else:
        options.object_options.alpha = parameters["object_alpha"]
        options.probe_options.alpha = parameters["probe_alpha"]
        if method.startswith("gm_"):
            rec.multigrid_levels = 1 if method == "gm_rpie" else int(np.log2(data.probe_init.shape[-1])) + 1
    return options


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"This case is already running: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _data(output, overlap, eta, seed, smoke):
    if overlap == .9375:
        from .high_overlap_data import HighOverlapDataConfig, load_or_make_data
        config = HighOverlapDataConfig(overlap_ratio=overlap, eta=eta, seed=seed, smoke=smoke)
    else:
        from .data import DataConfig, load_or_make_data
        config = DataConfig(overlap_ratio=overlap, eta=eta, seed=seed, smoke=smoke)
    return load_or_make_data(output, config)


def _complete(directory, epochs, error_every):
    """Check a completed case before reusing it."""
    try:
        result_file = directory / "result.json"
        if not result_file.is_file() or not (directory / "reconstruction.npz").is_file():
            return None
        result = json.loads(result_file.read_text())
        if (not isinstance(result, dict) or result.get("status") != "complete"
                or result.get("epochs_completed") != epochs):
            return None
        with (directory / "history.csv").open(newline="") as stream:
            points = list(csv.DictReader(stream))
        expected = sorted({0, epochs, *range(error_every, epochs + 1, error_every)})
        if [int(point["epoch"]) for point in points] != expected:
            return None
        for point in points:
            if any(not np.isfinite(float(point[key])) or float(point[key]) < 0
                   for key in ("object_nmse", "probe_nmse")):
                return None
        if any(float(points[-1][key]) != result[key] for key in ("object_nmse", "probe_nmse")):
            return None
        with np.load(directory / "reconstruction.npz", allow_pickle=False) as fields:
            required = {"object", "probe", "object_aligned", "probe_aligned", "roi_mask"}
            if not required.issubset(fields.files):
                return None
            if any(not np.isfinite(fields[key]).all() for key in required):
                return None
    except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError, BadZipFile, csv.Error):
        return None
    return result


class _ValidityMetric:
    """Use the original detached diagnostic to detect a failed native iteration."""
    def __init__(self, torch):
        self.torch = torch
        self.reset()

    def reset(self):
        self.squared_sum = None
        self.count = 0

    def __call__(self, predicted, measured):
        with self.torch.no_grad():
            delta = self.torch.sqrt(predicted.clamp_min(0) + 1e-7) - self.torch.sqrt(measured.clamp_min(0) + 1e-7)
            value = delta.square().sum()
            self.squared_sum = value if self.squared_sum is None else self.squared_sum + value
            self.count += delta.numel()
            return value / delta.numel()

    def value(self):
        if not self.count:
            raise RuntimeError("The native loss tracker did not receive predictions.")
        return float((self.squared_sum / self.count).item())


def run_case(method, overlap, eta, batch_size, parameters, *, epochs=None,
             output="runs", device="cuda", seed=21, data_seed=42,
             error_every=20, print_every=300, smoke=False):
    """Run one synthetic case or reuse its completed result.

    Parameters select a single pair; ``protocol.parameter_grid`` supplies the
    paper's search grids. Interrupted cases restart from the original fields.
    A native numerical failure is recorded and returned with ``status='failed'``.
    Smoke runs use separate small diagnostic arrays and at most two epochs.
    """
    overlap = float(overlap)
    case_token(overlap)
    eta = float(eta)
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative.")
    parameters = dict(parameters)
    pair = run_id(method, parameters)
    epochs = EPOCHS[overlap] if epochs is None else epochs
    for name, value in (("epochs", epochs), ("batch_size", batch_size),
                        ("error_every", error_every), ("print_every", print_every)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    for name, value in (("seed", seed), ("data_seed", data_seed)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'.")
    try:
        import torch
        from ptychi.api.task import PtychographyTask
        from ptychi.timing import timer_utils
    except ImportError as error:
        raise RuntimeError("Install the reconstruction extra to run synthetic cases.") from error
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is visible. Use device='cpu' for a small smoke run.")
    native = check_native_source()
    epochs = min(int(epochs), 2) if smoke else int(epochs)
    output = Path(output).expanduser().resolve()
    base = output / "smoke" if smoke else output
    condition = f"{case_token(overlap)}_eta_{float_token(eta)}_b{batch_size}"
    relative = Path("cases") / condition / method / pair
    directory = base / relative
    from .data import score
    logging.getLogger("ptychi").setLevel(logging.WARNING)
    torch.set_num_threads(1)
    with _lock(directory.parent / f".{pair}.lock"):
        data = _data(output, overlap, eta, int(data_seed), smoke)
        config = dict(method=method, overlap_ratio=overlap, eta=float(eta),
                      batch_size=int(batch_size), parameters=parameters, epochs=epochs,
                      seed=int(seed), data_seed=int(data_seed), device=device,
                      error_every=int(error_every), smoke=bool(smoke), scientific=not smoke,
                      native_source=native, dataset_array_sha256=data.metadata["array_sha256"],
                      dataset_fingerprint=data.fingerprint,
                      source_sha256={name: _sha(Path(__file__).parent / name) for name in (
                          "data.py", "high_overlap_data.py", "protocol.py", "reconstruct.py")})
        config_path = directory / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError(f"This output contains a different configuration: {directory}. Choose another output folder.")
        if config_path.exists():
            complete = _complete(directory, epochs, error_every)
            if complete is not None:
                print(f"{method}: using completed result at {directory}", flush=True)
                return complete
        _write_json(config_path, config)
        row = dict(method=method, overlap_ratio=overlap, eta=float(eta),
                   batch_size=int(batch_size), parameters=parameters, epochs=epochs,
                   seed=int(seed), data_seed=int(data_seed), scientific=not smoke,
                   status="running", epochs_completed=0, num_patterns=len(data.intensities),
                   relative_run_dir=relative.as_posix())
        _write_json(directory / "result.json", row)
        previous_timing = timer_utils.ENABLE_TIMING
        timer_utils.toggle_timer(False)
        task = None
        epoch = 0
        try:
            options = make_options(method, parameters, data, epochs=epochs,
                                   batch_size=batch_size, device=device, seed=seed)
            task = PtychographyTask(options, diffraction_data=data.intensities,
                object_data=data.object_init[None].copy(), probe_data=data.probe_init[None, None].copy(),
                probe_position_x_px=data.positions_px[:, 1].copy(),
                probe_position_y_px=data.positions_px[:, 0].copy())
            task.reconstructor.dataloader.generator.manual_seed(seed)
            collector = _ValidityMetric(torch)
            task.reconstructor.loss_tracker.metric_function = collector
            with (directory / "history.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("epoch", "object_nmse", "probe_nmse"))
                writer.writeheader()
                initial, _, _ = score(data.object_init, data.probe_init, data)
                writer.writerow({"epoch": 0, **{key: initial[key] for key in ("object_nmse", "probe_nmse")}})
                stream.flush()
                for epoch in range(1, epochs + 1):
                    collector.reset()
                    task.run(n_epochs=1)
                    if not np.isfinite(collector.value()):
                        raise FloatingPointError(f"Nonfinite native iteration at epoch {epoch}.")
                    if epoch % error_every == 0 or epoch == epochs:
                        if not (torch.isfinite(task.get_data("object")).all().item()
                                and torch.isfinite(task.get_data("probe")).all().item()):
                            raise FloatingPointError(f"Nonfinite fields at epoch {epoch}.")
                        obj = task.get_data_to_cpu("object", as_numpy=True).squeeze()
                        probe = task.get_data_to_cpu("probe", as_numpy=True).squeeze()
                        metrics, aligned_obj, aligned_probe = score(obj, probe, data)
                        metrics = {key: metrics[key] for key in ("object_nmse", "probe_nmse")}
                        if any(not np.isfinite(value) or value < 0 for value in metrics.values()):
                            raise FloatingPointError(f"Nonfinite reconstruction error at epoch {epoch}.")
                        writer.writerow({"epoch": epoch, **metrics})
                        stream.flush()
                        row.update(metrics)
                    row["epochs_completed"] = epoch
                    if epoch % print_every == 0 or epoch == epochs:
                        print(f"{method} {condition}: epoch {epoch}/{epochs}", flush=True)
                        _write_json(directory / "result.json", row)
            temporary = directory / ".reconstruction.tmp.npz"
            np.savez_compressed(temporary, object=obj, probe=probe,
                                object_aligned=aligned_obj, probe_aligned=aligned_probe,
                                roi_mask=data.roi_mask)
            os.replace(temporary, directory / "reconstruction.npz")
            row["status"] = "complete"
        except Exception as error:
            if task is None or isinstance(error, (ImportError, OSError, MemoryError,
                                                  torch.cuda.OutOfMemoryError)):
                raise
            row.update(status="failed", failure_epoch=epoch,
                       error=f"{type(error).__name__}: {error}")
            print(f"{method} {condition}: {row['error']}", flush=True)
        finally:
            if task is not None:
                task.reconstructor.pbar.close()
            del task
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            timer_utils.toggle_timer(previous_timing)
        _write_json(directory / "result.json", row)
        return row
