"""Native reconstruction, checkpoint scoring, and portable per-run records.

The numerical updates are the pinned upstream implementations. Online losses
reuse pre-update predictions; the paper's frozen residual is a separate final
evaluation with no updates. Only score checkpoints transfer full fields to CPU.
"""
from __future__ import annotations

import csv
import fcntl
import gc
import hashlib
import json
import logging
import os
import platform
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
import ptychi
import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.timing import timer_utils

from .data import DataConfig, load_or_make_data, score
from .protocol import (BASE_BATCH, BASE_ETA, EPOCHS, METHODS, PTYCHI_COMMIT,
                       PTYCHI_REPOSITORY, case_token, float_token, run_id)

SCHEMA_VERSION = 1
NATIVE_FILES = ("reconstructors/base.py", "reconstructors/pie.py",
                "reconstructors/lsqml.py", "reconstructors/magpie.py",
                "forward_models.py", "image_proc.py", "maths.py")


def write_json(path, value):
    """Replace one JSON record atomically, rejecting non-finite numbers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def results_root(root, smoke=False):
    return Path(root).resolve() / "results" / ("smoke" if smoke else "")


def relative_run_directory(method, overlap_ratio, eta, batch_size, parameters):
    condition = f"{case_token(overlap_ratio)}_eta_{float_token(eta)}_b{batch_size}"
    return Path("runs") / method / condition / run_id(method, parameters)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_native_source():
    """Require the exact upstream revision before generating any run data."""
    package = Path(ptychi.__file__).resolve().parent
    checkout = package.parent.parent
    commit = None
    if (checkout / ".git").exists():
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    else:
        try:
            direct_url = json.loads(metadata.distribution("ptychi").read_text("direct_url.json") or "{}")
            commit = direct_url.get("vcs_info", {}).get("commit_id")
        except metadata.PackageNotFoundError:
            pass
    if commit != PTYCHI_COMMIT:
        raise RuntimeError(f"Expected Pty-Chi {PTYCHI_COMMIT}; found {commit}. Run setup_conda.sh.")
    identity = {"commit": commit, "source_sha256": {
        name: _sha(package / name) for name in NATIVE_FILES}}
    # A Git checkout with local modifications must not masquerade as its HEAD.
    if (checkout / ".git").exists():
        for name in NATIVE_FILES:
            canonical = subprocess.check_output(
                ["git", "show", f"{PTYCHI_COMMIT}:src/ptychi/{name}"], cwd=checkout)
            if hashlib.sha256(canonical).hexdigest() != identity["source_sha256"][name]:
                raise RuntimeError(f"Pinned native source has local modifications: {name}")
    return identity


def _runner_source():
    directory = Path(__file__).resolve().parent
    identity = {name: _sha(directory / name) for name in
                ("engine.py", "overlap.py", "cli.py", "data.py", "protocol.py")}
    identity["selected_parameters.json"] = _sha(
        directory.parents[1] / "provenance" / "selected_parameters.json")
    return identity


def make_options(method, parameters, data, *, epochs, batch_size, device, seed=21):
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


class OnlineAmplitudeMSE:
    """Pixel-weighted online diagnostic from predictions already computed."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.squared_sum = None
        self.count = 0

    def __call__(self, predicted, measured):
        with torch.no_grad():
            delta = torch.sqrt(predicted.clamp_min(0) + 1e-7) - torch.sqrt(measured.clamp_min(0) + 1e-7)
            value = delta.square().sum()
            self.squared_sum = value if self.squared_sum is None else self.squared_sum + value
            self.count += delta.numel()
            return value / delta.numel()

    def value(self):
        if not self.count:
            raise RuntimeError("The native loss tracker did not receive predictions.")
        return float((self.squared_sum / self.count).item())


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def frozen_residual(task, data, batch_size=128):
    """Paper amplitude MSE on fixed, unregistered final fields, no updates.

    The native model emits unshifted detector intensities. Data files retain
    centered detector indexing, hence fftshift here. No truth registration is
    applied before this diagnostic.
    """
    device = task.get_data("object").device
    noisy_sum = torch.zeros((), dtype=torch.float64, device=device)
    noiseless_sum = noisy_sum.clone()
    count = 0
    with torch.no_grad():
        for start in range(0, len(data.intensities), batch_size):
            stop = min(start + batch_size, len(data.intensities))
            indices = torch.arange(start, stop, device=device)
            predicted = task.reconstructor.forward_model.forward(indices)
            predicted = torch.fft.fftshift(predicted, dim=(-2, -1))
            amplitude = torch.sqrt(predicted.clamp_min(0) + 1e-7)
            for array, accumulator in ((data.intensities, noisy_sum),
                                       (data.noiseless_intensities, noiseless_sum)):
                target = torch.as_tensor(array[start:stop], device=device)
                accumulator.add_((amplitude - torch.sqrt(target.clamp_min(0) + 1e-7)).square().sum(dtype=torch.float64))
            count += predicted.numel()
    return {"frozen_amplitude_mse": float((noisy_sum / count).item()),
            "frozen_noiseless_amplitude_mse": float((noiseless_sum / count).item())}


def _environment(device):
    versions = {}
    for name in ("torch", "numpy", "pandas", "scipy", "ptychi", "pydantic"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "source checkout"
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "ptychi_file": str(Path(ptychi.__file__).resolve()),
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
            "cpu_threads": torch.get_num_threads(), "native_timing_enabled": timer_utils.ENABLE_TIMING,
            "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING"),
            "PTYCHI_USE_TRITON": os.environ.get("PTYCHI_USE_TRITON")}


@contextmanager
def _run_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is running this parameter pair: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _compatible(saved, requested):
    return ({k: v for k, v in saved.items() if k != "print_every"}
            == {k: v for k, v in requested.items() if k != "print_every"})


def _complete_files(run_dir, row, epochs, error_every):
    if row.get("status") != "complete" or row.get("epochs_completed") != epochs:
        return False
    if not (run_dir / "reconstruction.npz").is_file():
        return False
    metric_names = ("object_nmse", "probe_nmse", "phase_rmse", "online_amplitude_mse",
                    "frozen_amplitude_mse", "frozen_noiseless_amplitude_mse")
    if any(not isinstance(row.get(k), (int, float)) or not np.isfinite(row[k]) or row[k] < 0 for k in metric_names):
        return False
    try:
        with (run_dir / "history.csv").open(newline="") as stream:
            history = list(csv.DictReader(stream))
        if [int(r["epoch"]) for r in history] != list(range(epochs + 1)):
            return False
        for r in history:
            epoch = int(r["epoch"])
            if epoch == 0 or epoch % error_every == 0 or epoch == epochs:
                if any(not np.isfinite(float(r[k])) for k in ("object_nmse", "probe_nmse", "phase_rmse")):
                    return False
        with np.load(run_dir / "reconstruction.npz") as archive:
            if not {"object", "probe", "object_aligned", "probe_aligned", "roi_mask"}.issubset(archive.files):
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def _save_fields(path, obj, probe, aligned_obj, aligned_probe, mask):
    temporary = path.with_name("." + path.name + ".tmp.npz")
    try:
        np.savez_compressed(temporary, object=obj, probe=probe,
                            object_aligned=aligned_obj, probe_aligned=aligned_probe, roi_mask=mask)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_pair(method, parameters, *, root, overlap_ratio=.75, eta=BASE_ETA,
             batch_size=BASE_BATCH, epochs=None, device="cuda", smoke=False,
             seed=21, error_every=20, print_every=300, rerun_failed=False):
    """Run or reuse one exact native reconstruction; interrupted runs restart.

    Failed runs are retained unless explicitly rerun. Any incompatible config
    raises without overwriting old scientific results. Smoke outputs are kept
    under results/smoke and never qualify as scientific results.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}.")
    if device not in ("cuda", "cpu"):
        raise ValueError("device must be 'cuda' or 'cpu'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is visible; request a GPU node or use CPU for smoke tests.")
    epochs = EPOCHS[overlap_ratio] if epochs is None else epochs
    for name, value in (("epochs", epochs), ("batch_size", batch_size),
                        ("error_every", error_every), ("print_every", print_every)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not parameters or any(not np.isfinite(v) or v <= 0 for v in parameters.values()):
        raise ValueError("Parameters must be positive and finite")
    source = check_native_source()
    epochs = min(int(epochs), 2) if smoke else int(epochs)
    root = Path(root).resolve()
    destination_root = results_root(root, smoke)
    relative = relative_run_directory(method, overlap_ratio, eta, batch_size, parameters)
    run_dir = destination_root / relative
    # Timing and native progress are disabled for every method identically.
    logging.getLogger("ptychi").setLevel(logging.WARNING)
    torch.set_num_threads(1)
    previous_timing = timer_utils.ENABLE_TIMING
    timer_utils.toggle_timer(False)
    try:
        with _run_lock(run_dir.parent / ("." + run_dir.name + ".lock")):
            data = load_or_make_data(root, DataConfig(overlap_ratio=overlap_ratio, eta=eta, smoke=smoke))
            data_path = Path(data.path)
            if not data_path.is_absolute():
                data_path = root / data_path
            config = {"schema_version": SCHEMA_VERSION, "method": method,
                      "parameters": dict(parameters), "epochs": epochs,
                      "batch_size": int(batch_size), "device": device, "seed": int(seed),
                      "overlap_ratio": float(overlap_ratio), "eta": float(eta),
                      "dataset_fingerprint": data.fingerprint,
                      "dataset_array_sha256": data.metadata["array_sha256"],
                      "data_reference": os.path.relpath(data_path, root),
                      "native_source": source, "runner_source_sha256": _runner_source(),
                      "ptychi_repository": PTYCHI_REPOSITORY,
                      "torch_version": torch.__version__, "smoke": bool(smoke),
                      "scientific": not smoke, "error_every": int(error_every),
                      "print_every": int(print_every), "step_size": 1.0,
                      "object_error": "complex NMSE after ROI-only affine phase-ramp and scalar alignment",
                      "roi": "fixed data.roi_mask, identical for all scientific conditions",
                      "online_residual": "pixel-weighted amplitude MSE of pre-update minibatch predictions",
                      "frozen_residual": "final native forward-model amplitude MSE before truth registration; epsilon=1e-7",
                      "amplitude_normalization": "native object/probe normalization every epoch",
                      "multigrid_levels": int(np.log2(data.probe_init.shape[-1])) + 1 if method == "gm_magpie" else 1}
            config_path = run_dir / "config.json"
            result_path = run_dir / "result.json"
            if config_path.exists() and not _compatible(json.loads(config_path.read_text()), config):
                raise ValueError(f"Incompatible saved settings or source at {run_dir}. Use a new project/results directory.")
            if result_path.exists():
                saved = json.loads(result_path.read_text())
                if not config_path.exists():
                    raise ValueError(f"Saved result has no configuration: {run_dir}")
                if saved.get("status") == "failed" and not rerun_failed:
                    print(f"{method} {run_dir.name}: retaining recorded failure", flush=True)
                    return saved
                if _complete_files(run_dir, saved, epochs, error_every):
                    print(f"{method} {run_dir.name}: using completed result", flush=True)
                    return saved
            if run_dir.exists() and any(run_dir.iterdir()):
                archive = run_dir.parent / "_previous_runs"
                archive.mkdir(exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                run_dir.rename(archive / f"{run_dir.name}_{stamp}")
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(config_path, config)
            environment = _environment(device)
            environment["native_timing_was_enabled"] = previous_timing
            write_json(run_dir / "environment.json", environment)
            row = {"run_id": run_id(method, parameters), "method": method, "algorithm": method,
                   "overlap_ratio": float(overlap_ratio), "eta": float(eta),
                   "batch_size": int(batch_size), **parameters, "epochs": epochs,
                   "epochs_completed": 0, "status": "running", "seed": int(seed),
                   "scientific": not smoke, "relative_run_dir": relative.as_posix(),
                   "dataset_fingerprint": data.fingerprint}
            write_json(result_path, row)
            task = None
            totals = {key: 0.0 for key in ("reconstruction_seconds", "transfer_seconds",
                      "scoring_seconds", "frozen_evaluation_seconds", "history_write_seconds")}
            started_run = time.perf_counter()
            epoch = 0
            try:
                options = make_options(method, parameters, data, epochs=epochs,
                                       batch_size=batch_size, device=device, seed=seed)
                task = PtychographyTask(options, diffraction_data=data.intensities,
                    object_data=data.object_init[None].copy(), probe_data=data.probe_init[None, None].copy(),
                    probe_position_x_px=data.positions_px[:, 1].copy(),
                    probe_position_y_px=data.positions_px[:, 0].copy())
                task.reconstructor.dataloader.generator.manual_seed(seed)
                collector = OnlineAmplitudeMSE()
                task.reconstructor.loss_tracker.metric_function = collector
                columns = ["epoch", "online_amplitude_mse", "object_nmse", "probe_nmse", "phase_rmse",
                           "errors_recorded", "reconstruction_seconds", "transfer_seconds",
                           "scoring_seconds", "patterns_processed"]
                with (run_dir / "history.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=columns)
                    writer.writeheader()
                    start = time.perf_counter()
                    initial, _, _ = score(data.object_init, data.probe_init, data)
                    totals["scoring_seconds"] += time.perf_counter() - start
                    writer.writerow({"epoch": 0, "online_amplitude_mse": "", "errors_recorded": True,
                                     "reconstruction_seconds": 0., "transfer_seconds": 0.,
                                     "scoring_seconds": totals["scoring_seconds"], "patterns_processed": 0, **initial})
                    stream.flush()
                    for epoch in range(1, epochs + 1):
                        collector.reset()
                        _sync(device)
                        start = time.perf_counter()
                        task.run(n_epochs=1)
                        _sync(device)
                        reconstruction_time = time.perf_counter() - start
                        totals["reconstruction_seconds"] += reconstruction_time
                        online = collector.value()
                        if not np.isfinite(online):
                            raise FloatingPointError(f"Non-finite online amplitude MSE at epoch {epoch}")
                        record_errors = epoch % error_every == 0 or epoch == epochs
                        metrics = {"object_nmse": "", "probe_nmse": "", "phase_rmse": ""}
                        transfer_time = scoring_time = 0.0
                        if record_errors:
                            # Check on device before transferring only at score checkpoints.
                            if not (torch.isfinite(task.get_data("object")).all().item()
                                    and torch.isfinite(task.get_data("probe")).all().item()):
                                raise FloatingPointError(f"Non-finite fields at epoch {epoch}")
                            start = time.perf_counter()
                            obj = task.get_data_to_cpu("object", as_numpy=True).squeeze()
                            probe = task.get_data_to_cpu("probe", as_numpy=True).squeeze()
                            transfer_time = time.perf_counter() - start
                            totals["transfer_seconds"] += transfer_time
                            start = time.perf_counter()
                            metrics, aligned_obj, aligned_probe = score(obj, probe, data)
                            scoring_time = time.perf_counter() - start
                            totals["scoring_seconds"] += scoring_time
                            if not all(np.isfinite(v) and v >= 0 for v in metrics.values()):
                                raise FloatingPointError(f"Non-finite truth metrics at epoch {epoch}")
                        start = time.perf_counter()
                        writer.writerow({"epoch": epoch, "online_amplitude_mse": online,
                            "errors_recorded": record_errors, "reconstruction_seconds": reconstruction_time,
                            "transfer_seconds": transfer_time, "scoring_seconds": scoring_time,
                            "patterns_processed": epoch * len(data.intensities), **metrics})
                        stream.flush()
                        totals["history_write_seconds"] += time.perf_counter() - start
                        row["epochs_completed"] = epoch
                        if epoch % print_every == 0 or epoch == epochs:
                            suffix = f"; object NMSE={metrics['object_nmse']:.5g}" if record_errors else ""
                            print(f"{method} {run_dir.name}: epoch {epoch}/{epochs}; online MSE={online:.5g}{suffix}", flush=True)
                            write_json(result_path, {**row, **totals})
                start = time.perf_counter()
                frozen = frozen_residual(task, data)
                _sync(device)
                totals["frozen_evaluation_seconds"] += time.perf_counter() - start
                if not all(np.isfinite(v) and v >= 0 for v in frozen.values()):
                    raise FloatingPointError("Non-finite frozen final residual")
                _save_fields(run_dir / "reconstruction.npz", obj, probe, aligned_obj, aligned_probe, data.roi_mask)
                row.update(status="complete", online_amplitude_mse=online, **metrics, **frozen,
                           **totals, patterns_processed=epochs * len(data.intensities),
                           total_seconds=time.perf_counter() - started_run)
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}", failure_epoch=epoch,
                           **totals, total_seconds=time.perf_counter() - started_run)
                print(f"{method} {run_dir.name}: FAILED {row['error']}", flush=True)
            finally:
                if task is not None:
                    task.reconstructor.pbar.close()
                del task
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
            write_json(result_path, row)
            return row
    finally:
        timer_utils.toggle_timer(previous_timing)
