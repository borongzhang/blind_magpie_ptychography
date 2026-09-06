from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from numbers import Integral
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
import tempfile
from typing import Any

import numpy as np

from utils.result_archive import (
    _SAFE_RUN_ID,
    _atomic_savez,
    _atomic_write_text,
    _canonical_json,
    _git_output,
    _notebook_source_sha256,
    _pip_freeze,
    _sha256_array,
    _sha256_file,
    _software_versions,
    _write_csv,
    _write_json,
)


CANONICAL_REAL_METHODS = ("rPIE", "LSQML", "GM-rPIE", "GM-MAGPIE")
REAL_ARCHIVE_KIND = "magpie-real-data-comparison"
REAL_CHECKPOINT_KIND = "magpie-real-data-method-checkpoint"
REAL_SNAPSHOT_KIND = "magpie-real-data-epoch-snapshot"
REAL_ARCHIVE_SCHEMA_VERSION = 2
_LEGACY_FIXED_POSITION_CHECKPOINT_SCHEMA_VERSION = 1


def _finite_array(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError(f"{label} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains nonfinite values.")
    if array.dtype.hasobject:
        raise TypeError(f"{label} must not use object dtype.")
    return array


def _positive_finite(value: Any, *, label: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{label} must be positive and finite.")
    return numeric


def _relative_or_absolute(path: Path, root: Path) -> str:
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def _alignment_values(alignment: Any) -> dict[str, Any]:
    try:
        slope_y = float(alignment.slope_y_rad_per_px)
        slope_x = float(alignment.slope_x_rad_per_px)
        object_scale = complex(alignment.object_scale)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "alignment must expose slope_y_rad_per_px, "
            "slope_x_rad_per_px, and object_scale."
        ) from error
    values = np.asarray(
        [slope_y, slope_x, object_scale.real, object_scale.imag],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("alignment contains nonfinite values.")
    return {
        "slope_y_rad_per_px": slope_y,
        "slope_x_rad_per_px": slope_x,
        "object_scale": object_scale,
    }


def _expected_metric_epochs(
    num_epochs: int,
    report_every: int,
    include_initial: bool,
) -> np.ndarray:
    return np.asarray(
        sorted(
            {
                1,
                num_epochs,
                *range(report_every, num_epochs + 1, report_every),
                *((0,) if include_initial else ()),
            }
        ),
        dtype=np.int64,
    )


def _real_software_versions() -> dict[str, str | None]:
    versions = _software_versions()
    for label, candidates in {
        "blind-magpie-real-nvidia": ("blind-magpie-real-nvidia",),
        "h5py": ("h5py",),
        "torchvision": ("torchvision",),
    }.items():
        versions[label] = None
        for distribution in candidates:
            try:
                versions[label] = importlib_metadata.version(distribution)
                break
            except importlib_metadata.PackageNotFoundError:
                continue
    return versions


def _write_checksum_file(directory: Path) -> None:
    paths = [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.name not in {"checksums.sha256", "_SUCCESS"}
        and not path.name.startswith(".")
    ]
    lines = [
        f"{_sha256_file(path)}  {PurePosixPath(path.relative_to(directory)).as_posix()}"
        for path in paths
    ]
    _atomic_write_text(directory / "checksums.sha256", "\n".join(lines) + "\n")


def _read_checksum_file(directory: Path) -> dict[str, str]:
    checksum_path = directory / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Missing checksum file: {checksum_path}")
    records: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        parts = line.split("  ", maxsplit=1)
        if (
            len(parts) != 2
            or len(parts[0]) != 64
            or any(character not in "0123456789abcdef" for character in parts[0])
        ):
            raise ValueError(
                f"Malformed checksum record at {checksum_path}:{line_number}."
            )
        relative = PurePosixPath(parts[1])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe checksum path: {parts[1]!r}.")
        if parts[1] in records:
            raise ValueError(f"Duplicate checksum path: {parts[1]!r}.")
        records[parts[1]] = parts[0]
    return records


def _verify_checksum_file(directory: Path) -> None:
    records = _read_checksum_file(directory)
    expected_paths = {
        PurePosixPath(path.relative_to(directory)).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
        and path.name not in {"checksums.sha256", "_SUCCESS"}
        and not path.name.startswith(".")
    }
    if set(records) != expected_paths:
        missing = sorted(expected_paths - set(records))
        extra = sorted(set(records) - expected_paths)
        raise RuntimeError(
            f"Checksum inventory mismatch; missing={missing!r}, extra={extra!r}."
        )
    for relative, expected_hash in records.items():
        path = directory / relative
        if _sha256_file(path) != expected_hash:
            raise RuntimeError(f"Checksum verification failed for {path}.")


def _validate_npz(path: Path, required_keys: set[str]) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing NPZ file: {path}")
    with np.load(path, allow_pickle=False) as payload:
        keys = set(payload.files)
        missing = required_keys - keys
        if missing:
            raise ValueError(f"{path} is missing NPZ keys: {sorted(missing)!r}.")
        for key in payload.files:
            array = payload[key]
            if array.dtype.hasobject:
                raise TypeError(f"{path}:{key} uses object dtype.")
    return keys


def _snapshot_filename(epoch: int) -> str:
    return f"epoch_{epoch:06d}.npz"


def _snapshot_directory(
    output_root: Path,
    *,
    run_id: str,
    method_file_stem: str,
) -> Path:
    return output_root / "_checkpoints" / run_id / "_epoch_snapshots" / method_file_stem


def _validate_real_reconstruction_snapshot(
    path: Path,
    *,
    run_id: str,
    method: str,
    method_file_stem: str,
    epoch: int,
) -> dict[str, Any]:
    path = Path(path).resolve()
    keys = _validate_npz(
        path,
        {"object", "probe", "positions_px", "epoch", "metadata_json"},
    )
    if keys != {"object", "probe", "positions_px", "epoch", "metadata_json"}:
        raise ValueError(f"Unexpected arrays in reconstruction snapshot: {path}.")

    with np.load(path, allow_pickle=False) as payload:
        object_snapshot = _finite_array(payload["object"], label=f"{path}:object")
        probe_snapshot = _finite_array(payload["probe"], label=f"{path}:probe")
        positions_snapshot = _finite_array(
            payload["positions_px"],
            label=f"{path}:positions_px",
        )
        if not np.iscomplexobj(object_snapshot) or not np.iscomplexobj(probe_snapshot):
            raise TypeError("Reconstruction snapshots must be complex-valued.")
        if (
            np.iscomplexobj(positions_snapshot)
            or not np.issubdtype(positions_snapshot.dtype, np.floating)
            or positions_snapshot.ndim != 2
            or positions_snapshot.shape[1] != 2
        ):
            raise TypeError("Snapshot positions_px must be a real (patterns, 2) array.")
        saved_epoch = np.asarray(payload["epoch"])
        if saved_epoch.shape != () or not np.issubdtype(saved_epoch.dtype, np.integer):
            raise TypeError(f"{path}:epoch must be an integer scalar.")
        if int(saved_epoch) != epoch:
            raise ValueError(f"Reconstruction snapshot epoch mismatch for {path}.")
        metadata_value = np.asarray(payload["metadata_json"])
        if metadata_value.shape != () or metadata_value.dtype.kind not in {"U", "S"}:
            raise TypeError(f"{path}:metadata_json must be a string scalar.")
        metadata_text = metadata_value.item()
        if isinstance(metadata_text, bytes):
            metadata_text = metadata_text.decode("utf-8")
        metadata = json.loads(metadata_text)

    expected = {
        "archive_kind": REAL_SNAPSHOT_KIND,
        "schema_version": REAL_ARCHIVE_SCHEMA_VERSION,
        "run_id": run_id,
        "method": method,
        "method_file_stem": method_file_stem,
        "epoch": epoch,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Reconstruction snapshot metadata mismatch for {path}:{key}."
            )
    expected_arrays = {
        "object": object_snapshot,
        "probe": probe_snapshot,
        "positions_px": positions_snapshot,
    }
    array_metadata = metadata.get("arrays")
    if not isinstance(array_metadata, Mapping) or set(array_metadata) != set(
        expected_arrays
    ):
        raise ValueError(
            f"Reconstruction snapshot array metadata is incomplete: {path}."
        )
    for name, value in expected_arrays.items():
        record = array_metadata[name]
        if not isinstance(record, Mapping):
            raise TypeError(f"Invalid reconstruction snapshot metadata: {path}:{name}.")
        if (
            record.get("shape") != list(value.shape)
            or record.get("dtype") != str(value.dtype)
            or record.get("sha256") != _sha256_array(value)
        ):
            raise ValueError(f"Reconstruction snapshot array mismatch: {path}:{name}.")
    return metadata


def save_real_reconstruction_snapshot(
    output_root: Path,
    *,
    run_id: str,
    method: str,
    method_file_stem: str,
    epoch: int,
    object_snapshot: np.ndarray,
    probe_snapshot: np.ndarray,
    positions_snapshot: np.ndarray,
) -> Path:
    """Atomically save one completed real-data reconstruction epoch.

    Snapshots live beside the per-method checkpoints so they survive a later
    method failure or scheduler timeout. A completed comparison archive copies
    and checksums the full expected snapshot sequence.
    """
    output_root = Path(output_root).resolve()
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must be one safe path component.")
    if method not in CANONICAL_REAL_METHODS:
        raise ValueError(f"Unknown real-data method {method!r}.")
    if not _SAFE_RUN_ID.fullmatch(method_file_stem):
        raise ValueError("method_file_stem must be one safe path component.")
    if isinstance(epoch, (bool, np.bool_)) or not isinstance(epoch, Integral):
        raise TypeError("epoch must be an integer.")
    epoch = int(epoch)
    if epoch < 1:
        raise ValueError("epoch must be positive.")

    object_array = _finite_array(object_snapshot, label="snapshot object")
    probe_array = _finite_array(probe_snapshot, label="snapshot probe")
    positions_array = _finite_array(
        positions_snapshot,
        label="snapshot positions_px",
    )
    if not np.iscomplexobj(object_array) or not np.iscomplexobj(probe_array):
        raise TypeError("Reconstruction snapshots must be complex-valued.")
    if (
        np.iscomplexobj(positions_array)
        or not np.issubdtype(positions_array.dtype, np.floating)
        or positions_array.ndim != 2
        or positions_array.shape[1] != 2
    ):
        raise TypeError("Snapshot positions_px must be a real (patterns, 2) array.")

    path = _snapshot_directory(
        output_root,
        run_id=run_id,
        method_file_stem=method_file_stem,
    ) / _snapshot_filename(epoch)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite reconstruction snapshot: {path}")
    metadata = {
        "archive_kind": REAL_SNAPSHOT_KIND,
        "schema_version": REAL_ARCHIVE_SCHEMA_VERSION,
        "run_id": run_id,
        "method": method,
        "method_file_stem": method_file_stem,
        "epoch": epoch,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _sha256_array(value),
            }
            for name, value in {
                "object": object_array,
                "probe": probe_array,
                "positions_px": positions_array,
            }.items()
        },
    }
    _atomic_savez(
        path,
        {
            "object": object_array,
            "probe": probe_array,
            "positions_px": positions_array,
            "epoch": np.asarray(epoch, dtype=np.int64),
            "metadata_json": np.asarray(_canonical_json(metadata)),
        },
    )
    _validate_real_reconstruction_snapshot(
        path,
        run_id=run_id,
        method=method,
        method_file_stem=method_file_stem,
        epoch=epoch,
    )
    return path


def _verify_npz_array_hashes(path: Path, expected: Mapping[str, str]) -> None:
    with np.load(path, allow_pickle=False) as payload:
        if set(expected) != set(payload.files) - {"metadata_json"}:
            raise ValueError(f"Array-hash inventory mismatch for {path}.")
        for key, expected_hash in expected.items():
            if _sha256_array(payload[key]) != expected_hash:
                raise RuntimeError(f"Array hash verification failed for {path}:{key}.")


def _validate_saved_position_arrays(
    path: Path,
    *,
    expected_count: int | None = None,
    initial_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        positions = _finite_array(payload["positions_px"], label=f"{path}:positions_px")
        shifts = _finite_array(
            payload["position_shifts_px"],
            label=f"{path}:position_shifts_px",
        )
    for name, value in (("positions_px", positions), ("position_shifts_px", shifts)):
        if np.iscomplexobj(value) or not np.issubdtype(value.dtype, np.floating):
            raise TypeError(f"{path}:{name} must contain real floating-point values.")
        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError(f"{path}:{name} must have shape (patterns, 2).")
    if shifts.shape != positions.shape:
        raise ValueError(f"{path}:position arrays have inconsistent shapes.")
    if expected_count is not None and positions.shape[0] != expected_count:
        raise ValueError(f"{path}:position arrays have the wrong pattern count.")
    if initial_positions is not None:
        initial = np.asarray(initial_positions)
        if initial.shape != positions.shape or not np.array_equal(
            shifts,
            positions - initial,
        ):
            raise ValueError(
                f"{path}:saved position shifts do not match the initial scan."
            )
    return positions, shifts


@dataclass
class RealResultArchiveSession:
    """Configuration for restartable, immutable real-data result exports.

    Call :meth:`save_method` immediately after each reconstruction has been
    aligned. The four method checkpoints are independent directories, so a
    later method failure cannot erase an earlier result. Call
    :func:`save_real_notebook_results` after all four checkpoints exist to
    build one validated comparison archive.
    """

    output_root: Path
    project_root: Path
    experiment_name: str
    notebook_path: Path
    run_started_utc: datetime
    run_id: str
    dataset: Any
    dataset_config: Any
    source_data_path: Path
    method_order: Sequence[str]
    method_file_stems: Mapping[str, str]
    method_parameters: Mapping[str, Mapping[str, Any]]
    metric_report_every: int
    include_initial_metrics: bool
    sampled_metric_keys: Sequence[str]
    final_metric_keys: Sequence[str]
    device: Any
    source_paths: Mapping[str, Path]
    metric_definitions: Mapping[str, str]
    timing_notes: str
    alignment_scale_amplitude: bool
    alignment_fit_mask: np.ndarray
    run_metadata: Mapping[str, Any] = field(default_factory=dict)
    snapshot_every: int | None = None

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root).resolve()
        self.project_root = Path(self.project_root).resolve()
        self.notebook_path = Path(self.notebook_path).resolve()
        self.source_data_path = Path(self.source_data_path).resolve()
        self.method_order = tuple(self.method_order)
        self.method_file_stems = dict(self.method_file_stems)
        self.method_parameters = {
            method: dict(parameters)
            for method, parameters in self.method_parameters.items()
        }
        self.sampled_metric_keys = tuple(self.sampled_metric_keys)
        self.final_metric_keys = tuple(self.final_metric_keys)
        self.source_paths = {
            str(name): Path(path).resolve() for name, path in self.source_paths.items()
        }
        self.metric_definitions = dict(self.metric_definitions)
        self.run_metadata = dict(self.run_metadata)
        self.alignment_fit_mask = np.asarray(
            self.alignment_fit_mask,
            dtype=bool,
        )
        self._validate_session()

    @property
    def checkpoint_root(self) -> Path:
        return self.output_root / "_checkpoints" / self.run_id

    @property
    def final_directory(self) -> Path:
        return self.output_root / self.run_id

    @property
    def epoch_snapshot_root(self) -> Path:
        return self.checkpoint_root / "_epoch_snapshots"

    @property
    def expected_snapshot_epochs(self) -> tuple[int, ...]:
        if self.snapshot_every is None:
            return ()
        num_epochs = int(self.dataset_config.num_epochs)
        return tuple(
            sorted(
                {
                    num_epochs,
                    *range(self.snapshot_every, num_epochs + 1, self.snapshot_every),
                }
            )
        )

    @property
    def session_fingerprint(self) -> str:
        payload = {
            "archive_kind": REAL_ARCHIVE_KIND,
            "schema_version": REAL_ARCHIVE_SCHEMA_VERSION,
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "run_started_utc": self.run_started_utc.astimezone(
                timezone.utc
            ).isoformat(),
            "dataset_config": self.dataset_config,
            "source_data_path": str(self.source_data_path),
            "method_order": self.method_order,
            "method_file_stems": self.method_file_stems,
            "method_parameters": self.method_parameters,
            "metric_report_every": self.metric_report_every,
            "include_initial_metrics": self.include_initial_metrics,
            "sampled_metric_keys": self.sampled_metric_keys,
            "final_metric_keys": self.final_metric_keys,
            "snapshot_every": self.snapshot_every,
            "metric_definitions": self.metric_definitions,
            "device": self.device,
            "timing_notes": self.timing_notes,
            "alignment_scale_amplitude": self.alignment_scale_amplitude,
            "run_metadata": self.run_metadata,
            "source_paths": {
                name: str(path) for name, path in sorted(self.source_paths.items())
            },
            "source_content_sha256": {
                name: (
                    _notebook_source_sha256(path)
                    if name == "notebook"
                    else _sha256_file(path)
                )
                for name, path in sorted(self.source_paths.items())
                if path != self.source_data_path
            },
            "selected_pattern_indices_sha256": _sha256_array(
                self.dataset.selected_pattern_indices
            ),
            "initial_positions_px_sha256": _sha256_array(self.dataset.positions_px),
            "probe_initial_sha256": _sha256_array(self.dataset.probe_init),
            "alignment_fit_mask_sha256": _sha256_array(self.alignment_fit_mask),
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def _validate_session(self) -> None:
        if self.method_order != CANONICAL_REAL_METHODS:
            raise ValueError(
                "method_order must be exactly "
                f"{CANONICAL_REAL_METHODS!r} for a real-data comparison."
            )
        expected_methods = set(self.method_order)
        for name, mapping in (
            ("method_file_stems", self.method_file_stems),
            ("method_parameters", self.method_parameters),
        ):
            if set(mapping) != expected_methods:
                raise ValueError(f"{name} must contain all four methods exactly once.")
        stems = tuple(self.method_file_stems[method] for method in self.method_order)
        if len(stems) != len(set(stems)) or any(
            not _SAFE_RUN_ID.fullmatch(stem) for stem in stems
        ):
            raise ValueError("Method file stems must be unique safe path components.")
        if (
            not isinstance(self.experiment_name, str)
            or not self.experiment_name.strip()
        ):
            raise ValueError("experiment_name must be a nonempty string.")
        if not _SAFE_RUN_ID.fullmatch(self.run_id):
            raise ValueError("run_id must be one safe path component.")
        if (
            not isinstance(self.run_started_utc, datetime)
            or self.run_started_utc.tzinfo is None
        ):
            raise ValueError("run_started_utc must be a timezone-aware datetime.")
        if isinstance(self.metric_report_every, bool) or not isinstance(
            self.metric_report_every, int
        ):
            raise TypeError("metric_report_every must be an integer.")
        if self.metric_report_every < 1:
            raise ValueError("metric_report_every must be positive.")
        if not isinstance(self.include_initial_metrics, bool):
            raise TypeError("include_initial_metrics must be boolean.")
        if len(set(self.sampled_metric_keys)) != len(self.sampled_metric_keys):
            raise ValueError("sampled_metric_keys must contain unique metric names.")
        if len(set(self.final_metric_keys)) != len(self.final_metric_keys):
            raise ValueError("final_metric_keys must contain unique metric names.")
        all_metric_keys = set(self.sampled_metric_keys) | set(self.final_metric_keys)
        expected_definitions = all_metric_keys | {"online_preupdate_residual"}
        if set(self.metric_definitions) != expected_definitions or any(
            not isinstance(value, str) or not value.strip()
            for value in self.metric_definitions.values()
        ):
            raise ValueError("metric_definitions must define every archived metric.")
        if not isinstance(self.timing_notes, str) or not self.timing_notes.strip():
            raise ValueError("timing_notes must be a nonempty string.")
        if not isinstance(self.alignment_scale_amplitude, bool):
            raise TypeError("alignment_scale_amplitude must be boolean.")
        if self.snapshot_every is not None:
            if isinstance(self.snapshot_every, bool) or not isinstance(
                self.snapshot_every,
                int,
            ):
                raise TypeError("snapshot_every must be an integer or None.")
            if self.snapshot_every < 1:
                raise ValueError("snapshot_every must be positive when provided.")
        if self.alignment_fit_mask.ndim != 2 or not np.any(self.alignment_fit_mask):
            raise ValueError(
                "alignment_fit_mask must be a nonempty two-dimensional mask."
            )

        required_dataset_fields = (
            "data",
            "positions_px",
            "probe_init",
            "dx_m",
            "wavelength_m",
            "selected_pattern_indices",
            "source_pattern_count",
        )
        missing_dataset_fields = [
            name for name in required_dataset_fields if not hasattr(self.dataset, name)
        ]
        if missing_dataset_fields:
            raise TypeError(
                "dataset is missing archive fields: "
                + ", ".join(missing_dataset_fields)
            )
        data = np.asarray(self.dataset.data)
        positions = _finite_array(
            self.dataset.positions_px, label="dataset.positions_px"
        )
        probe = _finite_array(self.dataset.probe_init, label="dataset.probe_init")
        indices = np.asarray(self.dataset.selected_pattern_indices)
        if data.ndim != 3 or data.shape[0] < 1:
            raise ValueError("dataset.data must have shape (patterns, height, width).")
        if positions.shape != (data.shape[0], 2):
            raise ValueError(
                "dataset.positions_px must contain one (y, x) pair per frame."
            )
        if probe.ndim < 2 or not np.iscomplexobj(probe):
            raise TypeError("dataset.probe_init must be a complex array.")
        if indices.shape != (data.shape[0],) or not np.issubdtype(
            indices.dtype, np.integer
        ):
            raise ValueError(
                "dataset.selected_pattern_indices must contain one integer per frame."
            )
        source_count = int(self.dataset.source_pattern_count)
        if (
            source_count < data.shape[0]
            or np.any(indices < 0)
            or np.any(indices >= source_count)
        ):
            raise ValueError("Selected pattern indices are outside the source dataset.")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Selected pattern indices must be unique.")
        _positive_finite(self.dataset.dx_m, label="dataset.dx_m")
        _positive_finite(self.dataset.wavelength_m, label="dataset.wavelength_m")
        if int(getattr(self.dataset_config, "num_epochs")) < 1:
            raise ValueError("dataset_config.num_epochs must be positive.")
        if int(getattr(self.dataset_config, "batch_size")) < 1:
            raise ValueError("dataset_config.batch_size must be positive.")

        for label, authoritative_path in (
            ("notebook", self.notebook_path),
            ("source_hdf5", self.source_data_path),
        ):
            supplied_path = self.source_paths.get(label)
            if supplied_path is not None and supplied_path != authoritative_path:
                raise ValueError(
                    f"source_paths[{label!r}] conflicts with the dedicated path."
                )
            self.source_paths[label] = authoritative_path
        missing_sources = [
            name for name, path in self.source_paths.items() if not path.is_file()
        ]
        if not self.notebook_path.is_file():
            missing_sources.append("notebook")
        if not self.source_data_path.is_file():
            missing_sources.append("source_hdf5")
        if missing_sources:
            raise FileNotFoundError(
                "Missing source files: " + ", ".join(sorted(set(missing_sources)))
            )
        _canonical_json(self.device)
        _canonical_json(self.run_metadata)

    def _validate_method_result(
        self,
        *,
        method: str,
        result: Any,
        runtime_seconds: float,
        aligned_object: np.ndarray,
        aligned_probe: np.ndarray,
        alignment: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if method not in self.method_order:
            raise ValueError(f"Unknown method {method!r}.")
        runtime = _positive_finite(runtime_seconds, label=f"{method} runtime_seconds")
        raw_object = _finite_array(result.object, label=f"{method} object")
        raw_probe = _finite_array(result.probe, label=f"{method} probe")
        object_aligned = _finite_array(
            aligned_object,
            label=f"{method} aligned_object",
        )
        probe_aligned = _finite_array(
            aligned_probe,
            label=f"{method} aligned_probe",
        )
        for label, value in (
            ("object", raw_object),
            ("probe", raw_probe),
            ("aligned_object", object_aligned),
            ("aligned_probe", probe_aligned),
        ):
            if not np.iscomplexobj(value):
                raise TypeError(f"{method} {label} must be complex-valued.")
        if np.squeeze(raw_object).shape != np.squeeze(object_aligned).shape:
            raise ValueError(f"{method} raw and aligned object shapes differ.")
        if np.squeeze(raw_probe).shape != np.squeeze(probe_aligned).shape:
            raise ValueError(f"{method} raw and aligned probe shapes differ.")
        if np.squeeze(object_aligned).shape != self.alignment_fit_mask.shape:
            raise ValueError(
                f"{method} aligned object does not match alignment_fit_mask."
            )

        initial_positions = _finite_array(
            self.dataset.positions_px,
            label="dataset.positions_px",
        )
        refined_positions = _finite_array(
            result.positions_px,
            label=f"{method} positions_px",
        )
        if np.iscomplexobj(refined_positions) or not np.issubdtype(
            refined_positions.dtype,
            np.floating,
        ):
            raise TypeError(
                f"{method} positions_px must be real floating-point values."
            )
        if refined_positions.shape != initial_positions.shape:
            raise ValueError(
                f"{method} positions_px must contain one (y, x) pair per frame."
            )
        position_shifts = refined_positions - initial_positions
        position_shift_norms = np.linalg.norm(position_shifts, axis=1)
        position_summary = {
            "coordinate_order": ["y", "x"],
            "units": "pixel",
            "pattern_count": int(len(refined_positions)),
            "rms_shift_px": float(
                np.sqrt(np.mean(np.square(position_shift_norms), dtype=np.float64))
            ),
            "max_shift_px": float(np.max(position_shift_norms)),
            "mean_shift_y_px": float(np.mean(position_shifts[:, 0], dtype=np.float64)),
            "mean_shift_x_px": float(np.mean(position_shifts[:, 1], dtype=np.float64)),
        }

        residual = _finite_array(
            result.residual_history,
            label=f"{method} residual_history",
        )
        expected_epochs = int(self.dataset_config.num_epochs)
        if residual.ndim != 1 or residual.shape != (expected_epochs,):
            raise ValueError(f"{method} residual history has the wrong length.")
        if np.any(residual < 0):
            raise ValueError(f"{method} residual history must be nonnegative.")
        metric_epochs = np.asarray(result.metric_epochs)
        expected_metric_epochs = (
            _expected_metric_epochs(
                expected_epochs,
                self.metric_report_every,
                self.include_initial_metrics,
            )
            if self.sampled_metric_keys
            else np.empty(0, dtype=np.int64)
        )
        if not np.issubdtype(metric_epochs.dtype, np.integer) or not np.array_equal(
            metric_epochs,
            expected_metric_epochs,
        ):
            raise ValueError(f"{method} metric epochs do not match the saved cadence.")
        if set(result.metric_history) != set(self.sampled_metric_keys):
            raise ValueError(f"{method} has unexpected sampled metric keys.")
        for key in self.sampled_metric_keys:
            values = _finite_array(
                result.metric_history[key],
                label=f"{method} metric_history[{key!r}]",
            )
            if values.shape != metric_epochs.shape:
                raise ValueError(
                    f"{method} sampled metric {key!r} has the wrong shape."
                )
            if np.any(values < 0):
                raise ValueError(
                    f"{method} sampled metric {key!r} must be nonnegative."
                )
        if set(result.final_metrics) != set(self.final_metric_keys):
            raise ValueError(f"{method} has unexpected final metric keys.")
        for key in self.final_metric_keys:
            value = float(result.final_metrics[key])
            if not math.isfinite(value):
                raise ValueError(f"{method} final metric {key!r} must be finite.")
            if value < 0:
                raise ValueError(f"{method} final metric {key!r} must be nonnegative.")
            if key in result.metric_history:
                np.testing.assert_allclose(
                    value,
                    np.asarray(result.metric_history[key])[-1],
                    rtol=1e-5,
                    atol=1e-7,
                    err_msg=f"{method} final metric {key!r} is stale.",
                )

        alignment_values = _alignment_values(alignment)
        payload: dict[str, Any] = {
            "object": raw_object,
            "probe": raw_probe,
            "positions_px": refined_positions,
            "position_shifts_px": position_shifts,
            "aligned_object": object_aligned,
            "aligned_probe": probe_aligned,
            "residual_history": residual,
            "metric_epochs": metric_epochs.astype(np.int64, copy=False),
            "runtime_seconds": np.asarray(runtime, dtype=np.float64),
            "alignment_slope_y_rad_per_px": np.asarray(
                alignment_values["slope_y_rad_per_px"], dtype=np.float64
            ),
            "alignment_slope_x_rad_per_px": np.asarray(
                alignment_values["slope_x_rad_per_px"], dtype=np.float64
            ),
            "alignment_object_scale": np.asarray(
                alignment_values["object_scale"], dtype=np.complex128
            ),
            "metadata_json": np.asarray(
                _canonical_json(
                    {
                        "archive_kind": REAL_CHECKPOINT_KIND,
                        "schema_version": REAL_ARCHIVE_SCHEMA_VERSION,
                        "experiment_name": self.experiment_name,
                        "run_id": self.run_id,
                        "method": method,
                        "parameters": self.method_parameters[method],
                        "alignment_scale_amplitude": self.alignment_scale_amplitude,
                    }
                )
            ),
        }
        payload.update(
            {
                f"metric_history__{key}": result.metric_history[key]
                for key in self.sampled_metric_keys
            }
        )
        payload.update(
            {
                f"final_metric__{key}": np.asarray(
                    result.final_metrics[key], dtype=np.float64
                )
                for key in self.final_metric_keys
            }
        )
        manifest = {
            "archive_kind": REAL_CHECKPOINT_KIND,
            "schema_version": REAL_ARCHIVE_SCHEMA_VERSION,
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "session_fingerprint": self.session_fingerprint,
            "method": method,
            "method_file_stem": self.method_file_stems[method],
            "archive": "reconstruction.npz",
            "parameters": self.method_parameters[method],
            "runtime_seconds": runtime,
            "final_online_preupdate_residual": float(residual[-1]),
            "metric_report_every": self.metric_report_every,
            "include_initial_metrics": self.include_initial_metrics,
            "sampled_metric_keys": list(self.sampled_metric_keys),
            "final_metric_keys": list(self.final_metric_keys),
            "final_metrics": result.final_metrics,
            "alignment": alignment_values,
            "alignment_scale_amplitude": self.alignment_scale_amplitude,
            "position_refinement": position_summary,
            "array_sha256": {
                name: _sha256_array(value)
                for name, value in payload.items()
                if name != "metadata_json"
            },
        }
        return payload, manifest

    def save_method(
        self,
        *,
        method: str,
        result: Any,
        runtime_seconds: float,
        aligned_object: np.ndarray,
        aligned_probe: np.ndarray,
        alignment: Any,
    ) -> Path:
        """Atomically save one method; an existing checkpoint is never replaced."""
        if method not in self.method_order:
            raise ValueError(f"Unknown method {method!r}.")
        snapshot_records: list[dict[str, Any]] = []
        if self.snapshot_every is not None:
            snapshot_records = _method_snapshot_records(
                self,
                method,
                directory=(self.epoch_snapshot_root / self.method_file_stems[method]),
            )
        payload, manifest = self._validate_method_result(
            method=method,
            result=result,
            runtime_seconds=runtime_seconds,
            aligned_object=aligned_object,
            aligned_probe=aligned_probe,
            alignment=alignment,
        )
        if snapshot_records:
            final_snapshot = (
                self.epoch_snapshot_root
                / self.method_file_stems[method]
                / snapshot_records[-1]["filename"]
            )
            with np.load(final_snapshot, allow_pickle=False) as snapshot:
                for name in ("object", "probe", "positions_px"):
                    if not np.array_equal(snapshot[name], payload[name]):
                        raise ValueError(
                            f"{method} final snapshot does not match final {name}."
                        )
        stem = self.method_file_stems[method]
        checkpoint_root = self.checkpoint_root
        final_directory = checkpoint_root / stem
        if final_directory.exists():
            raise FileExistsError(
                f"Refusing to overwrite method checkpoint: {final_directory}"
            )
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        temporary_directory = Path(
            tempfile.mkdtemp(prefix=f".{stem}.", dir=checkpoint_root)
        )
        try:
            _atomic_savez(temporary_directory / "reconstruction.npz", payload)
            _write_json(temporary_directory / "method_manifest.json", manifest)
            _write_checksum_file(temporary_directory)
            _verify_checksum_file(temporary_directory)
            _atomic_write_text(temporary_directory / "_SUCCESS", f"{method}\n")
            validate_real_method_checkpoint(temporary_directory)
            temporary_directory.replace(final_directory)
        except BaseException:
            if temporary_directory.exists():
                shutil.rmtree(temporary_directory, ignore_errors=True)
            raise
        return final_directory


def validate_real_method_checkpoint(directory: Path) -> dict[str, Any]:
    """Validate one completed v1 or v2 method checkpoint and return its manifest.

    Schema v1 is supported only as a read-only, fixed-position legacy format. New
    checkpoints are always written as schema v2 and include refined positions.
    """
    directory = Path(directory).resolve()
    manifest_path = directory / "method_manifest.json"
    success_path = directory / "_SUCCESS"
    if (
        not directory.is_dir()
        or not manifest_path.is_file()
        or not success_path.is_file()
    ):
        raise FileNotFoundError(f"Incomplete real-data checkpoint: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("archive_kind") != REAL_CHECKPOINT_KIND:
        raise ValueError(f"Not a real-data method checkpoint: {directory}")
    schema_version = manifest.get("schema_version")
    if schema_version not in {
        _LEGACY_FIXED_POSITION_CHECKPOINT_SCHEMA_VERSION,
        REAL_ARCHIVE_SCHEMA_VERSION,
    }:
        raise ValueError(f"Unsupported real-data checkpoint schema: {directory}")
    method = manifest.get("method")
    if success_path.read_text(encoding="utf-8") != f"{method}\n":
        raise RuntimeError(f"Checkpoint success marker does not match {method!r}.")
    _verify_checksum_file(directory)
    common_required_keys = {
        "object",
        "probe",
        "aligned_object",
        "aligned_probe",
        "residual_history",
        "metric_epochs",
        "runtime_seconds",
        "alignment_slope_y_rad_per_px",
        "alignment_slope_x_rad_per_px",
        "alignment_object_scale",
        "metadata_json",
        *{f"metric_history__{key}" for key in manifest.get("sampled_metric_keys", ())},
        *{f"final_metric__{key}" for key in manifest.get("final_metric_keys", ())},
    }
    required_keys = set(common_required_keys)
    if schema_version == REAL_ARCHIVE_SCHEMA_VERSION:
        required_keys.update({"positions_px", "position_shifts_px"})
    keys = _validate_npz(directory / "reconstruction.npz", required_keys)
    if any(key.startswith("data") or "intensity" in key for key in keys):
        raise ValueError("A checkpoint must not duplicate measured diffraction data.")
    _verify_npz_array_hashes(
        directory / "reconstruction.npz",
        manifest.get("array_sha256", {}),
    )
    if schema_version == _LEGACY_FIXED_POSITION_CHECKPOINT_SCHEMA_VERSION:
        position_keys = {"positions_px", "position_shifts_px"}
        if keys & position_keys or "position_refinement" in manifest:
            raise ValueError(
                "A schema-v1 checkpoint must remain a fixed-position legacy "
                f"record without refined-position fields: {directory}"
            )
        return manifest

    position_metadata = manifest.get("position_refinement")
    if not isinstance(position_metadata, Mapping):
        raise ValueError(f"Checkpoint position metadata is missing: {directory}")
    _validate_saved_position_arrays(
        directory / "reconstruction.npz",
        expected_count=int(position_metadata.get("pattern_count", -1)),
    )
    return manifest


def _method_snapshot_records(
    session: RealResultArchiveSession,
    method: str,
    *,
    directory: Path,
) -> list[dict[str, Any]]:
    if method not in session.method_order:
        raise ValueError(f"Unknown method {method!r}.")
    if session.snapshot_every is None:
        return []

    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing reconstruction snapshots: {directory}")
    stem = session.method_file_stems[method]
    expected_epochs = session.expected_snapshot_epochs
    expected_names = {_snapshot_filename(epoch) for epoch in expected_epochs}
    actual_paths = {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix == ".npz"
    }
    if set(actual_paths) != expected_names:
        missing = sorted(expected_names - set(actual_paths))
        extra = sorted(set(actual_paths) - expected_names)
        raise RuntimeError(
            f"Reconstruction snapshot sequence mismatch for {method}; "
            f"missing={missing!r}, extra={extra!r}."
        )

    records = []
    for epoch in expected_epochs:
        path = actual_paths[_snapshot_filename(epoch)]
        metadata = _validate_real_reconstruction_snapshot(
            path,
            run_id=session.run_id,
            method=method,
            method_file_stem=stem,
            epoch=epoch,
        )
        if metadata["arrays"]["positions_px"]["shape"] != [
            len(session.dataset.positions_px),
            2,
        ]:
            raise ValueError(
                f"Reconstruction snapshot has the wrong position count: {path}."
            )
        records.append(
            {
                "epoch": epoch,
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "arrays": metadata["arrays"],
            }
        )
    return records


def _source_file_records(session: RealResultArchiveSession) -> dict[str, Any]:
    cache: dict[Path, str] = {}

    def digest(path: Path) -> str:
        if path not in cache:
            cache[path] = _sha256_file(path)
        return cache[path]

    records = {
        name: {
            "path": _relative_or_absolute(path, session.project_root),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for name, path in sorted(session.source_paths.items())
    }
    records["notebook"]["semantic_source_sha256"] = _notebook_source_sha256(
        session.notebook_path
    )
    return records


def save_real_notebook_results(session: RealResultArchiveSession) -> Path:
    """Assemble all four immutable checkpoints into one verified run archive."""
    if not isinstance(session, RealResultArchiveSession):
        raise TypeError("session must be a RealResultArchiveSession.")
    final_directory = session.final_directory
    if final_directory.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result archive: {final_directory}"
        )
    checkpoints: dict[str, Path] = {}
    checkpoint_manifests: dict[str, dict[str, Any]] = {}
    snapshot_records: dict[str, list[dict[str, Any]]] = {}
    for method in session.method_order:
        checkpoint = session.checkpoint_root / session.method_file_stems[method]
        manifest = validate_real_method_checkpoint(checkpoint)
        if manifest.get("schema_version") != REAL_ARCHIVE_SCHEMA_VERSION:
            raise ValueError(
                "A new schema-v2 comparison archive cannot be assembled from "
                f"a legacy schema-v1 checkpoint: {checkpoint}"
            )
        if manifest.get("method") != method:
            raise ValueError(f"Checkpoint method mismatch for {checkpoint}.")
        if manifest.get("session_fingerprint") != session.session_fingerprint:
            raise ValueError(f"Checkpoint belongs to a different run: {checkpoint}.")
        checkpoints[method] = checkpoint
        checkpoint_manifests[method] = manifest
        snapshot_records[method] = _method_snapshot_records(
            session,
            method,
            directory=(session.epoch_snapshot_root / session.method_file_stems[method]),
        )

    session.output_root.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=f".{session.run_id}.", dir=session.output_root)
    )
    saved_utc = datetime.now(timezone.utc)
    try:
        summary_rows: list[dict[str, Any]] = []
        metric_rows: list[dict[str, Any]] = []
        residual_rows: list[dict[str, Any]] = []
        runtimes: dict[str, float] = {}
        for method in session.method_order:
            stem = session.method_file_stems[method]
            checkpoint = checkpoints[method]
            destination_npz = temporary_directory / "reconstructions" / f"{stem}.npz"
            destination_npz.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint / "reconstruction.npz", destination_npz)

            manifest = dict(checkpoint_manifests[method])
            manifest["archive_kind"] = REAL_ARCHIVE_KIND
            manifest["archive"] = f"reconstructions/{stem}.npz"
            manifest["checkpoint"] = _relative_or_absolute(
                checkpoint,
                session.output_root,
            )
            manifest["snapshots"] = [
                {
                    **record,
                    "archive": f"snapshots/{stem}/{record['filename']}",
                }
                for record in snapshot_records[method]
            ]
            _write_json(
                temporary_directory / "manifests" / f"{stem}.json",
                manifest,
            )

            for record in snapshot_records[method]:
                source = session.epoch_snapshot_root / stem / record["filename"]
                destination = (
                    temporary_directory / "snapshots" / stem / record["filename"]
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

            with np.load(destination_npz, allow_pickle=False) as payload:
                runtime = float(payload["runtime_seconds"])
                runtimes[method] = runtime
                residual = np.asarray(payload["residual_history"])
                metric_epochs = np.asarray(payload["metric_epochs"])
                summary_row: dict[str, Any] = {
                    "method": method,
                    "runtime_seconds": runtime,
                    "final_online_preupdate_residual": float(residual[-1]),
                    "position_rms_shift_px": float(
                        manifest["position_refinement"]["rms_shift_px"]
                    ),
                    "position_max_shift_px": float(
                        manifest["position_refinement"]["max_shift_px"]
                    ),
                    "num_epochs": int(session.dataset_config.num_epochs),
                    "batch_size": int(session.dataset_config.batch_size),
                    "seed": int(session.dataset_config.seed),
                    "pattern_count": int(len(session.dataset.selected_pattern_indices)),
                }
                summary_row.update(
                    {
                        key: float(payload[f"final_metric__{key}"])
                        for key in session.final_metric_keys
                    }
                )
                summary_rows.append(summary_row)
                for index, epoch in enumerate(metric_epochs):
                    row: dict[str, Any] = {
                        "method": method,
                        "epoch": int(epoch),
                    }
                    row.update(
                        {
                            key: float(payload[f"metric_history__{key}"][index])
                            for key in session.sampled_metric_keys
                        }
                    )
                    metric_rows.append(row)
                residual_rows.extend(
                    {
                        "method": method,
                        "epoch": epoch,
                        "online_preupdate_residual": float(value),
                    }
                    for epoch, value in enumerate(residual, start=1)
                )

        _write_csv(
            temporary_directory / "tables" / "summary.csv",
            [
                "method",
                "runtime_seconds",
                "final_online_preupdate_residual",
                "position_rms_shift_px",
                "position_max_shift_px",
                *session.final_metric_keys,
                "num_epochs",
                "batch_size",
                "seed",
                "pattern_count",
            ],
            summary_rows,
        )
        if session.sampled_metric_keys:
            _write_csv(
                temporary_directory / "curves" / "metric_history.csv",
                ["method", "epoch", *session.sampled_metric_keys],
                metric_rows,
            )
        _write_csv(
            temporary_directory / "curves" / "residual_history.csv",
            ["method", "epoch", "online_preupdate_residual"],
            residual_rows,
        )

        dataset_payload = {
            "selected_pattern_indices": np.asarray(
                session.dataset.selected_pattern_indices,
                dtype=np.int64,
            ),
            "positions_px": session.dataset.positions_px,
            "probe_initial": session.dataset.probe_init,
            "dx_m": np.asarray(session.dataset.dx_m, dtype=np.float64),
            "wavelength_m": np.asarray(
                session.dataset.wavelength_m,
                dtype=np.float64,
            ),
            "source_pattern_count": np.asarray(
                session.dataset.source_pattern_count,
                dtype=np.int64,
            ),
            "alignment_fit_mask": session.alignment_fit_mask,
        }
        _atomic_savez(
            temporary_directory / "data" / "real_dataset_metadata.npz",
            dataset_payload,
        )

        source_records = _source_file_records(session)
        source_hdf5_hash = source_records.get("source_hdf5", {}).get("sha256")
        if source_hdf5_hash is None:
            source_hdf5_hash = _sha256_file(session.source_data_path)
        source_hdf5 = {
            "path": _relative_or_absolute(
                session.source_data_path,
                session.project_root,
            ),
            "bytes": session.source_data_path.stat().st_size,
            "sha256": source_hdf5_hash,
        }
        git_status = _git_output(session.project_root, "status", "--porcelain")
        run_manifest = {
            "archive_kind": REAL_ARCHIVE_KIND,
            "schema_version": REAL_ARCHIVE_SCHEMA_VERSION,
            "experiment_name": session.experiment_name,
            "run_id": session.run_id,
            "session_fingerprint": session.session_fingerprint,
            "run_started_utc": session.run_started_utc.astimezone(
                timezone.utc
            ).isoformat(),
            "saved_utc": saved_utc.isoformat(),
            "device": session.device,
            "method_order": list(session.method_order),
            "method_file_stems": session.method_file_stems,
            "method_parameters": session.method_parameters,
            "metric_definitions": session.metric_definitions,
            "metric_report_every": session.metric_report_every,
            "include_initial_metrics": session.include_initial_metrics,
            "sampled_metric_keys": list(session.sampled_metric_keys),
            "final_metric_keys": list(session.final_metric_keys),
            "snapshot_every": session.snapshot_every,
            "snapshot_epochs": list(session.expected_snapshot_epochs),
            "snapshots": snapshot_records,
            "dataset_config": session.dataset_config,
            "dataset_positions_px_role": (
                "Initial mean-centered measured scan positions in (y, x) order."
            ),
            "run_metadata": session.run_metadata,
            "alignment": {
                "scale_amplitude": session.alignment_scale_amplitude,
                "fit_mask_sha256": _sha256_array(session.alignment_fit_mask),
            },
            "source_hdf5": source_hdf5,
            "selected_preprocessed_intensity": {
                "shape": list(np.asarray(session.dataset.data).shape),
                "dtype": str(np.asarray(session.dataset.data).dtype),
                "sha256": _sha256_array(session.dataset.data),
                "stored_in_archive": False,
            },
            "dataset_arrays": {
                name: {
                    "shape": list(np.asarray(value).shape),
                    "dtype": str(np.asarray(value).dtype),
                    "sha256": _sha256_array(value),
                }
                for name, value in dataset_payload.items()
            },
            "runtime_seconds": runtimes,
            "timing_notes": session.timing_notes,
            "software_versions": _real_software_versions(),
            "python": {
                "version": sys.version,
                "implementation": platform.python_implementation(),
            },
            "platform": platform.platform(),
            "git": {
                "commit": _git_output(session.project_root, "rev-parse", "HEAD"),
                "status_porcelain": [] if not git_status else git_status.splitlines(),
            },
            "source_files": source_records,
        }
        _write_json(temporary_directory / "run_manifest.json", run_manifest)
        _atomic_write_text(temporary_directory / "environment.txt", _pip_freeze())

        validation_report = {
            "archive_kind": REAL_ARCHIVE_KIND,
            "schema_version": REAL_ARCHIVE_SCHEMA_VERSION,
            "status": "passed",
            "methods": list(session.method_order),
            "method_count": len(session.method_order),
            "summary_rows": len(summary_rows),
            "sampled_metric_rows": len(metric_rows),
            "online_residual_rows": len(residual_rows),
            "reconstruction_snapshot_count": sum(
                len(records) for records in snapshot_records.values()
            ),
            "selected_pattern_count": len(session.dataset.selected_pattern_indices),
            "diffraction_data_duplicated": False,
            "checks": [
                "all four immutable method checkpoints are complete",
                "raw and aligned reconstructions are finite and complex-valued",
                "online residual histories contain every configured epoch",
                "optional sampled and final metrics have the configured cadence",
                "configured reconstruction snapshots are complete and validated",
                "initial and per-method refined positions are saved in (y, x) order",
                "selected indices, probe, pixel size, and wavelength are saved",
                "source HDF5 is content-addressed but diffraction data is not duplicated",
                "NPZ payloads round-trip exactly with allow_pickle=False",
                "all archived artifacts are covered by SHA-256 checksums",
                "no existing result directory was overwritten",
            ],
        }
        _write_json(
            temporary_directory / "validation_report.json",
            validation_report,
        )

        artifact_hashes = {}
        for path in sorted(temporary_directory.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            relative = PurePosixPath(path.relative_to(temporary_directory)).as_posix()
            artifact_hashes[relative] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        _write_json(
            temporary_directory / "release_manifest.json",
            {
                "archive_kind": REAL_ARCHIVE_KIND,
                "schema_version": REAL_ARCHIVE_SCHEMA_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "artifact_count": len(artifact_hashes),
                "artifacts": artifact_hashes,
            },
        )
        _write_checksum_file(temporary_directory)
        _verify_checksum_file(temporary_directory)
        _atomic_write_text(temporary_directory / "_SUCCESS", f"{session.run_id}\n")
        validate_real_results_archive(temporary_directory)
        temporary_directory.replace(final_directory)
    except BaseException:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    return final_directory


def validate_real_results_archive(directory: Path) -> dict[str, Any]:
    """Verify a completed real-data comparison archive and all file hashes."""
    directory = Path(directory).resolve()
    manifest_path = directory / "run_manifest.json"
    success_path = directory / "_SUCCESS"
    if (
        not directory.is_dir()
        or not manifest_path.is_file()
        or not success_path.is_file()
    ):
        raise FileNotFoundError(f"Incomplete real-data result archive: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("archive_kind") != REAL_ARCHIVE_KIND:
        raise ValueError(f"Not a real-data result archive: {directory}")
    if manifest.get("schema_version") != REAL_ARCHIVE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported real-data archive schema: {directory}")
    run_id = manifest.get("run_id")
    if success_path.read_text(encoding="utf-8") != f"{run_id}\n":
        raise RuntimeError("Archive success marker does not match run_manifest.json.")
    if tuple(manifest.get("method_order", ())) != CANONICAL_REAL_METHODS:
        raise ValueError("Archive does not contain the canonical four-method order.")
    _verify_checksum_file(directory)

    snapshot_every = manifest.get("snapshot_every")
    if snapshot_every is None:
        expected_snapshot_epochs: tuple[int, ...] = ()
    else:
        if isinstance(snapshot_every, bool) or not isinstance(snapshot_every, int):
            raise TypeError("Archive snapshot_every must be an integer or null.")
        if snapshot_every < 1:
            raise ValueError("Archive snapshot_every must be positive.")
        try:
            num_epochs = int(manifest["dataset_config"]["num_epochs"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Archive lacks a valid dataset epoch count.") from error
        expected_snapshot_epochs = tuple(
            sorted(
                {
                    num_epochs,
                    *range(snapshot_every, num_epochs + 1, snapshot_every),
                }
            )
        )
    if manifest.get("snapshot_epochs") != list(expected_snapshot_epochs):
        raise ValueError("Archive snapshot epochs do not match the configured cadence.")
    snapshot_records = manifest.get("snapshots")
    if not isinstance(snapshot_records, Mapping) or set(snapshot_records) != set(
        CANONICAL_REAL_METHODS
    ):
        raise ValueError("Archive snapshot manifest must cover all four methods.")

    stems = manifest.get("method_file_stems", {})
    for method in CANONICAL_REAL_METHODS:
        stem = stems.get(method)
        if not isinstance(stem, str):
            raise ValueError(f"Missing method file stem for {method}.")
        method_manifest_path = directory / "manifests" / f"{stem}.json"
        if not method_manifest_path.is_file():
            raise FileNotFoundError(f"Missing method manifest: {method_manifest_path}")
        method_manifest = json.loads(method_manifest_path.read_text(encoding="utf-8"))
        if method_manifest.get("method") != method:
            raise ValueError(f"Method manifest mismatch for {method}.")
        required_keys = {
            "object",
            "probe",
            "positions_px",
            "position_shifts_px",
            "aligned_object",
            "aligned_probe",
            "residual_history",
            "metric_epochs",
            "runtime_seconds",
            "alignment_slope_y_rad_per_px",
            "alignment_slope_x_rad_per_px",
            "alignment_object_scale",
            "metadata_json",
            *{
                f"metric_history__{key}"
                for key in manifest.get("sampled_metric_keys", ())
            },
            *{f"final_metric__{key}" for key in manifest.get("final_metric_keys", ())},
        }
        keys = _validate_npz(
            directory / "reconstructions" / f"{stem}.npz",
            required_keys,
        )
        if any(key.startswith("data") or "intensity" in key for key in keys):
            raise ValueError(
                "Reconstruction archives must not duplicate intensity data."
            )
        expected_metric_arrays = {
            *{
                f"metric_history__{key}"
                for key in manifest.get("sampled_metric_keys", ())
            },
            *{f"final_metric__{key}" for key in manifest.get("final_metric_keys", ())},
        }
        actual_metric_arrays = {
            key
            for key in keys
            if key.startswith("metric_history__") or key.startswith("final_metric__")
        }
        if actual_metric_arrays != expected_metric_arrays:
            raise ValueError(
                f"Unexpected metric arrays in reconstruction for {method}."
            )
        _verify_npz_array_hashes(
            directory / "reconstructions" / f"{stem}.npz",
            method_manifest.get("array_sha256", {}),
        )

        records = snapshot_records[method]
        if (
            not isinstance(records, list)
            or [
                record.get("epoch") for record in records if isinstance(record, Mapping)
            ]
            != list(expected_snapshot_epochs)
            or any(not isinstance(record, Mapping) for record in records)
        ):
            raise ValueError(f"Snapshot records have the wrong cadence for {method}.")
        method_records = method_manifest.get("snapshots")
        if not isinstance(method_records, list) or len(method_records) != len(records):
            raise ValueError(f"Method snapshot manifest is incomplete for {method}.")
        expected_names = {
            _snapshot_filename(epoch) for epoch in expected_snapshot_epochs
        }
        snapshot_directory = directory / "snapshots" / stem
        actual_names = (
            {
                path.name
                for path in snapshot_directory.iterdir()
                if path.is_file() and path.suffix == ".npz"
            }
            if snapshot_directory.is_dir()
            else set()
        )
        if actual_names != expected_names:
            raise RuntimeError(
                f"Archived reconstruction snapshots are incomplete for {method}."
            )
        for record, method_record, epoch in zip(
            records,
            method_records,
            expected_snapshot_epochs,
            strict=True,
        ):
            path = snapshot_directory / _snapshot_filename(epoch)
            metadata = _validate_real_reconstruction_snapshot(
                path,
                run_id=run_id,
                method=method,
                method_file_stem=stem,
                epoch=epoch,
            )
            expected_archive = f"snapshots/{stem}/{path.name}"
            expected_method_record = {**record, "archive": expected_archive}
            if (
                record.get("filename") != path.name
                or record.get("bytes") != path.stat().st_size
                or record.get("sha256") != _sha256_file(path)
                or record.get("arrays") != metadata["arrays"]
                or method_record != expected_method_record
            ):
                raise ValueError(
                    f"Snapshot manifest mismatch for {method} epoch {epoch}."
                )

    metadata_keys = _validate_npz(
        directory / "data" / "real_dataset_metadata.npz",
        {
            "selected_pattern_indices",
            "positions_px",
            "probe_initial",
            "dx_m",
            "wavelength_m",
            "source_pattern_count",
            "alignment_fit_mask",
        },
    )
    if any("intensity" in key or key == "data" for key in metadata_keys):
        raise ValueError("Dataset metadata must not duplicate measured intensity data.")
    with np.load(
        directory / "data" / "real_dataset_metadata.npz",
        allow_pickle=False,
    ) as metadata:
        dataset_hashes = manifest.get("dataset_arrays", {})
        if set(dataset_hashes) != set(metadata.files):
            raise ValueError("Dataset-array hash inventory is incomplete.")
        for key in metadata.files:
            if _sha256_array(metadata[key]) != dataset_hashes[key].get("sha256"):
                raise RuntimeError(f"Dataset array hash verification failed for {key}.")
        initial_positions = np.array(metadata["positions_px"], copy=True)
    for method in CANONICAL_REAL_METHODS:
        stem = stems[method]
        reconstruction_path = directory / "reconstructions" / f"{stem}.npz"
        refined_positions, _ = _validate_saved_position_arrays(
            reconstruction_path,
            expected_count=len(initial_positions),
            initial_positions=initial_positions,
        )
        if expected_snapshot_epochs:
            final_snapshot_path = (
                directory
                / "snapshots"
                / stem
                / _snapshot_filename(expected_snapshot_epochs[-1])
            )
            with np.load(final_snapshot_path, allow_pickle=False) as final_snapshot:
                if not np.array_equal(
                    final_snapshot["positions_px"],
                    refined_positions,
                ):
                    raise ValueError(f"Final snapshot positions do not match {method}.")
    required_artifacts = [
        "tables/summary.csv",
        "curves/residual_history.csv",
        "validation_report.json",
        "release_manifest.json",
        "environment.txt",
    ]
    metric_history_path = directory / "curves" / "metric_history.csv"
    if manifest.get("sampled_metric_keys"):
        required_artifacts.append("curves/metric_history.csv")
    elif metric_history_path.exists():
        raise ValueError("Online-only archives must not contain metric_history.csv.")
    for relative in required_artifacts:
        if not (directory / relative).is_file():
            raise FileNotFoundError(f"Missing archive artifact: {relative}")
    with (directory / "tables" / "summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        if len(list(csv.DictReader(handle))) != len(CANONICAL_REAL_METHODS):
            raise RuntimeError("Summary table must contain exactly four rows.")
    source_hdf5 = manifest.get("source_hdf5", {})
    if (
        not isinstance(source_hdf5.get("sha256"), str)
        or len(source_hdf5["sha256"]) != 64
        or int(source_hdf5.get("bytes", 0)) <= 0
    ):
        raise ValueError("run_manifest.json lacks a valid source HDF5 record.")
    if manifest.get("selected_preprocessed_intensity", {}).get("stored_in_archive"):
        raise ValueError("Manifest incorrectly claims diffraction data was archived.")
    return manifest


__all__ = [
    "CANONICAL_REAL_METHODS",
    "REAL_ARCHIVE_SCHEMA_VERSION",
    "REAL_SNAPSHOT_KIND",
    "RealResultArchiveSession",
    "save_real_reconstruction_snapshot",
    "save_real_notebook_results",
    "validate_real_method_checkpoint",
    "validate_real_results_archive",
]
