from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


ARCHIVE_SCHEMA_VERSION = 1
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        _jsonable(payload),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            _jsonable(payload),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    json.loads(path.read_text(encoding="utf-8"))


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="raise",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _jsonable(row[name]) for name in fieldnames})
    temporary.replace(path)

    with path.open("r", encoding="utf-8", newline="") as handle:
        saved_rows = list(csv.DictReader(handle))
    if len(saved_rows) != len(rows):
        raise RuntimeError(f"CSV row-count verification failed for {path}.")


def _atomic_savez(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = {name: np.asarray(value) for name, value in arrays.items()}
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **expected)
    temporary.replace(path)

    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(expected):
            raise RuntimeError(f"NPZ key verification failed for {path}.")
        for name, value in expected.items():
            if not np.array_equal(saved[name], value):
                raise RuntimeError(
                    f"NPZ round-trip verification failed for {path}:{name}."
                )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: Any) -> str:
    contiguous = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(_canonical_json(contiguous.shape).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _notebook_source_sha256(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    semantic_cells = []
    for cell in document.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        semantic_cells.append(
            {
                "cell_type": str(cell.get("cell_type", "")),
                "source": str(source),
            }
        )
    return hashlib.sha256(
        _canonical_json(
            {
                "nbformat": int(document.get("nbformat", 0)),
                "cells": semantic_cells,
            }
        ).encode("utf-8")
    ).hexdigest()


def _git_output(project_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.rstrip()


def _software_versions() -> dict[str, str | None]:
    distributions = {
        "blind-magpie": ("blind-magpie",),
        "imageio": ("imageio",),
        "matplotlib": ("matplotlib",),
        "numpy": ("numpy",),
        "ptychi": ("ptychi", "pty-chi"),
        "scikit-image": ("scikit-image",),
        "torch": ("torch",),
    }
    versions: dict[str, str | None] = {}
    for label, candidates in distributions.items():
        versions[label] = None
        for distribution in candidates:
            try:
                versions[label] = importlib_metadata.version(distribution)
                break
            except importlib_metadata.PackageNotFoundError:
                continue
    return versions


def _pip_freeze() -> str:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout if completed.returncode == 0 else ""


def _finite_array(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError(f"{label} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains nonfinite values.")
    return array


def _finite_scalar(value: Any, *, label: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite.")
    return numeric


def _validate_export_state(
    *,
    shared_config: Any,
    method_order: Sequence[str],
    method_file_stems: Mapping[str, str],
    results: Mapping[str, Any],
    runtimes_seconds: Mapping[str, float],
    scores: Mapping[str, Mapping[str, float]],
    aligned_objects: Mapping[str, np.ndarray],
    aligned_probes: Mapping[str, np.ndarray],
    registered_truth: np.ndarray,
    illuminated: np.ndarray,
    dataset: Any,
    noiseless_dataset: Any,
    metric_report_every: int,
    include_initial_metrics: bool,
    sampled_metric_keys: Sequence[str],
    final_metric_keys: Sequence[str],
    truth_metric_keys: Sequence[str],
) -> None:
    if isinstance(metric_report_every, bool) or not isinstance(
        metric_report_every,
        int,
    ):
        raise TypeError("metric_report_every must be an integer.")
    if metric_report_every < 1:
        raise ValueError("metric_report_every must be positive.")
    if not isinstance(include_initial_metrics, bool):
        raise TypeError("include_initial_metrics must be boolean.")

    methods = tuple(method_order)
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("method_order must contain unique method names.")
    expected_methods = set(methods)
    named_method_mappings = {
        "method_file_stems": method_file_stems,
        "results": results,
        "runtimes_seconds": runtimes_seconds,
        "scores": scores,
        "aligned_objects": aligned_objects,
        "aligned_probes": aligned_probes,
    }
    for name, mapping in named_method_mappings.items():
        if set(mapping) != expected_methods:
            raise ValueError(f"{name} must contain exactly: {', '.join(methods)}.")
    stems = tuple(method_file_stems[method] for method in methods)
    if any(not _SAFE_RUN_ID.fullmatch(stem) for stem in stems):
        raise ValueError("Method file stems must be safe path components.")
    if len(stems) != len(set(stems)):
        raise ValueError("Method file stems must be unique.")

    registered_truth = _finite_array(
        registered_truth,
        label="registered_truth",
    )
    illuminated = np.asarray(illuminated, dtype=bool)
    if illuminated.shape != registered_truth.shape or not np.any(illuminated):
        raise ValueError("illuminated must be a nonempty mask over registered_truth.")

    probe_truth = _finite_array(dataset.probe_truth, label="dataset.probe_truth")
    dataset_fields = (
        "truth",
        "probe_truth",
        "probe_init",
        "data",
        "positions_px",
        "position_shifts_px",
    )
    invariant_dataset_fields = {
        "truth",
        "probe_truth",
        "positions_px",
        "position_shifts_px",
    }
    for field in dataset_fields:
        noisy_value = _finite_array(
            getattr(dataset, field),
            label=f"dataset.{field}",
        )
        noiseless_value = _finite_array(
            getattr(noiseless_dataset, field),
            label=f"noiseless_dataset.{field}",
        )
        if field in invariant_dataset_fields and not np.array_equal(
            noisy_value,
            noiseless_value,
        ):
            raise ValueError(f"Noisy and noiseless dataset {field} values differ.")
    if np.asarray(dataset.data).shape != np.asarray(noiseless_dataset.data).shape:
        raise ValueError("Noisy and noiseless dataset shapes differ.")

    sampled_keys = set(sampled_metric_keys)
    final_keys = set(final_metric_keys)
    truth_keys = set(truth_metric_keys)
    expected_epochs = int(getattr(shared_config, "num_epochs"))
    expected_metric_epochs = np.asarray(
        sorted(
            {
                1,
                expected_epochs,
                *range(metric_report_every, expected_epochs + 1, metric_report_every),
                *((0,) if include_initial_metrics else ()),
            }
        ),
        dtype=np.int64,
    )
    for method in methods:
        result = results[method]
        residual = _finite_array(
            result.residual_history,
            label=f"{method} residual_history",
        )
        if residual.ndim != 1:
            raise ValueError(f"{method} residual_history must be one dimensional.")
        if residual.shape != (expected_epochs,):
            raise ValueError(f"{method} residual_history has the wrong length.")
        metric_epochs = _finite_array(
            result.metric_epochs,
            label=f"{method} metric_epochs",
        )
        if metric_epochs.ndim != 1:
            raise ValueError(f"{method} metric_epochs must be one dimensional.")
        if not np.issubdtype(metric_epochs.dtype, np.integer) or not np.array_equal(
            metric_epochs,
            expected_metric_epochs,
        ):
            raise ValueError(f"{method} metric_epochs do not match the saved cadence.")
        if set(result.metric_history) != sampled_keys:
            raise ValueError(f"{method} has unexpected sampled metric keys.")
        for key, values in result.metric_history.items():
            metric_values = _finite_array(
                values,
                label=f"{method} metric_history[{key!r}]",
            )
            if metric_values.shape != metric_epochs.shape:
                raise ValueError(f"{method} metric {key!r} has the wrong shape.")
        if set(result.final_metrics) != final_keys:
            raise ValueError(f"{method} has unexpected final metric keys.")
        for key, value in result.final_metrics.items():
            _finite_scalar(value, label=f"{method} final_metrics[{key!r}]")
        if set(scores[method]) != truth_keys:
            raise ValueError(f"{method} has unexpected endpoint score keys.")
        for key, value in scores[method].items():
            _finite_scalar(value, label=f"{method} scores[{key!r}]")
            np.testing.assert_allclose(
                value,
                result.metric_history[key][-1],
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"{method} endpoint score {key!r} is stale.",
            )
        for key, value in result.final_metrics.items():
            np.testing.assert_allclose(
                value,
                result.metric_history[key][-1],
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"{method} final metric {key!r} is stale.",
            )
        runtime = _finite_scalar(
            runtimes_seconds[method],
            label=f"{method} runtime_seconds",
        )
        if runtime <= 0:
            raise ValueError(f"{method} runtime_seconds must be positive.")
        object_aligned = _finite_array(
            aligned_objects[method],
            label=f"{method} aligned_object",
        )
        probe_aligned = _finite_array(
            aligned_probes[method],
            label=f"{method} aligned_probe",
        )
        object_raw = _finite_array(result.object, label=f"{method} object")
        probe_raw = _finite_array(result.probe, label=f"{method} probe")
        for label, value in (
            ("object", object_raw),
            ("probe", probe_raw),
            ("aligned_object", object_aligned),
            ("aligned_probe", probe_aligned),
        ):
            if not np.iscomplexobj(value):
                raise TypeError(f"{method} {label} must be complex-valued.")
        if object_aligned.shape != registered_truth.shape:
            raise ValueError(f"{method} aligned object has the wrong shape.")
        if probe_aligned.shape != probe_truth.shape:
            raise ValueError(f"{method} aligned probe has the wrong shape.")


def save_synthetic_notebook_results(
    *,
    output_root: Path,
    project_root: Path,
    experiment_name: str,
    notebook_path: Path,
    run_started_utc: datetime,
    dataset_config: Any,
    shared_config: Any,
    method_order: Sequence[str],
    method_file_stems: Mapping[str, str],
    method_parameters: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Any],
    runtimes_seconds: Mapping[str, float],
    scores: Mapping[str, Mapping[str, float]],
    aligned_objects: Mapping[str, np.ndarray],
    aligned_probes: Mapping[str, np.ndarray],
    registered_truth: np.ndarray,
    illuminated: np.ndarray,
    dataset: Any,
    noiseless_dataset: Any,
    metric_report_every: int,
    include_initial_metrics: bool,
    frozen_metric_eps: float,
    sampled_metric_keys: Sequence[str],
    final_metric_keys: Sequence[str],
    truth_metric_keys: Sequence[str],
    device: Any,
    source_paths: Mapping[str, Path],
    metric_definitions: Mapping[str, str],
    timing_notes: str,
    dataset_extra_arrays: Mapping[str, Any] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> Path:
    """Save and verify one complete synthetic comparison without overwriting it."""
    _validate_export_state(
        shared_config=shared_config,
        method_order=method_order,
        method_file_stems=method_file_stems,
        results=results,
        runtimes_seconds=runtimes_seconds,
        scores=scores,
        aligned_objects=aligned_objects,
        aligned_probes=aligned_probes,
        registered_truth=registered_truth,
        illuminated=illuminated,
        dataset=dataset,
        noiseless_dataset=noiseless_dataset,
        metric_report_every=metric_report_every,
        include_initial_metrics=include_initial_metrics,
        sampled_metric_keys=sampled_metric_keys,
        final_metric_keys=final_metric_keys,
        truth_metric_keys=truth_metric_keys,
    )
    if set(method_parameters) != set(method_order):
        raise ValueError("method_parameters must contain every method exactly once.")
    frozen_metric_eps = _finite_scalar(
        frozen_metric_eps,
        label="frozen_metric_eps",
    )
    if frozen_metric_eps < 0:
        raise ValueError("frozen_metric_eps must be nonnegative.")
    expected_definition_keys = set(sampled_metric_keys) | {"online_preupdate_residual"}
    if set(metric_definitions) != expected_definition_keys or any(
        not isinstance(definition, str) or not definition.strip()
        for definition in metric_definitions.values()
    ):
        raise ValueError("metric_definitions must define every archived metric.")
    if not isinstance(run_started_utc, datetime) or run_started_utc.tzinfo is None:
        raise ValueError("run_started_utc must be a timezone-aware datetime.")
    if dataset_extra_arrays is None:
        dataset_extra_arrays = {}
    elif not isinstance(dataset_extra_arrays, Mapping):
        raise TypeError("dataset_extra_arrays must be a mapping or None.")
    if run_metadata is None:
        run_metadata = {}
    elif not isinstance(run_metadata, Mapping):
        raise TypeError("run_metadata must be a mapping or None.")
    _canonical_json(run_metadata)

    project_root = Path(project_root).resolve()
    output_root = Path(output_root).resolve()
    notebook_path = Path(notebook_path).resolve()
    resolved_sources = {
        str(name): Path(path).resolve() for name, path in source_paths.items()
    }
    resolved_sources.setdefault("notebook", notebook_path)
    missing_sources = [
        name for name, path in resolved_sources.items() if not path.is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(
            "Missing source files: " + ", ".join(sorted(missing_sources))
        )

    saved_utc = datetime.now(timezone.utc)
    if run_id is None:
        run_id = saved_utc.strftime("%Y%m%dT%H%M%S_%fZ")
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must be one safe path component.")

    final_directory = output_root / run_id
    if final_directory.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result archive: {final_directory}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=output_root))

    try:
        detector_element_count = int(np.asarray(dataset.data).size)
        expected_epochs = int(getattr(shared_config, "num_epochs"))
        batch_size = int(getattr(shared_config, "batch_size"))
        position_seed = int(getattr(dataset_config, "position_seed"))
        data_seed = int(getattr(dataset_config, "data_seed"))
        reconstruction_seed = int(getattr(shared_config, "reconstruction_seed"))
        summary_rows = []
        metric_rows = []
        residual_rows = []
        for method in method_order:
            result = results[method]
            summary_row: dict[str, Any] = {
                "method": method,
                "runtime_seconds": float(runtimes_seconds[method]),
                "final_online_preupdate_residual": float(
                    np.asarray(result.residual_history)[-1]
                ),
                "num_epochs": expected_epochs,
                "batch_size": batch_size,
                "position_seed": position_seed,
                "data_seed": data_seed,
                "reconstruction_seed": reconstruction_seed,
            }
            summary_row.update(
                {key: float(scores[method][key]) for key in truth_metric_keys}
            )
            summary_row.update(
                {key: float(result.final_metrics[key]) for key in final_metric_keys}
            )
            summary_rows.append(summary_row)

            for index, epoch in enumerate(np.asarray(result.metric_epochs)):
                row: dict[str, Any] = {
                    "method": method,
                    "epoch": int(epoch),
                }
                row.update(
                    {
                        key: float(result.metric_history[key][index])
                        for key in sampled_metric_keys
                    }
                )
                metric_rows.append(row)
            residual_rows.extend(
                {
                    "method": method,
                    "epoch": epoch,
                    "online_preupdate_residual": float(value),
                }
                for epoch, value in enumerate(
                    np.asarray(result.residual_history),
                    start=1,
                )
            )

            reconstruction_payload: dict[str, Any] = {
                "object": result.object,
                "probe": result.probe,
                "aligned_object": aligned_objects[method],
                "aligned_probe": aligned_probes[method],
                "residual_history": result.residual_history,
                "metric_epochs": result.metric_epochs,
                "runtime_seconds": np.asarray(runtimes_seconds[method]),
                "registered_object_truth": registered_truth,
                "illuminated_mask": np.asarray(illuminated, dtype=bool),
                "metadata_json": np.asarray(
                    _canonical_json(
                        {
                            "schema_version": ARCHIVE_SCHEMA_VERSION,
                            "experiment_name": experiment_name,
                            "run_id": run_id,
                            "method": method,
                            "parameters": method_parameters[method],
                        }
                    )
                ),
            }
            reconstruction_payload.update(
                {
                    f"metric_history__{key}": result.metric_history[key]
                    for key in sampled_metric_keys
                }
            )
            reconstruction_payload.update(
                {
                    f"final_metric__{key}": np.asarray(result.final_metrics[key])
                    for key in final_metric_keys
                }
            )
            reconstruction_payload.update(
                {
                    f"score__{key}": np.asarray(scores[method][key])
                    for key in truth_metric_keys
                }
            )
            _atomic_savez(
                temporary_directory
                / "reconstructions"
                / f"{method_file_stems[method]}.npz",
                reconstruction_payload,
            )
            _write_json(
                temporary_directory / "manifests" / f"{method_file_stems[method]}.json",
                {
                    "schema_version": ARCHIVE_SCHEMA_VERSION,
                    "experiment_name": experiment_name,
                    "run_id": run_id,
                    "method": method,
                    "archive": (f"reconstructions/{method_file_stems[method]}.npz"),
                    "parameters": method_parameters[method],
                    "runtime_seconds": runtimes_seconds[method],
                    "scores": scores[method],
                    "final_metrics": result.final_metrics,
                    "array_sha256": {
                        name: _sha256_array(value)
                        for name, value in reconstruction_payload.items()
                        if name != "metadata_json"
                    },
                },
            )

        summary_fields = [
            "method",
            "runtime_seconds",
            "final_online_preupdate_residual",
            "num_epochs",
            "batch_size",
            "position_seed",
            "data_seed",
            "reconstruction_seed",
            *truth_metric_keys,
            *final_metric_keys,
        ]
        _write_csv(
            temporary_directory / "tables" / "summary.csv",
            summary_fields,
            summary_rows,
        )
        _write_csv(
            temporary_directory / "curves" / "metric_history.csv",
            ["method", "epoch", *sampled_metric_keys],
            metric_rows,
        )
        _write_csv(
            temporary_directory / "curves" / "residual_history.csv",
            ["method", "epoch", "online_preupdate_residual"],
            residual_rows,
        )

        dataset_payload = {
            "object_truth": dataset.truth,
            "probe_truth": dataset.probe_truth,
            "probe_initial": dataset.probe_init,
            "noisy_intensity": dataset.data,
            "noiseless_intensity": noiseless_dataset.data,
            "positions_px": dataset.positions_px,
            "position_shifts_px": dataset.position_shifts_px,
            "registered_object_truth": registered_truth,
            "illuminated_mask": np.asarray(illuminated, dtype=bool),
        }
        for name, value in dataset_extra_arrays.items():
            if not isinstance(name, str) or not _SAFE_RUN_ID.fullmatch(name):
                raise ValueError("Dataset extra-array names must be safe identifiers.")
            if name in dataset_payload:
                raise ValueError(
                    f"Dataset extra array {name!r} collides with a core key."
                )
            array = _finite_array(value, label=f"dataset_extra_arrays[{name!r}]")
            if array.dtype.hasobject:
                raise TypeError(
                    f"Dataset extra array {name!r} must not use object dtype."
                )
            dataset_payload[name] = array
        _atomic_savez(
            temporary_directory / "data" / "synthetic_dataset.npz",
            dataset_payload,
        )

        source_hashes = {
            name: {
                "path": (
                    str(path.relative_to(project_root))
                    if path.is_relative_to(project_root)
                    else str(path)
                ),
                "sha256": _sha256_file(path),
            }
            for name, path in sorted(resolved_sources.items())
        }
        source_hashes["notebook"]["semantic_source_sha256"] = _notebook_source_sha256(
            notebook_path
        )
        git_status = _git_output(project_root, "status", "--porcelain")
        run_manifest = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "experiment_name": experiment_name,
            "run_id": run_id,
            "run_started_utc": run_started_utc.astimezone(timezone.utc).isoformat(),
            "saved_utc": saved_utc.isoformat(),
            "device": device,
            "method_order": list(method_order),
            "method_file_stems": method_file_stems,
            "method_parameters": method_parameters,
            "metric_definitions": metric_definitions,
            "metric_report_every": metric_report_every,
            "include_initial_metrics": include_initial_metrics,
            "frozen_metric_eps": frozen_metric_eps,
            "sampled_metric_keys": list(sampled_metric_keys),
            "final_metric_keys": list(final_metric_keys),
            "truth_metric_keys": list(truth_metric_keys),
            "dataset_config": dataset_config,
            "shared_config": shared_config,
            "run_metadata": run_metadata,
            "dataset_arrays": {
                name: {
                    "shape": list(np.asarray(value).shape),
                    "dtype": str(np.asarray(value).dtype),
                    "sha256": _sha256_array(value),
                }
                for name, value in dataset_payload.items()
            },
            "runtime_seconds": runtimes_seconds,
            "timing_notes": timing_notes,
            "software_versions": _software_versions(),
            "python": {
                "version": sys.version,
                "implementation": platform.python_implementation(),
            },
            "platform": platform.platform(),
            "git": {
                "commit": _git_output(project_root, "rev-parse", "HEAD"),
                "status_porcelain": ([] if not git_status else git_status.splitlines()),
            },
            "source_files": source_hashes,
        }
        _write_json(temporary_directory / "run_manifest.json", run_manifest)
        _atomic_write_text(
            temporary_directory / "environment.txt",
            _pip_freeze(),
        )

        validation_report = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "status": "passed",
            "methods": list(method_order),
            "method_count": len(method_order),
            "summary_rows": len(summary_rows),
            "sampled_metric_rows": len(metric_rows),
            "online_residual_rows": len(residual_rows),
            "detector_element_count_per_metric": detector_element_count,
            "checks": [
                "all required methods, runtimes, and scores present",
                "all arrays and scalar metrics finite",
                "all metric keys and epoch sequences complete",
                "endpoint scores agree with final sampled truth metrics",
                "final frozen metrics agree with final sampled frozen metrics",
                "raw and aligned reconstructions are complex-valued",
                "NPZ files round-trip exactly with allow_pickle=False",
                "CSV files round-trip with the expected row counts",
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
        release_manifest = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_count": len(artifact_hashes),
            "artifacts": artifact_hashes,
        }
        release_manifest_path = temporary_directory / "release_manifest.json"
        _write_json(release_manifest_path, release_manifest)

        checksum_paths = [
            path
            for path in sorted(temporary_directory.rglob("*"))
            if path.is_file()
            and path.name != "checksums.sha256"
            and not path.name.startswith(".")
        ]
        checksum_lines = [
            f"{_sha256_file(path)}  "
            f"{PurePosixPath(path.relative_to(temporary_directory)).as_posix()}"
            for path in checksum_paths
        ]
        _atomic_write_text(
            temporary_directory / "checksums.sha256",
            "\n".join(checksum_lines) + "\n",
        )

        _atomic_write_text(
            temporary_directory / "_SUCCESS",
            f"{run_id}\n",
        )

        temporary_directory.replace(final_directory)
    except BaseException:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise

    return final_directory
