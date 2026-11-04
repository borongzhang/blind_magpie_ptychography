"""Portable access to saved records and the original scientific runners.

Inspection reads the curated ``results/index.json`` and ``results/records``.
Reruns write to the scientific runners' separate native directories and never
replace the curated index or publication figures.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


STUDIES = ("overlap", "noise", "batch", "real")
METHODS = ("rpie", "lsqml", "gm_rpie", "gm_magpie")
OVERLAPS = (.75, .875, .9375)


def repository_root(root=None):
    """Find the checkout containing the publication index."""
    if root is not None:
        path = Path(root).expanduser().resolve()
        if not (path / "results/index.json").is_file():
            raise FileNotFoundError(f"No publication index at {path / 'results/index.json'}")
        return path
    candidates = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]]
    for path in dict.fromkeys(candidates):
        if (path / "results/index.json").is_file():
            return path
    raise FileNotFoundError("Run from this checkout, or supply root=the checkout path.")


def _inside(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"An indexed path must be relative to the checkout: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"An indexed path leaves the checkout: {relative}")
    return path


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_index(root=None):
    """Return the complete publication index as a dictionary."""
    root = repository_root(root)
    index = json.loads((root / "results/index.json").read_text())
    cases = index.get("cases")
    if not isinstance(cases, list) or len({row["id"] for row in cases}) != len(cases):
        raise ValueError("Publication cases must be a list with unique IDs.")
    return index


def _matches(value, expected):
    if expected is None:
        return True
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return value is not None and math.isclose(float(value), float(expected), rel_tol=0, abs_tol=1e-12)
    return value == expected


def select_cases(study=None, *, overlap=None, eta=None, batch_size=None,
                 dataset=None, method=None, selected=None, root=None):
    """Select saved cases, including reused baselines in logical studies.

    ``selected=True`` means the method-specific object-selected pair for the
    requested study. Noise includes the reused eta=0.1 baselines; batch includes
    the reused batch-27 overlap sweep. Real data have no quality-based selection.
    """
    index = load_index(root)
    if study is not None and study not in STUDIES:
        raise ValueError(f"study must be one of {STUDIES}")
    if method == "all":
        method = None
    if method is not None and method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    allowed = None if study is None else set(index["studies"][study]["case_ids"])
    if selected is None:
        selected_ids = None
    elif study is None:
        selected_ids = set().union(*(set(values) for values in index["selections"].values()))
    else:
        selected_ids = set(index["selections"][study])
    rows = []
    for row in index["cases"]:
        if allowed is not None and row["id"] not in allowed:
            continue
        if selected_ids is not None and (row["id"] in selected_ids) != selected:
            continue
        if all(_matches(row.get(key), expected) for key, expected in (
                ("overlap_ratio", overlap), ("eta", eta), ("batch_size", batch_size),
                ("dataset", dataset), ("method", method))):
            rows.append(row)
    return rows


def file_path(case, kind, *, root=None):
    """Resolve one of a case's indexed config/history/result/reconstruction files."""
    try:
        relative = case["files"][kind]
    except KeyError as error:
        raise FileNotFoundError(f"No {kind} file is published for case {case.get('id', '?')}") from error
    path = _inside(repository_root(root), relative)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _verify_case_file(case, kind, path):
    expected = case.get("files_sha256", {}).get(kind)
    expected = expected or case.get("source_identity", {}).get(f"{kind}_sha256")
    if expected is not None and _sha(path) != expected:
        raise ValueError(f"Published {kind} checksum differs for {case['id']}")


def read_history(case, *, root=None):
    """Read an unchanged history CSV; unrecorded truth-error entries stay NaN."""
    import pandas as pd
    path = file_path(case, "history", root=root)
    _verify_case_file(case, "history", path)
    return pd.read_csv(path, float_precision="round_trip")


def load_fields(case, *, root=None):
    """Load a published selected NPZ, preserving every original array value."""
    import numpy as np
    path = file_path(case, "reconstruction", root=root)
    _verify_case_file(case, "reconstruction", path)
    with np.load(path, allow_pickle=False) as archive:
        fields = {key: archive[key] for key in archive.files}
    if any(not np.isfinite(value).all() for value in fields.values() if np.issubdtype(value.dtype, np.number)):
        raise ValueError(f"Nonfinite values in {case['id']}")
    return fields


def run_command(study, *, overlap=None, dataset=None, smoke=False,
                device=None, method="all", batch_size=None, root=None):
    """Build the fresh-process invocation without importing Torch or Pty-Chi."""
    root = repository_root(root)
    if study not in STUDIES:
        raise ValueError(f"study must be one of {STUDIES}")
    if overlap is not None and overlap not in OVERLAPS:
        raise ValueError(f"overlap must be one of {OVERLAPS}")
    if dataset is not None and dataset not in ("test_pattern", "integrated_circuit"):
        raise ValueError("dataset must be test_pattern or integrated_circuit")
    if dataset is not None and study != "real":
        raise ValueError("dataset is only meaningful for the real study")
    if study == "real" and overlap is not None:
        raise ValueError("The real study does not use a synthetic overlap filter")
    if study == "batch" and overlap not in (None, .875):
        raise ValueError("The batch study uses only 87.5% overlap")
    if batch_size is not None and (study != "batch" or batch_size not in (27, 108, 432)):
        raise ValueError("batch_size is only valid for batch, with 27, 108, or 432")
    if method not in (*METHODS, "all"):
        raise ValueError(f"method must be one of {(*METHODS, 'all')}")
    device = ("cpu" if smoke else "cuda") if device is None else device
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    command = [sys.executable, "-m", "magpie_publication._runner", "--root", str(root),
               "--study", study, "--device", device, "--method", method]
    for flag, value in (("--overlap", overlap), ("--dataset", dataset), ("--batch-size", batch_size)):
        if value is not None:
            command.extend([flag, str(value)])
    if smoke:
        command.append("--smoke")
    return command


def run_study(study, *, overlap=None, dataset=None, smoke=False, device=None,
              method="all", batch_size=None, dry_run=False, root=None):
    """Run the original scientific implementation in a fresh subprocess.

    Smoke mode defaults to CPU and uses separate tiny-data output directories.
    Full runs default to CUDA. Neither mode refreshes the archived publication
    index: use the native output directories printed by the runner for new runs.
    """
    root = repository_root(root)
    if dry_run:
        return rerun_plan(study, overlap=overlap, dataset=dataset, smoke=smoke,
                          device=device, method=method, batch_size=batch_size, root=root)
    command = run_command(study, overlap=overlap, dataset=dataset, smoke=smoke,
                          device=device, method=method, batch_size=batch_size, root=root)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src") + (os.pathsep + environment["PYTHONPATH"]
                                                    if environment.get("PYTHONPATH") else "")
    return subprocess.run(command, cwd=root, env=environment, text=True, check=True)


def rerun_plan(study, *, overlap=None, dataset=None, smoke=False, device=None,
               method="all", batch_size=None, root=None):
    """Preview routes, baseline preparation, and output paths without a solver."""
    root = repository_root(root)
    command = run_command(study, overlap=overlap, dataset=dataset, smoke=smoke,
                          device=device, method=method, batch_size=batch_size, root=root)
    rows = select_cases(study, overlap=overlap, dataset=dataset, method=method,
                        batch_size=batch_size, root=root)
    if study == "noise":
        rows = [row for row in rows if row["eta"] in (.2, .4)]
    if study == "batch" and batch_size is None:
        rows = [row for row in rows if row["batch_size"] in (108, 432)]
    steps, outputs = [], []
    overlaps = OVERLAPS if overlap is None else (overlap,)
    if study in ("noise", "batch"):
        prepare = (.875,) if study == "batch" else overlaps
        for value in prepare:
            steps.append(f"Generate and validate eta=0.1 baseline inputs at {value * 100:g}% overlap")
            if value == .9375:
                steps.append("Generate the 87.5% baseline first, then the nested 93.75% baseline")
    if study == "overlap":
        for value in overlaps:
            if value == .75:
                steps.append("Run the frozen 75% package in its own fresh Python namespace")
                outputs.append("studies/reference_075/results")
            elif value == .875:
                steps.append("Run magpie_1024.overlap_sweep with original historical task IDs")
                outputs.append("results/runs")
            else:
                steps.append("Generate the 87.5% baseline, then run magpie_1024.high_overlap_sweep")
                outputs.append("evaluations/high_overlap_sweep/results")
    elif study == "noise":
        steps.append("Verify the frozen selected pairs, then run magpie_1024.short_noise at eta=0.2 and 0.4")
        outputs.append("evaluations/noise_short/results")
    elif study == "batch":
        steps.append("Run the 87.5% overlap sweep" if batch_size == 27
                     else "Verify the frozen batch-27 records, then run magpie_1024.batch_sweep")
        outputs.append("results/runs" if batch_size == 27 else "evaluations/batch_0875_sweep/results")
    else:
        steps.append("Load the local raw HDF5 inputs, then run magpie_1024.real with the paper's fixed settings")
        outputs.append("evaluations/real/results")
    if smoke:
        steps.append("Use separate smoke directories and the original tiny-data protocol")
    return dict(study=study, overlap=overlap, dataset=dataset, method=method,
                batch_size=batch_size, smoke=bool(smoke), device="cpu" if smoke and device is None else device or "cuda",
                scientific_cases=len(rows), command=command, steps=steps,
                native_output_directories=outputs, curated_index_and_figures_unchanged=True)
