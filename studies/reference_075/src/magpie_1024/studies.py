"""The 75% parameter sweep and authoritative per-run summaries.

Importing this module does not import Torch or the native reconstructor.
LSQML has sixteen planned pairs and PIE methods have nine. Selection uses final object
NMSE on the predefined ROI after all pairs have completed or failed.
"""
from __future__ import annotations

import hashlib
import fcntl
import json
import math
import os
import tempfile
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd

from .protocol import (BASE_BATCH, BASE_ETA, EPOCHS, METHODS,
                       case_token, float_token, parameter_grid, run_id)

OVERLAP = .75
METRIC_COLUMNS = ("object_nmse", "probe_nmse", "phase_rmse", "online_amplitude_mse",
                  "frozen_amplitude_mse", "frozen_noiseless_amplitude_mse")
TIMING_COLUMNS = ("reconstruction_seconds", "transfer_seconds", "scoring_seconds",
                  "frozen_evaluation_seconds", "history_write_seconds", "total_seconds")


def retained_parameter_grid(method):
    """Sixteen LSQML pairs and nine PIE pairs, preserving original pair IDs."""
    return [dict(p) for p in parameter_grid(method)
            if method == 'lsqml' or (p['object_alpha'] != .5 and p['probe_alpha'] != .5)]


def _results_root(root, smoke=False):
    return Path(root).resolve() / "results" / ("smoke" if smoke else "")


@contextmanager
def method_lock(root, method, smoke=False):
    """Serialize shared summaries, selection, and report rendering per method.

    This lock blocks while another publisher or renderer is active. Individual
    reconstructions retain their separate nonblocking per-pair locks.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    path = _results_root(root, smoke) / ".locks" / f"{method}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _relative_run(method, parameters):
    condition = f"{case_token(OVERLAP)}_eta_{float_token(BASE_ETA)}_b{BASE_BATCH}"
    return Path("runs") / method / condition / run_id(method, parameters)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection_path(root, method, smoke=False):
    """Location of the sweep's selected-pair record and its input hashes."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    return _results_root(root, smoke) / "selections" / method / f"{case_token(OVERLAP)}.json"


def _plan(method, root, smoke=False, seed=21):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    epochs = min(EPOCHS[OVERLAP], 2) if smoke else EPOCHS[OVERLAP]
    rows = []
    for parameters in retained_parameter_grid(method):
        row = {"study": "sweep", "method": method, "algorithm": method,
               "overlap_ratio": OVERLAP, "eta": BASE_ETA, "batch_size": BASE_BATCH,
               **parameters, "run_id": run_id(method, parameters), "epochs": epochs,
               "epochs_completed": 0, "status": "missing", "seed": int(seed),
               "data_seed": 42, "scientific": not smoke,
               "relative_run_dir": _relative_run(method, parameters).as_posix()}
        row.update({name: np.nan for name in METRIC_COLUMNS + TIMING_COLUMNS})
        rows.append(row)
    return rows


def _check_saved_plan(saved, planned, path):
    for key in ("method", "overlap_ratio", "eta", "batch_size", "run_id", "epochs", "seed", "scientific"):
        if saved.get(key) != planned[key]:
            raise ValueError(f"Saved run metadata differs from the requested plan ({key}): {path}")
    for key in ("object_alpha", "probe_alpha", "damping", "sigma"):
        if key in planned and saved.get(key) != planned[key]:
            raise ValueError(f"Saved run has incompatible {key}: {path}")


def load_sweep(method, root, *, smoke=False, seed=21, **kwargs):
    """Read all approved planned rows, including missing and failed results.

    Per-run result.json files are authoritative. CSV summaries are reproducible
    exports, never the input to selection or plotting. Run-only options may be
    passed so notebooks can share the same settings dictionary when loading.
    """
    allowed = {"device", "error_every", "print_every", "rerun_failed", "shard_index", "num_shards"}
    if set(kwargs) - allowed:
        raise TypeError(f"Unexpected options: {sorted(set(kwargs) - allowed)}")
    planned = _plan(method, root, smoke=smoke, seed=seed)
    base = _results_root(root, smoke)
    for row in planned:
        path = base / row["relative_run_dir"] / "result.json"
        if path.is_file():
            saved = json.loads(path.read_text())
            _check_saved_plan(saved, row, path)
            row.update(saved)
            row["study"] = "sweep"
    table = pd.DataFrame(planned)
    table.attrs["all_terminal"] = bool(table.status.isin(["complete", "failed"]).all())
    table.attrs["selection_final"] = bool(table.attrs["all_terminal"] and table.status.eq("complete").any())
    return table


def assigned_run_ids(method, shard_index=None, num_shards=4):
    """Return one shard's interleaved run IDs; None assigns the full grid."""
    if (isinstance(num_shards, bool) or not isinstance(num_shards, (int, np.integer))
            or not 1 <= num_shards <= 16):
        raise ValueError("num_shards must be an integer between 1 and 16")
    if shard_index is not None and (
            isinstance(shard_index, bool) or not isinstance(shard_index, (int, np.integer))
            or not 0 <= shard_index < num_shards):
        raise ValueError("shard_index must be an integer in [0, num_shards), or None")
    return [run_id(method, parameters) for index, parameters in enumerate(parameter_grid(method))
            if parameters in retained_parameter_grid(method)
            and (shard_index is None or index % num_shards == shard_index)]


def _select(root, method, smoke=False, seed=21):
    table = load_sweep(method, root, smoke=smoke, seed=seed)
    if not table.status.isin(["complete", "failed"]).all():
        return None
    completed = table[(table.status == "complete") & np.isfinite(table.object_nmse)
                      & (table.object_nmse >= 0) & (table.epochs_completed == table.epochs)]
    if completed.empty:
        return None
    best = completed.sort_values(["object_nmse", "run_id"], kind="stable").iloc[0]
    parameter_names = ("damping", "sigma") if method == "lsqml" else ("object_alpha", "probe_alpha")
    parameters = {name: float(best[name]) for name in parameter_names}
    base = _results_root(root, smoke)
    candidates = []
    for row in table.to_dict("records"):
        relative = Path(row["relative_run_dir"]) / "result.json"
        config = base / row["relative_run_dir"] / "config.json"
        candidates.append({"run_id": row["run_id"], "status": row["status"],
                           "relative_result_path": relative.as_posix(),
                           "result_sha256": _file_sha(base / relative),
                           "config_sha256": _file_sha(config)})
    selection = {"schema_version": 1, "study": "sweep", "method": method,
                 "overlap_ratio": OVERLAP, "eta": BASE_ETA, "batch_size": BASE_BATCH,
                 "epochs": int(best.epochs), "seed": int(seed),
                 "scientific": not smoke, "metric": "final fixed-ROI object_nmse",
                 "parameters": parameters, "run_id": str(best.run_id),
                 "object_nmse": float(best.object_nmse), "candidate_count": len(table),
                 "completed_count": len(completed), "failed_count": int((table.status == "failed").sum()),
                 "candidates": candidates}
    _write_json(selection_path(root, method, smoke=smoke), selection)
    return selection


def _save_summary(root, method, smoke, table):
    directory = _results_root(root, smoke) / "studies" / "sweep" / method
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
        table.to_csv(stream, index=False)
    try:
        os.replace(temporary, directory / "summary.csv")
    finally:
        temporary.unlink(missing_ok=True)
    records = []
    for row in table.to_dict("records"):
        records.append({key: None if isinstance(value, (float, np.floating)) and not math.isfinite(value) else value
                        for key, value in row.items()})
    _write_json(directory / "summary.json", records)


def verify_completed_pair(method, parameters, root, *, smoke=False, seed=21):
    """Reuse a completed historical pair only with its original sealed records."""
    directory=_results_root(root,smoke)/_relative_run(method,parameters)
    paths={name:directory/name for name in ('config.json','result.json','history.csv')}
    if not all(path.is_file() for path in paths.values()):
        raise ValueError('A completed pair lacks its configuration or history')
    config=json.loads(paths['config.json'].read_text())
    result=json.loads(paths['result.json'].read_text())
    expected=dict(method=method,parameters=parameters,overlap_ratio=OVERLAP,
        eta=BASE_ETA,batch_size=BASE_BATCH,epochs=2 if smoke else EPOCHS[OVERLAP],
        seed=int(seed),smoke=smoke,scientific=not smoke,error_every=20,print_every=300)
    if any(config.get(key)!=value for key,value in expected.items()):
        raise ValueError('Completed pair configuration changed')
    current={name:_file_sha(Path(__file__).parent/name)
             for name in ('engine.py','studies.py','data.py','protocol.py')}
    saved=config['runner_source_sha256']
    if saved!=current:
        if smoke or set(saved)!=set(current) or any(saved[name]!=digest for name,digest in current.items() if name!='studies.py'):
            raise ValueError('Completed pair numerical sources changed')
        project=Path(__file__).resolve().parents[4]
        manifest=Path(os.environ.get('MAGPIE_SHORT_NOISE_MANIFEST','provenance/short_noise_parameters.json'))
        manifest=manifest if manifest.is_absolute() else project/manifest
        frozen=json.loads(manifest.read_text())
        selection=next((entry for entry in frozen['selections']
                        if entry['method']==method and entry['overlap_ratio']==OVERLAP),None)
        candidate=next((entry for entry in (selection or {}).get('candidates',[])
                        if entry['parameters']==parameters),None)
        if candidate is None:
            raise ValueError('No sealed historical source proof for this retained pair')
        hashes={Path(name).name:digest for name,digest in candidate['source_files_sha256'].items()}
        if any(hashes.get(name)!=_file_sha(path) for name,path in paths.items()):
            raise ValueError('Completed pair records differ from their historical proof')
    if result['status']!='complete' or result['epochs_completed']!=expected['epochs']:
        raise ValueError('Completed pair endpoint differs from its budget')
    return result


def run_sweep(method, root, *, device="cuda", smoke=False, seed=21,
              error_every=20, print_every=300, rerun_failed=False,
              shard_index=None, num_shards=4):
    """Run or reuse the approved method-specific pairs, then select the smallest final ROI error.

    Completed compatible pairs are reused; interrupted attempts restart from
    initialization and are archived. Failed pairs are retained unless
    rerun_failed=True. Smoke outputs and selections are explicitly non-scientific.
    A shard runs indices i % num_shards == shard_index. The returned table and
    published summaries always contain the full method-specific plan; its
    assigned_run_ids attribute identifies only this caller's work.
    """
    assigned = assigned_run_ids(method, shard_index=shard_index, num_shards=num_shards)
    runner = None
    selected = None
    table = load_sweep(method, root, smoke=smoke, seed=seed)
    owned = [row for row in table.to_dict("records") if row["run_id"] in assigned]
    for index, row in enumerate(owned, 1):
        names = ("damping", "sigma") if method == "lsqml" else ("object_alpha", "probe_alpha")
        parameters = {name: row[name] for name in names}
        print(f"[{index}/{len(owned)}] sweep: {method}, overlap={OVERLAP:.1%}, "
              f"eta={BASE_ETA:g}, batch={BASE_BATCH}, {parameters}", flush=True)
        if row['status']=='complete':
            verify_completed_pair(method,parameters,root,smoke=smoke,seed=seed)
        elif row['status']=='failed' and not rerun_failed:
            print(f"Preserving recorded failed pair {row['run_id']}", flush=True)
        else:
            if runner is None:
                from .engine import check_native_source, run_pair
                check_native_source()
                runner=run_pair
            runner(method, parameters, root=root, overlap_ratio=OVERLAP,
                 eta=BASE_ETA, batch_size=BASE_BATCH, epochs=row["epochs"],
                 device=device, smoke=smoke, seed=seed, error_every=error_every,
                 print_every=print_every, rerun_failed=rerun_failed)
        with method_lock(root, method, smoke=smoke):
            table = load_sweep(method, root, smoke=smoke, seed=seed)
            _save_summary(root, method, smoke, table)
            if table.attrs["all_terminal"]:
                selected = _select(root, method, smoke=smoke, seed=seed)
    table.attrs["assigned_run_ids"] = assigned
    table.attrs["shard_index"] = shard_index
    table.attrs["num_shards"] = int(num_shards)
    if selected:
        print(f"Selected {method}: {selected['parameters']}, "
              f"final ROI object NMSE={selected['object_nmse']:.5g}", flush=True)
    return table
