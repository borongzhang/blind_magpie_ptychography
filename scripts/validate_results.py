"""Validate the published files, experiment grids, selections, and saved fields."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("rpie", "lsqml", "gm_rpie", "gm_magpie")
EPOCHS = {.75: 3000, .875: 800, .9375: 200}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def local_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path leaves the checkout: {value}")
    return path


def parameters(row):
    return tuple(sorted(row["parameters"].items()))


def check_grid(rows, selected_ids):
    for method in METHODS:
        candidates = [row for row in rows if row["method"] == method]
        if method == "lsqml":
            expected = {tuple(sorted(dict(damping=d, sigma=s).items()))
                        for d, s in itertools.product((.04, .1, .2, 1.), (.3, .4, .5, .6))}
        else:
            expected = {tuple(sorted(dict(object_alpha=a, probe_alpha=p).items()))
                        for a, p in itertools.product((.004, .02, .1), repeat=2)}
        assert len(candidates) == len(expected)
        assert {parameters(row) for row in candidates} == expected
        completed = [row for row in candidates if row["status"] == "complete"]
        best = min(completed, key=lambda row: row["object_nmse"])
        selected = [row for row in candidates if row["id"] in selected_ids]
        assert len(selected) == 1 and selected[0]["object_nmse"] == best["object_nmse"]


def main():
    index = json.loads((ROOT / "results/index.json").read_text())
    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    for relative, entry in manifest["files"].items():
        path = local_path(relative)
        assert path.stat().st_size == entry["bytes"], relative
        assert digest(path) == entry["sha256"], relative

    cases = index["cases"]
    by_id = {row["id"]: row for row in cases}
    assert len(cases) == len(by_id) == 247
    assert Counter(row["status"] for row in cases) == {"complete": 238, "failed": 9}
    assert Counter(row["study"] for row in cases) == {"overlap": 129, "noise": 24, "batch": 86, "real": 8}
    for case in cases:
        for relative in case["files"].values():
            assert local_path(relative).is_file(), relative
        with local_path(case["files"]["history"]).open(newline="") as stream:
            history = list(csv.DictReader(stream))
        epochs = [int(row["epoch"]) for row in history]
        assert epochs == list(range(epochs[-1] + 1)), case["id"]
        if case["status"] == "complete":
            assert epochs[-1] == case["epochs_completed"] == case["epochs"]
            if case["study"] != "real":
                value = case["object_nmse"]
                assert math.isfinite(value) and value >= 0
                assert float(history[-1]["object_nmse"]) == value

    for overlap, budget in EPOCHS.items():
        rows = [by_id[key] for key in index["studies"]["overlap"]["case_ids"]
                if by_id[key]["overlap_ratio"] == overlap]
        assert len(rows) == 43
        assert all(row["eta"] == .1 and row["batch_size"] == 27 and row["epochs"] == budget for row in rows)
        check_grid(rows, set(index["selections"]["overlap"]))
    baseline = {(by_id[key]["overlap_ratio"], by_id[key]["method"]): by_id[key]
                for key in index["selections"]["overlap"]}
    for key in index["studies"]["noise"]["case_ids"]:
        row = by_id[key]
        assert row["eta"] in (.1, .2, .4)
        assert row["parameters"] == baseline[row["overlap_ratio"], row["method"]]["parameters"]
        assert row["epochs"] == EPOCHS[row["overlap_ratio"]] and row["batch_size"] == 27
    for batch in (27, 108, 432):
        rows = [by_id[key] for key in index["studies"]["batch"]["case_ids"] if by_id[key]["batch_size"] == batch]
        assert len(rows) == 43
        assert all(row["overlap_ratio"] == .875 and row["eta"] == .1 and row["epochs"] == 800 for row in rows)
        check_grid(rows, set(index["selections"]["batch"]))

    for key in index["studies"]["real"]["case_ids"]:
        row = by_id[key]
        assert row["epochs"] == 100 and row["status"] == "complete"
        assert row["batch_size"] == {"test_pattern": 244, "integrated_circuit": 29}[row["dataset"]]

    truth_info = index["synthetic_truth"]
    with np.load(local_path(truth_info["path"]), allow_pickle=False) as archive:
        truth, mask = archive["truth"], archive["roi_mask"]
        for name, expected in truth_info["arrays_sha256"].items():
            assert hashlib.sha256(np.ascontiguousarray(archive[name]).tobytes()).hexdigest() == expected
    reconstructed = [row for row in cases if "reconstruction" in row["files"]]
    assert len(reconstructed) == 12
    for row in reconstructed:
        with np.load(local_path(row["files"]["reconstruction"]), allow_pickle=False) as archive:
            assert all(np.isfinite(archive[key]).all() for key in archive.files
                       if np.issubdtype(archive[key].dtype, np.number))
            if row["study"] != "real":
                aligned = archive["object_aligned"]
                score = float(np.sum(np.abs(aligned[mask] - truth[mask]) ** 2)
                              / np.sum(np.abs(truth[mask]) ** 2))
                assert math.isclose(score, row["object_nmse"], rel_tol=1e-6, abs_tol=1e-12)

    print(json.dumps({"verified_files": len(manifest["files"]), "unique_runs": len(cases),
                      "selected_reconstruction_archives": len(reconstructed),
                      "parameter_grids_and_endpoint_selections": "passed",
                      "saved_object_scores": "passed"}, indent=2))


if __name__ == "__main__":
    main()
