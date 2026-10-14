"""Archive identity stays stable across notebook saves, but not science changes."""

from dataclasses import dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from utils import real_result_archive as archive


@dataclass
class Config:
    data_path: Path
    num_epochs: int = 3
    batch_size: int = 2
    seed: int = 42


def make_session(root):
    notebook = root / "run.ipynb"
    notebook.write_text(json.dumps({
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [{
            "cell_type": "code",
            "id": "fixture",
            "metadata": {},
            "source": ["# Display configuration\n"],
            "execution_count": None,
            "outputs": [],
        }],
    }), encoding="utf-8")
    source = root / "algorithm.py"
    source.write_text("alpha = 1\n", encoding="utf-8")
    data = root / "data.hdf5"
    data.write_bytes(b"small fixture source")
    config = Config(data)
    methods = archive.CANONICAL_REAL_METHODS
    stems = dict(zip(methods, ("rpie", "lsqml", "gm_rpie", "gm_magpie")))
    return archive.RealResultArchiveSession(
        output_root=root / "exports",
        project_root=root,
        experiment_name="fixture",
        notebook_path=notebook,
        run_started_utc=datetime(2026, 10, 14, tzinfo=timezone.utc),
        run_id="fixture_run",
        dataset=SimpleNamespace(
            data=np.ones((2, 2, 2), dtype=np.float32),
            positions_px=np.array([[0., 0.], [1., 2.]], dtype=np.float32),
            probe_init=np.ones((2, 2), dtype=np.complex64),
            dx_m=1e-9,
            wavelength_m=1e-10,
            selected_pattern_indices=np.array([0, 1], dtype=np.int64),
            source_pattern_count=2,
        ),
        dataset_config=config,
        source_data_path=data,
        method_order=methods,
        method_file_stems=stems,
        method_parameters={
            method: {"runner": stems[method], "config": config}
            for method in methods
        },
        metric_report_every=1,
        include_initial_metrics=False,
        sampled_metric_keys=(),
        final_metric_keys=(),
        device={"backend": "fixture"},
        source_paths={"algorithm": source},
        metric_definitions={"online_preupdate_residual": "fixture residual"},
        timing_notes="fixture timing",
        alignment_scale_amplitude=True,
        alignment_fit_mask=np.ones((3, 3), dtype=bool),
        snapshot_every=2,
    )


def method_values(session, method):
    index = session.method_order.index(method)
    obj = np.ones((1, 3, 3), dtype=np.complex64) * (1 + index * 0.1j)
    probe = np.ones((1, 1, 2, 2), dtype=np.complex64) * (2 + index * 0.1j)
    return {
        "method": method,
        "result": SimpleNamespace(
            object=obj,
            probe=probe,
            positions_px=session.dataset.positions_px.copy(),
            residual_history=np.array([3., 2., 1.], dtype=np.float32),
            metric_epochs=np.array([], dtype=np.int64),
            metric_history={},
            final_metrics={},
        ),
        "runtime_seconds": float(index + 1),
        "aligned_object": obj.squeeze().copy(),
        "aligned_probe": probe.squeeze().copy(),
        "alignment": SimpleNamespace(
            slope_y_rad_per_px=0.,
            slope_x_rad_per_px=0.,
            object_scale=1 + 0j,
        ),
    }


def save_method(session, method):
    values = method_values(session, method)
    result = values["result"]
    for epoch in session.expected_snapshot_epochs:
        archive.save_real_reconstruction_snapshot(
            session.output_root,
            run_id=session.run_id,
            method=method,
            method_file_stem=session.method_file_stems[method],
            epoch=epoch,
            object_snapshot=result.object,
            probe_snapshot=result.probe,
            positions_snapshot=result.positions_px,
        )
    return session.save_method(**values)


class ArchiveIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.session = make_session(Path(self.temporary_directory.name))

    def test_notebook_saves_preserve_identity_and_original_source(self):
        fingerprint = self.session.session_fingerprint
        original_notebook = self.session.notebook_path.read_bytes()
        for index, method in enumerate(self.session.method_order):
            checkpoint = save_method(self.session, method)
            archive.validate_real_method_checkpoint(checkpoint)
            notebook = json.loads(self.session.notebook_path.read_text())
            notebook["cells"][0]["source"] = [f"# Display edit {index}\n"]
            notebook["cells"][0]["execution_count"] = index + 1
            notebook["cells"][0]["outputs"] = [{
                "output_type": "stream", "name": "stdout", "text": "output\n",
            }]
            self.session.notebook_path.write_text(
                json.dumps(notebook), encoding="utf-8",
            )
            self.assertEqual(self.session.session_fingerprint, fingerprint)

        result_directory = archive.save_real_notebook_results(self.session)
        manifest = archive.validate_real_results_archive(result_directory)
        self.assertEqual(tuple(manifest["method_order"]), self.session.method_order)
        self.assertEqual(manifest["session_fingerprint"], fingerprint)
        identity = json.loads(
            (result_directory / "session" / "identity.json").read_text()
        )
        saved_notebook = (
            result_directory / "session"
            / identity["source_files"]["notebook"]["archive"]
        )
        self.assertEqual(saved_notebook.read_bytes(), original_notebook)
        self.assertEqual(
            manifest["source_files"]["notebook"]["sha256"],
            hashlib.sha256(original_notebook).hexdigest(),
        )
        for method, stem in self.session.method_file_stems.items():
            expected = method_values(self.session, method)["result"]
            with np.load(
                result_directory / "reconstructions" / f"{stem}.npz",
                allow_pickle=False,
            ) as saved:
                for name in ("object", "probe", "positions_px", "residual_history"):
                    np.testing.assert_array_equal(saved[name], getattr(expected, name))
        with self.assertRaises(FileExistsError):
            archive.save_real_notebook_results(self.session)

    def test_changed_scientific_source_is_rejected(self):
        self.session.source_paths["algorithm"].write_text(
            "alpha = 100\n", encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Non-notebook source changed"):
            _ = self.session.session_fingerprint

    def test_changed_settings_are_rejected(self):
        self.session.dataset_config.num_epochs = 100
        with self.assertRaisesRegex(ValueError, "configuration or initial arrays changed"):
            _ = self.session.session_fingerprint

    def test_same_run_id_cannot_reuse_a_different_saved_identity(self):
        checkpoint = save_method(self.session, "rPIE")
        original_manifest = (checkpoint / "method_manifest.json").read_bytes()
        self.session.notebook_path.write_text(
            '{"nbformat": 4, "cells": []}', encoding="utf-8",
        )
        other = archive.RealResultArchiveSession(**{
            field.name: getattr(self.session, field.name)
            for field in fields(self.session) if field.init
        })
        with self.assertRaisesRegex(ValueError, "different identity"):
            other.save_method(**method_values(other, "rPIE"))
        self.assertEqual(
            (checkpoint / "method_manifest.json").read_bytes(), original_manifest,
        )


if __name__ == "__main__":
    unittest.main()
