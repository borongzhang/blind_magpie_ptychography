"""Reproduce all ten current manuscript figures on CPU from saved results.

Run from any working directory. Outputs default to reproduced_manuscript_figures/
beside this repository's README; use --output to choose a separate directory.
The archived results and figures are never overwritten. No reconstruction runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RENDERERS = Path(__file__).resolve().parent / "manuscript_figures"
METHODS = ("rpie", "lsqml", "gm_rpie", "gm_magpie")
FIGURES = (
    "numerical_synthetic_setup", "numerical_overlap_convergence",
    "numerical_0875_baseline_objects", "numerical_synthetic_probe_reconstructions",
    "numerical_noise_summary", "numerical_batch_convergence",
    "numerical_real_test_pattern", "numerical_real_test_pattern_probes",
    "numerical_real_integrated_circuit", "numerical_real_integrated_circuit_probes",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reproduced_manuscript_figures")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.is_relative_to(ROOT / "results"):
        parser.error("Choose a separate output directory to preserve archived results.")
    output.mkdir(parents=True, exist_ok=True)
    source_index = ROOT / "results/index.json"
    environment = dict(os.environ)
    environment.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "magpie_manuscript_mpl"))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for script in ("synthetic.py", "setup.py", "probes.py", "real_objects.py"):
        subprocess.run([sys.executable, str(RENDERERS / script),
                        "--output", str(output)], check=True, env=environment)
    paths = [output / f"{stem}.{suffix}" for stem in FIGURES for suffix in ("pdf", "png")]
    if not all(path.is_file() for path in paths):
        raise RuntimeError("The reproduction did not produce all ten PDF/PNG figures.")
    report = dict(figure_count=len(FIGURES), scientific_iterations_performed=False,
                  archived_results_changed=False,
                  source_index_sha256=sha(source_index),
                  output_files_sha256={p.name: sha(p) for p in paths})
    (output / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Reproduced {len(FIGURES)} figures in {output}")


if __name__ == "__main__":
    main()
