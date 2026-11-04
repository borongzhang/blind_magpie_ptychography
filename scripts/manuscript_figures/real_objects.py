"""Reproduce the two unchanged experimental object figures from saved arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from magpie_publication.figures import _real


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPOSITORY / "reproduced_manuscript_figures")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8.5, "axes.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "image.interpolation": "none", "image.resample": False})
    paths = []
    for dataset in ("test_pattern", "integrated_circuit"):
        paths.extend(_real(REPOSITORY, dataset, output, ("pdf", "png")))
    evidence = output / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / "real_object_figures.json").write_text(json.dumps({
        "scientific_iterations_performed": False,
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    }, indent=2) + "\n")
    print(json.dumps({"outputs": [str(p) for p in paths]}))


if __name__ == "__main__":
    main()
