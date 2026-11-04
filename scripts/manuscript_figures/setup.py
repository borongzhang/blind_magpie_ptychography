"""Reproduce the synthetic truth/initialization figure from verified public arrays."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import NullLocator
import numpy as np
REPOSITORY = Path(__file__).resolve().parents[2]
OUTPUT = REPOSITORY / "reproduced_manuscript_figures"
sys.path.insert(0, str(REPOSITORY / "src"))
from magpie_publication.api import load_index
from magpie_publication.figures import _truth, ROI
PHASE_MASK_FRACTION = .01

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def phase(field, *, mask_weak=False):
    value = np.angle(field.astype(np.complex128))
    if mask_weak:
        value = np.ma.masked_where(np.abs(field) < PHASE_MASK_FRACTION * np.max(np.abs(field)), value)
    return value


def show(axis, values, norm, cmap="gray"):
    handle = axis.imshow(values, norm=norm, cmap=cmap, interpolation="none", resample=False)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return handle


def add_colorbar(fig, handle, rectangle, ticks, ticklabels=None, *, horizontal=False, label=None):
    bar = fig.colorbar(handle, cax=fig.add_axes(rectangle), ticks=ticks,
                       orientation="horizontal" if horizontal else "vertical")
    bar.outline.set_linewidth(.45)
    bar.ax.tick_params(labelsize=7.3, length=2, width=.45, pad=2)
    bar.ax.yaxis.set_minor_locator(NullLocator())
    if ticklabels:
        bar.set_ticklabels(ticklabels)
    if label:
        bar.set_label(label, fontsize=8, labelpad=3)


def save(fig, stem, outputs):
    for extension in ("pdf", "png"):
        path = OUTPUT / f"{stem}.{extension}"
        fig.savefig(path, dpi=220, facecolor="white")
        outputs.append(path)
    size = list(fig.get_size_inches())
    plt.close(fig)
    return size


def plot_setup(data, outputs):
    fields = (data["truth"][ROI], data["probe_truth"], data["probe_init"])
    width, height, panel = 6.5, 3.8, 1.65
    fig = plt.figure(figsize=(width, height))
    probe_phase_map = plt.get_cmap("twilight").copy()
    probe_phase_map.set_bad("#202020")
    headers = ("True object", "True probe (−700 µm)", "Initial probe (−400 µm)")
    for col, (field, header) in enumerate(zip(fields, headers)):
        x = .6 + col * 1.925
        magnitude_axis = fig.add_axes([x / width, 1.9 / height, panel / width, panel / height])
        phase_axis = fig.add_axes([x / width, .08 / height, panel / width, panel / height])
        magnitude_axis.set_title(header, fontsize=8.5, pad=5)
        show(magnitude_axis, np.abs(field), Normalize(0, 1))
        if col == 0:
            phase_values = phase(field)
            phase_map = "gray"
        else:
            # Use Figure 5's cyclic phase map and weak-pixel mask exactly.
            phase_values = np.ma.masked_where(
                np.abs(field) < .05 * np.abs(field).max(),
                np.angle(field.astype(np.complex128)))
            phase_map = probe_phase_map
        show(phase_axis, phase_values, Normalize(-np.pi, np.pi), phase_map)
        if col == 0:
            magnitude_axis.set_ylabel("Magnitude", fontsize=8.5, labelpad=7)
            phase_axis.set_ylabel("Phase", fontsize=8.5, labelpad=7)
    size = save(fig, "numerical_synthetic_setup", outputs)
    return dict(figure_size_inches=size, object_crop=[46, 978, 46, 978],
                object_magnitude_limits=[0, 1], probe_magnitude_limits=[0, 1],
                phase_limits=[-np.pi, np.pi], object_phase_mask="none",
                object_phase_colormap="gray", probe_phase_colormap="twilight",
                probe_phase_mask="Below 5% of each probe's own maximum magnitude; #202020",
                true_probe_parameter_m=-.0007, initial_probe_parameter_m=-.0004,
                probe_squared_norm=300., probe_actual_magnitude_max=max(float(np.abs(x).max()) for x in fields[1:]))


def main(argv=None):
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    OUTPUT = args.output.expanduser().resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    evidence_dir = OUTPUT / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    index = load_index(REPOSITORY)
    data = _truth(index, REPOSITORY)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.linewidth": .5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "image.interpolation": "none", "image.resample": False})
    outputs = []
    display = plot_setup(data, outputs)
    evidence = dict(source_truth=index["synthetic_truth"], setup_display=display,
                    fields_unchanged=True, scientific_iterations_performed=False,
                    output_files_sha256={p.name: sha(p) for p in outputs})
    (evidence_dir / "setup_figure.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"outputs": [str(p) for p in outputs]}))

if __name__ == "__main__":
    main()
