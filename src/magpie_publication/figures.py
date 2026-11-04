"""Reproduce the seven publication figures from archived CPU-readable files.

Outputs go to a separate directory. Scientific records and curated figures are
never overwritten by default. No reconstruction iterations are performed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter, NullLocator
import numpy as np

from .api import METHODS, _inside, _sha, load_fields, load_index, read_history, repository_root, select_cases
from ._phase_alignment import align_blind_object_probe


LABELS = dict(zip(METHODS, ("rPIE", "LSQML", "GM-rPIE", "GM-MAGPIE")))
COLORS = dict(zip(METHODS, ("#4477AA", "#EE7733", "#228833", "#AA3377")))
ROI = np.s_[46:978, 46:978]


def _truth(index, root):
    record = index["synthetic_truth"]
    path = _inside(root, record["path"])
    if _sha(path) != record["sha256"]:
        raise ValueError("The synthetic truth fixture checksum differs.")
    with np.load(path, allow_pickle=False) as archive:
        fields = {name: archive[name] for name in record["arrays_sha256"]}
    for name, field in fields.items():
        if hashlib.sha256(np.ascontiguousarray(field).tobytes()).hexdigest() != record["arrays_sha256"][name]:
            raise ValueError(f"Synthetic truth array checksum differs: {name}")
    return fields


def _save(fig, stem, output, formats, *, dpi=240, tight=False):
    paths = []
    for suffix in formats:
        path = output / f"{stem}.{suffix}"
        options = dict(bbox_inches="tight", pad_inches=.015) if tight else {}
        fig.savefig(path, dpi=dpi, facecolor="white", **options)
        paths.append(path)
    plt.close(fig)
    return paths


def _style_axis(axis, limits):
    axis.set_yscale("log")
    axis.set_ylim(*limits)
    axis.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    axis.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.grid(axis="y", which="major", color=".86", linewidth=.45)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(which="both", direction="out", width=.55, length=2.6, pad=2)


def _selected(root, study, **filters):
    rows = select_cases(study, selected=True, root=root, **filters)
    if len(rows) != 4 or {row["method"] for row in rows} != set(METHODS):
        raise ValueError(f"Expected four selected methods: {study}, {filters}")
    return {row["method"]: row for row in rows}


def _curve_grid(root, study, output, formats):
    values = (.75, .875, .9375) if study == "overlap" else (27, 108, 432)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5), sharey=True)
    for n, (axis, value) in enumerate(zip(axes, values)):
        filters = dict(overlap=value, eta=.1, batch_size=27) if study == "overlap" else dict(overlap=.875, eta=.1, batch_size=value)
        rows = _selected(root, study, **filters)
        for method in METHODS:
            history = read_history(rows[method], root=root)
            points = history[["epoch", "object_nmse"]].dropna()
            if points.empty or not np.isfinite(points.to_numpy()).all() or (points.object_nmse <= 0).any():
                raise ValueError("A selected trajectory has invalid object scores.")
            axis.plot(points.epoch, points.object_nmse, color=COLORS[method], label=LABELS[method],
                      linewidth=1.2, solid_capstyle="round")
        epochs = rows[METHODS[0]]["epochs"]
        _style_axis(axis, (10 ** -4.5, 1))
        axis.set_xlim(0, epochs)
        axis.set_xticks({3000: [0, 1000, 2000, 3000], 800: [0, 200, 400, 600, 800],
                        200: [0, 50, 100, 150, 200]}[epochs])
        axis.set_xlabel("Epoch")
        title = f"{value * 100:g}% overlap" if study == "overlap" else f"Batch {value}"
        axis.set_title(f"({chr(97+n)}) {title}")
    axes[0].set_ylabel("Object error")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.53, .005),
               ncol=4, frameon=False, handlelength=2.2, columnspacing=1.4)
    fig.subplots_adjust(left=.105, right=.975, bottom=.28, top=.88, wspace=.16)
    return _save(fig, f"numerical_{study}_convergence", output, formats)


def _noise(root, output, formats):
    previous_style = plt.rcParams.copy()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 7.4, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 8, "axes.linewidth": .55})
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.45), sharey=True)
    for n, (axis, overlap) in enumerate(zip(axes, (.75, .875, .9375))):
        for method in METHODS:
            rows = sorted(select_cases("noise", overlap=overlap, method=method, selected=True, root=root), key=lambda row: row["eta"])
            if [row["eta"] for row in rows] != [.1, .2, .4]:
                raise ValueError("Noise figure requires baseline and both fixed-pair follow-ups.")
            axis.plot([row["eta"] for row in rows], [row["object_nmse"] for row in rows],
                      color=COLORS[method], label=LABELS[method], marker="o", markersize=3.5,
                      markeredgewidth=0, linewidth=1.2)
        _style_axis(axis, (10 ** -4.5, .1))
        axis.set_xticks([.1, .2, .4])
        axis.set_xlim(.085, .415)
        axis.set_xlabel(r"Noise parameter $\eta$")
        axis.set_title(f"({chr(97+n)}) {overlap * 100:g}% overlap", fontsize=8.3)
    axes[0].set_ylabel("Final object NMSE")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.53, .005),
               ncol=4, frameon=False, handlelength=2.2, columnspacing=1.4)
    fig.subplots_adjust(left=.105, right=.975, bottom=.28, top=.88, wspace=.16)
    paths = _save(fig, "numerical_noise_summary", output, formats)
    plt.rcParams.update(previous_style)
    return paths


def _show(axis, values, norm, cmap="gray"):
    handle = axis.imshow(values, norm=norm, cmap=cmap, interpolation="none", resample=False)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return handle


def _setup(truth, output, formats):
    width, height, panel = 6.5, 3.8, 1.65
    fig = plt.figure(figsize=(width, height))
    fields = (truth["truth"][ROI], truth["probe_truth"], truth["probe_init"])
    headers = ("True object", "True probe (−700 µm)", "Initial probe (−400 µm)")
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("white")
    for col, (field, header) in enumerate(zip(fields, headers)):
        x = .6 + col * 1.925
        first = fig.add_axes([x / width, 1.9 / height, panel / width, panel / height])
        second = fig.add_axes([x / width, .08 / height, panel / width, panel / height])
        first.set_title(header, fontsize=8.5, pad=5)
        _show(first, np.abs(field), Normalize(0, 1))
        phase = np.angle(field.astype(np.complex128))
        if col:
            phase = np.ma.masked_where(np.abs(field) < .01 * np.abs(field).max(), phase)
        _show(second, phase, Normalize(-np.pi, np.pi), cmap)
        if col == 0:
            first.set_ylabel("Magnitude", fontsize=8.5, labelpad=7)
            second.set_ylabel("Phase", fontsize=8.5, labelpad=7)
    return _save(fig, "numerical_synthetic_setup", output, formats, dpi=220)


def _objects(root, truth, output, formats):
    selected = _selected(root, "overlap", overlap=.875, eta=.1, batch_size=27)
    reference = truth["truth"][ROI]
    fields = [reference]
    for method in METHODS:
        arrays = load_fields(selected[method], root=root)
        if not np.array_equal(arrays["roi_mask"], truth["roi_mask"]):
            raise ValueError("A selected field has a different scoring region.")
        obj = arrays["object_aligned"][ROI]
        native = float(np.sum(np.abs(reference - obj) ** 2, dtype=np.float64)
                       / np.sum(np.abs(reference) ** 2, dtype=np.float64))
        if abs(native - selected[method]["object_nmse"]) > 1e-12:
            raise ValueError("Saved aligned object does not reproduce its endpoint score.")
        fields.append(obj)
    errors = [np.abs(field.astype(np.complex128) - reference.astype(np.complex128)) for field in fields]
    if max(np.abs(field).max() for field in fields) > 2 or max(error.max() for error in errors) > .8:
        raise ValueError("A fixed publication scale would clip a reconstruction.")
    width, height, panel, rowgap = 6.5, 3.8382, .9594, .12
    fig = plt.figure(figsize=(width, height))
    cursor = height - .16
    fig.text(.501, (cursor - .015) / height, "87.5% overlap · η = 0.1 · batch 27 · 800 epochs",
             ha="center", va="top", fontsize=9)
    for row in range(3):
        y = cursor - .40 - row * (panel + rowgap) - panel
        for col, (field, error) in enumerate(zip(fields, errors)):
            axis = fig.add_axes([(.112 * width + col * (panel + .065)) / width,
                                 y / height, panel / width, panel / height])
            if row == 0:
                handle = _show(axis, np.abs(field), Normalize(0, 2))
                axis.set_title("Ground truth" if col == 0 else LABELS[METHODS[col-1]],
                               color="black" if col == 0 else COLORS[METHODS[col-1]], fontsize=8.5, pad=4)
            elif row == 1:
                handle = _show(axis, np.angle(field.astype(np.complex128)), Normalize(-np.pi, np.pi))
            else:
                handle = _show(axis, np.maximum(error, 1e-3), LogNorm(1e-3, .8), "gray_r")
            if col == 0:
                axis.set_ylabel(("Magnitude", "Phase", "Abs. complex error")[row], fontsize=8.3, labelpad=7)
        ticks = ([0, 1, 2], [-np.pi, 0, np.pi], [1e-3, 1e-2, 1e-1, .8])[row]
        labels = (None, [r"$-\pi$", "0", r"$\pi$"], [r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "0.8"])[row]
        bar = fig.colorbar(handle, cax=fig.add_axes([.912, y / height, .014, panel / height]), ticks=ticks)
        bar.outline.set_linewidth(.45)
        bar.ax.tick_params(labelsize=7.3, length=2, width=.45, pad=2)
        bar.ax.yaxis.set_minor_locator(NullLocator())
        if labels:
            bar.set_ticklabels(labels)
    return _save(fig, "numerical_0875_baseline_objects", output, formats, dpi=220)


def _real(root, dataset, output, formats):
    selected = _selected(root, "real", dataset=dataset)
    fields = [load_fields(selected[method], root=root) for method in METHODS]
    shape = fields[0]["object"].shape
    geometry = ("initial_positions_px", "initial_probe", "selected_pattern_indices", "object_origin_coords", "dx_m")
    for candidate in fields[1:]:
        if candidate["object"].shape != shape or any(not np.array_equal(fields[0][name], candidate[name]) for name in geometry):
            raise ValueError("Real methods differ in shared geometry or initialization.")
    centers = fields[0]["initial_positions_px"] + fields[0]["object_origin_coords"].reshape(1, 2)
    y0, x0 = np.ceil(centers.min(axis=0)).astype(int)
    y1, x1 = np.floor(centers.max(axis=0)).astype(int) + 1
    crop = np.s_[y0:y1, x0:x1]
    half_y, half_x = (side // 2 for side in fields[0]["initial_probe"].shape)
    border = np.ones(shape, dtype=bool)
    border[half_y:shape[0]-half_y, half_x:shape[1]-half_x] = False
    magnitudes, phases = [], []
    for candidate in fields:
        obj = candidate["object"]
        aligned, _, _ = align_blind_object_probe(np.ones_like(obj), obj, candidate["probe"],
            mask=border, scale_amplitude=dataset == "integrated_circuit")
        magnitudes.append(np.abs(obj))
        phases.append(np.angle(aligned))
    limits = [np.percentile(np.concatenate([value[crop].ravel() for value in values]), percentiles)
              for values, percentiles in ((magnitudes, (1, 99)), (phases, (2, 98)))]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig = plt.figure(figsize=(6.9, 3.5), layout="constrained")
    grid = fig.add_gridspec(2, 5, width_ratios=(1, 1, 1, 1, .045), wspace=.035, hspace=.035)
    dx = float(fields[0]["dx_m"])
    for row, (values, label) in enumerate(((magnitudes, "Magnitude"), (phases, "Phase"))):
        for col, (value, method) in enumerate(zip(values, METHODS)):
            axis = fig.add_subplot(grid[row, col])
            handle = axis.imshow(value[crop], cmap="gray", vmin=limits[row][0], vmax=limits[row][1], interpolation="none")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(LABELS[method], pad=4)
            if col == 0:
                axis.set_ylabel(label, labelpad=5)
            axis.text(.035, .96, f"({chr(97 + row * 4 + col)})", transform=axis.transAxes,
                      va="top", ha="left", fontsize=7.5,
                      bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=.8))
            height, width = value[crop].shape
            left, bottom, length = width * .07, height * .91, 2e-6 / dx
            axis.plot([left, left+length], [bottom, bottom], color="white", lw=2.2, solid_capstyle="butt")
            axis.text(left+length/2, bottom-height*.025, r"2 $\mu$m", color="white", fontsize=7,
                      ha="center", va="bottom", bbox=dict(facecolor="black", alpha=.5, edgecolor="none", pad=.4))
            for spine in axis.spines.values():
                spine.set_linewidth(.45)
        bar = fig.colorbar(handle, cax=fig.add_subplot(grid[row, 4]))
        bar.locator = MaxNLocator(nbins=4)
        bar.update_ticks()
        if row == 1:
            bar.set_label("rad", fontsize=7, labelpad=2)
        bar.outline.set_linewidth(.45)
        bar.ax.tick_params(width=.45, length=2, pad=2)
    return _save(fig, f"numerical_real_{dataset}", output, formats, dpi=330, tight=True)


def reproduce_figures(*, root=None, output=None, formats=("pdf", "png")):
    """Render all seven final figures from hashes checked against the index."""
    root = repository_root(root)
    output = Path(output).expanduser().resolve() if output else root / "reproduced_figures"
    if output.is_relative_to(root / "results"):
        raise ValueError("Choose a separate output directory; curated publication figures are immutable.")
    if not formats or any(suffix not in ("pdf", "png") for suffix in formats):
        raise ValueError("formats must contain pdf and/or png")
    output.mkdir(parents=True, exist_ok=True)
    index = load_index(root)
    truth = _truth(index, root)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8.5, "axes.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "image.interpolation": "none", "image.resample": False})
    paths = _setup(truth, output, formats)
    paths += _curve_grid(root, "overlap", output, formats)
    paths += _objects(root, truth, output, formats)
    paths += _noise(root, output, formats)
    paths += _curve_grid(root, "batch", output, formats)
    for dataset in ("test_pattern", "integrated_circuit"):
        paths += _real(root, dataset, output, formats)
    report = dict(source_index_sha256=_sha(root / "results/index.json"),
                  scientific_iterations_performed=False, curated_files_changed=False,
                  output_files_sha256={str(path.name): _sha(path) for path in paths})
    (output / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
    return paths
