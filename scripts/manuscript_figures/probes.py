"""Render separate probe figures from the unchanged publication reconstructions.

Synthetic probes use the saved truth-aligned arrays from the selected 87.5%
baseline. Real probes receive exactly the paired object/probe gauge transform
used by the existing real object plots. No independent probe fitting or
per-panel amplitude normalization is performed. Only phase values at very weak
probe pixels are hidden for display.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "magpie_probe_figures_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator, NullLocator
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
OUTPUT = REPOSITORY / "reproduced_manuscript_figures"
SUPPORTING = OUTPUT / "evidence"
sys.path.insert(0, str(REPOSITORY / "src"))

from magpie_publication.api import METHODS, file_path, load_fields, load_index, select_cases
from magpie_publication._phase_alignment import align_blind_object_probe

LABELS = dict(zip(METHODS, ("rPIE", "LSQML", "GM-rPIE", "GM-MAGPIE")))
COLORS = dict(zip(METHODS, ("#4477AA", "#EE7733", "#228833", "#AA3377")))
PHASE_MASK_FRACTION = .05
PHASE_MASK_COLOR = "#202020"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def array_record(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=array_sha(value),
                magnitude_max=float(np.abs(value).max()),
                squared_norm=float(np.sum(np.abs(value).astype(np.float64) ** 2, dtype=np.float64)))


def selected(study, **filters):
    rows = select_cases(study, selected=True, root=REPOSITORY, **filters)
    if len(rows) != 4 or {row["method"] for row in rows} != set(METHODS):
        raise ValueError(f"Expected four selected methods: {study}, {filters}")
    return {row["method"]: row for row in rows}


def show(axis, values, norm, cmap):
    handle = axis.imshow(values, norm=norm, cmap=cmap, interpolation="none", resample=False)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return handle


def phase_display(field):
    angle = np.angle(field.astype(np.complex128))
    return np.ma.masked_where(
        np.abs(field) < PHASE_MASK_FRACTION * np.abs(field).max(), angle)


def phase_colormap():
    """Use the old manuscript's cyclic phase map and dark weak-pixel mask."""
    cmap = plt.get_cmap("twilight").copy()
    cmap.set_bad(PHASE_MASK_COLOR)
    return cmap


def method_frame(axis, header):
    color = next((COLORS[m] for m in METHODS if LABELS[m] == header), "#999999")
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(.8)


def plot_probes(fields, headers, stem, *, magnitude_max, dx_m=None, scale_bar_um=None,
                error_reference=None):
    """Match the corresponding object figure and share a colorbar per row."""
    if dx_m is not None:
        return plot_real_probes(fields, headers, stem, magnitude_max=magnitude_max,
                                dx_m=dx_m, scale_bar_um=scale_bar_um)
    columns = len(fields)
    width = 6.5
    # These horizontal positions are identical to the synthetic object figure.
    left, right, gap = .112 * width, .89 * width, .065
    panel = (right - left - gap * (columns - 1)) / columns
    bottom, row_gap, title_space = .16, .075, .28
    rows = 3 if error_reference is not None else 2
    height = 2 * bottom + rows * panel + (rows - 1) * row_gap + title_space
    error_limits = [1e-4, 1e-1]
    error_maxima = [float(np.abs(field-error_reference).max()) for field in fields] if rows == 3 else None
    if rows == 3 and max(error_maxima) > error_limits[1]:
        raise ValueError("The shared probe error upper limit would clip a reconstruction")
    fig = plt.figure(figsize=(width, height))
    phase_map = phase_colormap()
    for row in range(rows):
        y = bottom + (rows - 1 - row) * (panel + row_gap)
        for col, (field, header) in enumerate(zip(fields, headers)):
            axis = fig.add_axes([(left + col * (panel + gap)) / width,
                                 y / height, panel / width, panel / height])
            if row == 0:
                handle = show(axis, np.abs(field), Normalize(0, magnitude_max), "gray")
                axis.set_title(header, fontsize=8.5, pad=4,
                               color=next((COLORS[m] for m in METHODS if LABELS[m] == header), "black"))
            elif row == 1:
                handle = show(axis, phase_display(field), Normalize(-np.pi, np.pi), phase_map)
            else:
                error = np.abs(field - error_reference)
                handle = show(axis, np.maximum(error, error_limits[0]), LogNorm(*error_limits), "gray_r")
            method_frame(axis, header)
            if col == 0:
                axis.set_ylabel(("Magnitude", "Phase", "Abs. complex error")[row], fontsize=8.3, labelpad=7)
        ticks = ([0, .5, 1], [-np.pi, 0, np.pi], [1e-4, 1e-3, 1e-2, 1e-1])[row]
        bar = fig.colorbar(handle, cax=fig.add_axes([.912, y / height, .014, panel / height]), ticks=ticks)
        bar.outline.set_linewidth(.45)
        bar.ax.tick_params(labelsize=7.3, length=2, width=.45, pad=2)
        bar.ax.yaxis.set_minor_locator(NullLocator())
        if row == 1:
            bar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
            bar.set_label("rad", fontsize=7.3, labelpad=2)
        elif row == 2:
            bar.set_ticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"])
    paths = []
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"{stem}.{suffix}"
        fig.savefig(path, dpi=260, facecolor="white")
        paths.append(dict(path=str(path), sha256=sha(path)))
    plt.close(fig)
    return dict(stem=stem, outputs=paths, magnitude_limits=[0, magnitude_max],
                phase_limits=[-np.pi, np.pi], phase_mask_relative_magnitude=PHASE_MASK_FRACTION,
                phase_colormap="twilight", masked_phase_color=PHASE_MASK_COLOR,
                complete_probe_arrays_shown=True, per_panel_amplitude_normalization=False,
                colorbars=True, colorbar_count=rows, colorbar_shared_by="row",
                error_limits=error_limits if rows == 3 else None,
                error_scale="logarithmic" if rows == 3 else None,
                absolute_error_maxima_by_column=error_maxima,
                absolute_error_definition="abs(probe_aligned - probe_truth)" if rows == 3 else None,
                error_upper_clipping=False if rows == 3 else None,
                style="Old manuscript probe display: grayscale magnitude, cyclic twilight phase, dark weak-pixel mask, colored method frames; shared row colorbars and logarithmic absolute complex errors",
                dimensions_inches=[width, height])


def plot_real_probes(fields, headers, stem, *, magnitude_max, dx_m, scale_bar_um):
    """Use the real-object figure's serif grid, labels, bars and colorbars."""
    with plt.rc_context({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7}):
        fig = plt.figure(figsize=(6.9, 3.5), layout="constrained")
        grid = fig.add_gridspec(2, 5, width_ratios=(1, 1, 1, 1, .045), wspace=.035, hspace=.035)
        phase_map = phase_colormap()
        for row, label in enumerate(("Magnitude", "Phase")):
            for col, (field, header) in enumerate(zip(fields, headers)):
                axis = fig.add_subplot(grid[row, col])
                value = np.abs(field) if row == 0 else phase_display(field)
                norm = Normalize(0, magnitude_max) if row == 0 else Normalize(-np.pi, np.pi)
                handle = axis.imshow(value, cmap="gray" if row == 0 else phase_map,
                                     norm=norm, interpolation="none", resample=False)
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(header, pad=4, color=next(COLORS[m] for m in METHODS if LABELS[m] == header))
                if col == 0:
                    axis.set_ylabel(label, labelpad=5)
                axis.text(.035, .96, f"({chr(97 + row * 4 + col)})", transform=axis.transAxes,
                          va="top", ha="left", fontsize=7.5,
                          bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=.8))
                height, width = field.shape
                left, bottom, length = width * .07, height * .91, scale_bar_um * 1e-6 / dx_m
                if length > .25 * width:
                    raise ValueError("Probe scale bar would obscure too much of the field")
                axis.plot([left, left + length], [bottom, bottom], color="white", lw=2.2,
                          solid_capstyle="butt")
                axis.text(left + length / 2, bottom - height * .025,
                          rf"{scale_bar_um:g} $\mu$m", color="white", fontsize=7,
                          ha="center", va="bottom",
                          bbox=dict(facecolor="black", alpha=.5, edgecolor="none", pad=.4))
                method_frame(axis, header)
            bar = fig.colorbar(handle, cax=fig.add_subplot(grid[row, 4]))
            if row == 0:
                bar.locator = MaxNLocator(nbins=4)
                bar.update_ticks()
            else:
                bar.set_ticks([-np.pi, 0, np.pi], labels=[r"$-\pi$", "0", r"$\pi$"])
                bar.set_label("rad", fontsize=7, labelpad=2)
            bar.outline.set_linewidth(.45)
            bar.ax.tick_params(width=.45, length=2, pad=2)
        paths = []
        for suffix in ("pdf", "png"):
            path = OUTPUT / f"{stem}.{suffix}"
            fig.savefig(path, dpi=330, facecolor="white", bbox_inches="tight")
            paths.append(dict(path=str(path), sha256=sha(path)))
        plt.close(fig)
    return dict(stem=stem, outputs=paths, magnitude_limits=[0, magnitude_max],
                phase_limits=[-np.pi, np.pi], phase_mask_relative_magnitude=PHASE_MASK_FRACTION,
                phase_colormap="twilight", masked_phase_color=PHASE_MASK_COLOR,
                complete_probe_arrays_shown=True, per_panel_amplitude_normalization=False,
                colorbars=True, colorbar_count=2, colorbar_shared_by="row",
                scale_bar_um=scale_bar_um, dx_m=dx_m,
                style="Old manuscript probe display: grayscale magnitude, cyclic twilight phase, dark weak-pixel mask and colored method frames; real-object serif grid, panel letters, physical scale bars and shared row colorbars",
                dimensions_inches=[6.9, 3.5])


def synthetic():
    index = load_index(REPOSITORY)
    truth_record = index["synthetic_truth"]
    truth_path = REPOSITORY / truth_record["path"]
    if sha(truth_path) != truth_record["sha256"]:
        raise ValueError("Synthetic truth file checksum differs")
    with np.load(truth_path, allow_pickle=False) as archive:
        truth = archive["probe_truth"]
    if array_sha(truth) != truth_record["arrays_sha256"]["probe_truth"]:
        raise ValueError("Synthetic probe truth array checksum differs")
    rows = selected("overlap", overlap=.875, eta=.1, batch_size=27)
    probes, evidence = [truth], []
    for method in METHODS:
        row = rows[method]
        if row["status"] != "complete" or row["epochs"] != 800:
            raise ValueError("Synthetic probe selection is not the completed 800-epoch baseline")
        fields = load_fields(row, root=REPOSITORY)
        probe = fields["probe_aligned"]
        error = float(np.sum(np.abs(truth - probe) ** 2, dtype=np.float64)
                      / np.sum(np.abs(truth) ** 2, dtype=np.float64))
        if abs(error - row["probe_nmse"]) > 1e-12:
            raise ValueError(f"Probe array does not reproduce saved endpoint error: {method}")
        probes.append(probe)
        path = file_path(row, "reconstruction", root=REPOSITORY)
        evidence.append(dict(case_id=row["id"], reconstruction_path=str(path),
                             reconstruction_sha256=sha(path), probe=array_record(probe),
                             recorded_probe_nmse=row["probe_nmse"], recomputed_probe_nmse=error))
    if max(float(np.abs(p).max()) for p in probes) > 1:
        raise ValueError("Synthetic fixed magnitude scale would clip a field")
    result = plot_probes(probes, ["Ground truth", *[LABELS[m] for m in METHODS]],
                         "numerical_synthetic_probe_reconstructions", magnitude_max=1.,
                         error_reference=truth)
    result.update(study="synthetic", overlap_ratio=.875, eta=.1, batch_size=27,
                  epochs=800, truth_file=str(truth_path), truth_probe=array_record(truth),
                  gauge="Saved probe_aligned, identical truth-registration gauge as the object reconstruction panels",
                  cases=evidence)
    return result


def real(dataset):
    rows = selected("real", dataset=dataset)
    fields = {method: load_fields(rows[method], root=REPOSITORY) for method in METHODS}
    common = fields[METHODS[0]]
    geometry = ("initial_positions_px", "initial_probe", "selected_pattern_indices", "object_origin_coords", "dx_m")
    for candidate in fields.values():
        if candidate["object"].shape != common["object"].shape or any(
                not np.array_equal(common[name], candidate[name]) for name in geometry):
            raise ValueError("Real methods have different geometry or initialization")
    half_y, half_x = (side // 2 for side in common["initial_probe"].shape)
    border = np.ones(common["object"].shape, dtype=bool)
    border[half_y:border.shape[0]-half_y, half_x:border.shape[1]-half_x] = False
    probes, evidence = [], []
    for method in METHODS:
        row, data = rows[method], fields[method]
        if row["status"] != "complete" or row["epochs"] != 100:
            raise ValueError("Real probe selection is not the completed 100-epoch run")
        _, probe, alignment = align_blind_object_probe(
            np.ones_like(data["object"]), data["object"], data["probe"],
            mask=border, scale_amplitude=dataset == "integrated_circuit")
        probes.append(probe)
        path = file_path(row, "reconstruction", root=REPOSITORY)
        record = asdict(alignment)
        record["object_scale"] = [alignment.object_scale.real, alignment.object_scale.imag]
        evidence.append(dict(case_id=row["id"], reconstruction_path=str(path),
                             reconstruction_sha256=sha(path), native_probe=array_record(data["probe"]),
                             displayed_probe=array_record(probe), paired_object_alignment=record))
    magnitude_max = max(float(np.abs(p).max()) for p in probes)
    result = plot_probes(probes, [LABELS[m] for m in METHODS],
                         f"numerical_real_{dataset}_probes", magnitude_max=magnitude_max,
                         dx_m=float(common["dx_m"]),
                         scale_bar_um=.5 if dataset == "test_pattern" else 1.)
    result.update(study="real", dataset=dataset, epochs=100,
                  gauge="Opposite affine phase ramp and inverse scalar from the same border-based object alignment used by current real object plots",
                  object_alignment_mask=array_record(border), cases=evidence)
    return result


def main(argv=None):
    global OUTPUT, SUPPORTING
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    OUTPUT = args.output.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SUPPORTING = OUTPUT / "evidence"
    SUPPORTING.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "image.interpolation": "none", "image.resample": False})
    evidence = dict(source_repository=str(REPOSITORY), figures=[synthetic(), real("test_pattern"), real("integrated_circuit")])
    destination = SUPPORTING / "probe_figures.json"
    destination.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"evidence": str(destination), "outputs": [output for figure in evidence["figures"] for output in figure["outputs"]]}))


if __name__ == "__main__":
    main()
