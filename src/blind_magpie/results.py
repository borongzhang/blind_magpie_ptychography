"""Read and visualize the compact, recorded paper results on CPU."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

METHODS = ("rpie", "lsqml", "gm_rpie", "gm_magpie")
LABELS = dict(zip(METHODS, ("rPIE", "LSQML", "GM-rPIE", "GM-MAGPIE")))
COLORS = dict(zip(METHODS, ("#4477AA", "#EE7733", "#228833", "#AA3377")))


def results_directory(root=None):
    """Locate results from a checkout, a notebook, or an explicit directory."""
    candidates = ([Path(root).expanduser().resolve()] if root is not None else
                  [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]])
    for candidate in candidates:
        for directory in (candidate / "results", candidate):
            if (directory / "endpoints.csv").is_file():
                return directory
    raise FileNotFoundError("Cannot find results/endpoints.csv; run from the checkout.")


def endpoints(root=None):
    """All synthetic candidates, including unsuccessful sweep candidates."""
    return pd.read_csv(results_directory(root) / "endpoints.csv", float_precision="round_trip")


def selected_cases(study, root=None):
    """Cases selected by final object NMSE, including shared baselines."""
    if study not in ("overlap", "noise", "batch"):
        raise ValueError("Choose overlap, noise, or batch.")
    frame = endpoints(root)
    frame = frame.loc[frame[f"selected_{study}"]].copy()
    frame["parameters"] = frame.apply(
        lambda row: (f"damping={row.damping:g}, σ={row.sigma:g}" if row.method == "lsqml"
                     else f"αO={row.object_alpha:g}, αP={row.probe_alpha:g}"), axis=1)
    frame["_method_order"] = frame.method.map({m: i for i, m in enumerate(METHODS)})
    return (frame.sort_values(["overlap", "eta", "batch_size", "_method_order"])
            .drop(columns="_method_order").reset_index(drop=True))


def real_parameters(root=None):
    """Settings of the eight displayed real-data reconstructions."""
    return pd.read_csv(results_directory(root) / "real_parameters.csv",
                       float_precision="round_trip")


def protocol(root=None):
    return json.loads((results_directory(root) / "protocol.json").read_text())


def show_figure(stem, root=None):
    """Return an embedded PNG, so its result is saved inside an executed notebook."""
    from IPython.display import Image
    if Path(stem).name != stem:
        raise ValueError("Supply a figure name without a directory.")
    name = stem if stem.endswith(".png") else f"{stem}.png"
    return Image(filename=str(results_directory(root) / "figures" / name), embed=True)


def _finish(fig, axes, ylabel):
    for axis in axes:
        axis.set_yscale("log")
        axis.grid(axis="y", color=".87", linewidth=.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(ylabel)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(.53, .01))
    fig.subplots_adjust(left=.08, right=.98, bottom=.25, top=.87, wspace=.14)
    return fig


def plot_convergence(study, metric="object", root=None):
    """Raw saved NMSE checkpoints for the object-selected parameter pairs."""
    if study not in ("overlap", "batch") or metric not in ("object", "probe"):
        raise ValueError("Choose overlap/batch and object/probe.")
    selected = selected_cases(study, root)
    history = pd.read_csv(results_directory(root) / "convergence.csv.gz",
                          float_precision="round_trip")
    column = f"{metric}_nmse"
    group = "overlap" if study == "overlap" else "batch_size"
    values = sorted(selected[group].unique())
    fig, axes = plt.subplots(1, len(values), figsize=(12, 3.6), sharey=True, squeeze=False)
    axes = axes[0]
    for axis, value in zip(axes, values):
        rows = selected.loc[selected[group] == value]
        for method in METHODS:
            case = rows.loc[rows.method == method].iloc[0]
            points = history.loc[history.case_id == case.case_id]
            if points.empty or not np.isfinite(points[column]).all() or (points[column] <= 0).any():
                raise ValueError(f"Invalid recorded trajectory: {case.case_id}")
            axis.plot(points.epoch, points[column], color=COLORS[method], label=LABELS[method],
                      linewidth=1.7)
        epochs = int(rows.epochs.iloc[0])
        axis.set_xlim(0, epochs)
        axis.set_xticks({3000: [0, 1000, 2000, 3000], 800: [0, 200, 400, 600, 800],
                         200: [0, 50, 100, 150, 200]}[epochs])
        axis.set_ylim(*( (10**-4.5, 1) if metric == "object" else (10**-5.5, 10)))
        title = (f"{100*value:g}% overlap" if study == "overlap" else
                 f"Batch {value} ({rows.batch_percent.iloc[0]:.2f}%)")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
    return _finish(fig, axes, f"{metric.capitalize()} NMSE")


def plot_noise(metric="object", root=None):
    """Final NMSE at three noise levels, with overlap-selected parameters fixed."""
    if metric not in ("object", "probe"):
        raise ValueError("Choose object or probe.")
    selected = selected_cases("noise", root)
    column = f"{metric}_nmse"
    values = sorted(selected.overlap.unique())
    fig, axes = plt.subplots(1, len(values), figsize=(12, 3.6), sharey=True, squeeze=False)
    axes = axes[0]
    positive = selected[column].to_numpy()
    if not np.isfinite(positive).all() or (positive <= 0).any():
        raise ValueError("The recorded noise errors must be finite and positive.")
    limits = (10**np.floor(np.log10(positive.min())), 10**np.ceil(np.log10(positive.max())))
    for axis, overlap in zip(axes, values):
        for method in METHODS:
            rows = selected.loc[(selected.overlap == overlap) & (selected.method == method)].sort_values("eta")
            axis.plot(rows.eta, rows[column], color=COLORS[method], label=LABELS[method],
                      marker="o", markersize=5, linewidth=1.7)
        axis.set_title(f"{100*overlap:g}% overlap")
        axis.set_xticks([.1, .2, .4])
        axis.set_xlim(.08, .42)
        axis.set_ylim(*limits)
        axis.set_xlabel(r"Noise parameter $\eta$")
    return _finish(fig, axes, f"Final {metric} NMSE")


def plot_sweep_heatmaps(study="overlap", root=None):
    """All searched pairs; boxes mark final-object-error selections."""
    if study not in ("overlap", "batch"):
        raise ValueError("Parameter sweeps are available for overlap or batch.")
    data = endpoints(root)
    if study == "overlap":
        data = data.loc[data.study == "overlap"]
        group = "overlap"
    else:
        data = data.loc[(data.study == "batch") |
                        ((data.study == "overlap") & (data.overlap == .875))]
        group = "batch_size"
    values = sorted(data[group].unique())
    valid = data.loc[data.status == "complete", "object_nmse"]
    norm = LogNorm(float(valid.min()), float(valid.max()))
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad(".85")
    fig, axes = plt.subplots(len(values), 4, figsize=(14, 3.2*len(values)), squeeze=False)
    for row_index, value in enumerate(values):
        for col_index, method in enumerate(METHODS):
            axis = axes[row_index, col_index]
            rows = data.loc[(data[group] == value) & (data.method == method)]
            xname, yname = ("sigma", "damping") if method == "lsqml" else ("object_alpha", "probe_alpha")
            xs, ys = sorted(rows[xname].unique()), sorted(rows[yname].unique())
            matrix = np.full((len(ys), len(xs)), np.nan)
            for _, case in rows.iterrows():
                y, x = ys.index(case[yname]), xs.index(case[xname])
                if case.status == "complete":
                    matrix[y, x] = case.object_nmse
            image = axis.imshow(np.ma.masked_invalid(matrix), origin="lower", norm=norm,
                                cmap=cmap, aspect="auto")
            for y in range(len(ys)):
                for x in range(len(xs)):
                    number = matrix[y, x]
                    label = f"{number:.1e}" if np.isfinite(number) else "failed"
                    color = "white" if np.isfinite(number) and norm(number) > .45 else "black"
                    axis.text(x, y, label, ha="center", va="center", fontsize=8, color=color)
            chosen = rows.loc[rows[f"selected_{study}"]]
            if len(chosen) != 1:
                raise ValueError(f"Expected one selected pair: {study}, {value}, {method}")
            selected = chosen.iloc[0]
            axis.add_patch(Rectangle((xs.index(selected[xname])-.48, ys.index(selected[yname])-.48),
                                     .96, .96, fill=False, edgecolor="#DD3355", linewidth=2))
            axis.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
            axis.set_yticks(range(len(ys)), [f"{y:g}" for y in ys])
            axis.set_xlabel(r"$\sigma$" if method == "lsqml" else r"Object $\alpha$")
            axis.set_ylabel("Damping" if method == "lsqml" else r"Probe $\alpha$")
            condition = f"{100*value:g}% overlap" if study == "overlap" else f"Batch {value}"
            axis.set_title(f"{LABELS[method]} · {condition}", fontsize=10)
    fig.subplots_adjust(left=.07, right=.90, bottom=.06, top=.95, wspace=.30, hspace=.55)
    fig.colorbar(image, cax=fig.add_axes([.93, .12, .015, .76]), label="Final object NMSE")
    return fig
