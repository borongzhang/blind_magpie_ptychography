"""Reports from saved numerical results; importing this module needs no Torch."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


_NAMES = {"rpie": "rPIE", "lsqml": "LSQML", "gm_rpie": "GM-rPIE", "gm_magpie": "GM-MAGPIE"}
_METRICS = (
    ("object_nmse", "Object NMSE (fixed ROI)"),
    ("probe_nmse", "Probe NMSE"),
    ("phase_rmse", "Object phase RMSE (rad)"),
    ("online_amplitude_mse", "Online amplitude MSE"),
)


def _save(fig, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=160, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _results(root, smoke):
    return Path(root) / "results" / ("smoke" if smoke else "")


def _run_path(root, row, smoke):
    value = row.get("relative_run_dir")
    if pd.isna(value) or not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = (Path(root) / path, _results(root, smoke) / path)
    return next((p for p in candidates if p.exists()), candidates[0])


def _label(row):
    if pd.notna(row.get("damping")) and pd.notna(row.get("sigma")):
        return f"d={float(row['damping']):g}, s={float(row['sigma']):g}"
    if pd.notna(row.get("object_alpha")) and pd.notna(row.get("probe_alpha")):
        return f"aO={float(row['object_alpha']):g}, aP={float(row['probe_alpha']):g}"
    return str(row.get("run_id", "run"))


def _completed(summary):
    if summary.empty or "object_nmse" not in summary:
        return summary.iloc[:0].copy()
    finite = np.isfinite(pd.to_numeric(summary["object_nmse"], errors="coerce"))
    status = summary["status"].astype(str).str.lower() if "status" in summary else pd.Series("completed", index=summary.index)
    return summary.loc[finite & status.isin(["completed", "complete", "success", "succeeded"])].copy()


def _histories(root, rows, smoke, directory, stem, title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = plt.get_cmap("tab20")
    plotted = False
    for i, (_, row) in enumerate(rows.iterrows()):
        path = _run_path(root, row, smoke)
        if path is None or not (path / "history.csv").exists():
            continue
        history = pd.read_csv(path / "history.csv")
        if history.empty or "epoch" not in history:
            continue
        for axis, (metric, label) in zip(axes.flat, _METRICS):
            if metric not in history:
                continue
            values = pd.to_numeric(history[metric], errors="coerce")
            valid = np.isfinite(values) & (values >= 0)
            if not valid.any():
                continue
            axis.plot(history.loc[valid, "epoch"], np.maximum(values.loc[valid], 1e-14),
                      lw=1.25, color=colors(i % 20), label=_label(row))
            axis.set_yscale("log")
            axis.set_ylabel(label)
            plotted = True
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    if plotted:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(4, len(handles)),
                       fontsize=8, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(title)
        fig.tight_layout(rect=(0, min(0.04 + len(handles) / 4 * 0.025, 0.18), 1, 0.96))
        _save(fig, directory, f"convergence_{stem}".rstrip("_"))
    else:
        plt.close(fig)


def _heatmap(rows, method, directory, stem, title):
    from .studies import retained_parameter_grid

    keys = ("damping", "sigma") if method == "lsqml" else ("object_alpha", "probe_alpha")
    if not all(key in rows for key in keys):
        return
    finite = rows.dropna(subset=[*keys, "object_nmse"])
    if finite.empty:
        return
    grid = retained_parameter_grid(method)
    table = finite.pivot_table(index=keys[0], columns=keys[1], values="object_nmse", aggfunc="min")
    table = table.reindex(index=sorted({p[keys[0]] for p in grid}),
                          columns=sorted({p[keys[1]] for p in grid}))
    values = table.to_numpy(float)
    positive = values[np.isfinite(values) & (values > 0)]
    if not positive.size:
        return
    lo, hi = positive.min(), positive.max()
    fig, axis = plt.subplots(figsize=(6.5, 5.2))
    norm = LogNorm(vmin=max(lo, 1e-14), vmax=max(hi, lo * 1.00001))
    shown = axis.imshow(np.ma.masked_invalid(np.maximum(values, 1e-14)), norm=norm, cmap="viridis_r", aspect="auto")
    axis.set_xticks(range(len(table.columns)), [f"{x:g}" for x in table.columns])
    axis.set_yticks(range(len(table.index)), [f"{x:g}" for x in table.index])
    axis.set_xlabel("sigma" if method == "lsqml" else "Probe alpha")
    axis.set_ylabel("Damping" if method == "lsqml" else "Object alpha")
    for (iy, ix), value in np.ndenumerate(values):
        label = f"{value:.2e}" if np.isfinite(value) else "missing"
        axis.text(ix, iy, label, ha="center", va="center", fontsize=8,
                  color="white" if np.isfinite(value) and norm(max(value, 1e-14)) < 0.55 else "black")
    axis.set_title(title)
    fig.colorbar(shown, ax=axis, label="Final object NMSE (fixed ROI)")
    _save(fig, directory, f"parameter_grid_{stem}".rstrip("_"))


def _gallery(root, row, smoke, directory, stem, title):
    path = _run_path(root, row, smoke)
    if path is None or not (path / "reconstruction.npz").exists():
        return
    from .data import DataConfig, load_or_make_data

    data = load_or_make_data(root, DataConfig(smoke=smoke))
    with np.load(path / "reconstruction.npz", allow_pickle=False) as saved:
        obj = saved["object_aligned"] if "object_aligned" in saved else saved["object"]
        probe = saved["probe_aligned"] if "probe_aligned" in saved else saved["probe"]
    # A unit-modulus scalar aligns only phase, preserving both initial and true norms.
    initial_probe = np.asarray(data.probe_init)
    phase_scalar = np.exp(1j * np.angle(np.vdot(initial_probe, data.probe_truth)))
    fields = [
        [np.asarray(data.truth), np.asarray(data.object_init), obj],
        [np.asarray(data.probe_truth), initial_probe * phase_scalar, probe],
    ]
    fig, axes = plt.subplots(4, 3, figsize=(11.5, 13))
    for group, matrices in enumerate(fields):
        vmax = max(float(np.abs(x).max()) for x in matrices)
        for col, matrix in enumerate(matrices):
            mag = axes[group * 2, col].imshow(np.abs(matrix), cmap="gray", vmin=0, vmax=vmax)
            phase = axes[group * 2 + 1, col].imshow(np.angle(matrix), cmap="twilight", vmin=-np.pi, vmax=np.pi)
            if group == 0:
                yy, xx = np.where(data.roi_mask)
                axes[0, col].add_patch(Rectangle((xx.min() - 0.5, yy.min() - 0.5),
                                               xx.max() - xx.min() + 1, yy.max() - yy.min() + 1,
                                               fill=False, edgecolor="#ef4ea7", linewidth=1))
            axes[group * 2, col].set_title(["Ground truth", "Initialization", "Reconstruction"][col])
            for axis in (axes[group * 2, col], axes[group * 2 + 1, col]):
                axis.set_xticks([])
                axis.set_yticks([])
        fig.colorbar(mag, ax=axes[group * 2, :].tolist(), fraction=0.017, pad=0.025)
        fig.colorbar(phase, ax=axes[group * 2 + 1, :].tolist(), fraction=0.017, pad=0.025, ticks=[-np.pi, 0, np.pi])
    for axis, label in zip(axes[:, 0], ["Object magnitude", "Object phase", "Probe magnitude", "Probe phase"]):
        axis.set_ylabel(label)
    fig.suptitle(f"{title}\n{_label(row)}; object NMSE={float(row['object_nmse']):.4g}")
    fig.subplots_adjust(left=0.07, right=0.87, top=0.92, hspace=0.22, wspace=0.07)
    _save(fig, directory, f"fields_{stem}".rstrip("_"))



def save_report(method, root, summary, smoke=False):
    """Export saved 75% sweep results without importing Torch or reconstructing.

    Raw per-run results remain authoritative. The winner minimizes the final
    fixed-ROI object NMSE among completed runs. Missing and failed settings stay
    visible in tables; neither is a candidate for parameter selection.
    """
    from .protocol import METHODS
    from .studies import retained_parameter_grid

    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    directory = _results(root, smoke) / "studies" / "sweep" / method
    directory.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary).copy()
    allowed = retained_parameter_grid(method)
    keys = ("damping", "sigma") if method == "lsqml" else ("object_alpha", "probe_alpha")
    if not summary.empty:
        if not all(key in summary for key in keys):
            raise ValueError("Saved summary has no parameter columns")
        keep = summary.apply(lambda row: any(all(row[key] == pair[key] for key in keys) for pair in allowed), axis=1)
        summary = summary.loc[keep].copy()
    expected_count = len(allowed)
    terminal_statuses = {"complete", "completed", "failed"}
    terminal_count = int(summary["status"].astype(str).str.lower().isin(terminal_statuses).sum()) if "status" in summary else 0
    selection_complete = len(summary) == expected_count and terminal_count == expected_count
    summary["selection_complete"] = selection_complete
    summary.to_csv(directory / "summary.csv", index=False)
    runtime_cols = [c for c in ("method", "overlap_ratio", "eta", "batch_size", "run_id", "status", "epochs", "epochs_completed", "object_alpha", "probe_alpha", "damping", "sigma", "reconstruction_seconds", "transfer_seconds", "scoring_seconds", "frozen_evaluation_seconds", "total_seconds", "relative_run_dir") if c in summary]
    summary[runtime_cols].to_csv(directory / "runtime.csv", index=False)
    traces = []
    for _, row in summary.iterrows():
        path = _run_path(root, row, smoke)
        if path is None or not (path / "history.csv").exists():
            continue
        history = pd.read_csv(path / "history.csv")
        if "reconstruction_seconds" in history:
            # History timings are per-epoch increments, not cumulative values.
            history["cumulative_reconstruction_seconds"] = history["reconstruction_seconds"].cumsum()
        if "object_nmse" in history:
            history = history.loc[history.object_nmse.notna()].copy()
        for key in ("method", "run_id", "object_alpha", "probe_alpha", "damping", "sigma"):
            if key in row:
                history[key] = row[key]
        traces.append(history)
    (pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()).to_csv(directory / "convergence.csv", index=False)
    completed = _completed(summary)
    sort_keys = ["object_nmse", "run_id"] if "run_id" in completed else "object_nmse"
    completed = completed.sort_values(sort_keys, kind="stable") if len(completed) else completed
    best = completed.head(1)
    best.to_csv(directory / "best_final.csv", index=False)
    title = f"{_NAMES.get(method, method)} — 75% overlap, eta=0.1, batch=27"
    if not selection_complete:
        title += f"\n(provisional: {terminal_count}/{expected_count} terminal)"
    if len(completed):
        _histories(root, completed, smoke, directory, "", title)
        _heatmap(completed, method, directory, "", title)
        _gallery(root, best.iloc[0], smoke, directory, "", title)
    note = {
        "study": "sweep", "method": method, "smoke": bool(smoke),
        "planned_rows": len(summary), "completed_rows": len(completed),
        "expected_pairs": expected_count, "terminal_rows": terminal_count,
        "selection_complete": selection_complete,
        "selection": "lowest final object NMSE on the fixed ROI",
        "initial_probe_preview": "unit-modulus phase alignment preserves squared norm; optimal-complex-scalar shape NMSE is reported separately",
        "curves": "Online amplitude MSE is recorded during updates; final frozen residuals are separate summary fields.",
        "timing": "Raw history reconstruction_seconds is per epoch; convergence.csv additionally contains its cumulative sum.",
        "alignment": "Object scale and phase ramp are fitted on the fixed ROI; saved aligned reconstruction fields are displayed.",
    }
    (directory / "report.json").write_text(json.dumps(note, indent=2) + "\n")
    return directory


plot_sweep = save_report


def plot_setup(root, smoke=False):
    """Preview the fixed 75% dataset without running any reconstruction."""
    from .data import DataConfig, load_or_make_data

    root = Path(root)
    directory = _results(root, smoke) / "studies" / "setup"
    directory.mkdir(parents=True, exist_ok=True)
    data = load_or_make_data(root, DataConfig(smoke=smoke))
    truth = np.asarray(data.probe_truth)
    initial_raw = np.asarray(data.probe_init)
    inner = np.vdot(initial_raw, truth)
    phase_scalar = np.exp(1j * np.angle(inner))
    initial_display = initial_raw * phase_scalar
    optimal_scalar = inner / np.vdot(initial_raw, initial_raw).real
    denominator = float(np.sum(np.abs(truth) ** 2))
    shape_nmse = float(np.sum(np.abs(truth - optimal_scalar * initial_raw) ** 2) / denominator)
    phase_only_nmse = float(np.sum(np.abs(truth - initial_display) ** 2) / denominator)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8))
    vmax = max(float(np.abs(truth).max()), float(np.abs(initial_display).max()))
    for col, (field, title) in enumerate(((truth, "Ground-truth probe"), (initial_display, "Initial probe (phase aligned)"))):
        magnitude = axes[0, col].imshow(np.abs(field), cmap="gray", vmin=0, vmax=vmax)
        phase = axes[1, col].imshow(np.angle(field), cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[0, col].set_title(title)
        for axis in axes[:, col]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_ylabel("Magnitude")
    axes[1, 0].set_ylabel("Phase (rad)")
    fig.colorbar(magnitude, ax=axes[0, :].tolist(), fraction=0.025, pad=0.03)
    fig.colorbar(phase, ax=axes[1, :].tolist(), fraction=0.025, pad=0.03, ticks=[-np.pi, 0, np.pi])
    fig.suptitle(f"Both squared norms: 300; displayed phase-only NMSE: {phase_only_nmse:.5f}\nOptimal-complex-scalar shape NMSE: {shape_nmse:.5f}")
    fig.subplots_adjust(top=0.88, right=0.86, wspace=0.12, hspace=0.1)
    _save(fig, directory, "probes")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    yy, xx = np.where(data.roi_mask)
    for axis, values, title, cmap, bounds in (
        (axes[0], np.abs(data.truth), "Ground-truth object magnitude", "gray", (0, 1)),
        (axes[1], np.angle(data.truth), "Ground-truth object phase", "twilight", (-np.pi, np.pi)),
    ):
        shown = axis.imshow(values, cmap=cmap, vmin=bounds[0], vmax=bounds[1])
        axis.add_patch(Rectangle((xx.min() - 0.5, yy.min() - 0.5), xx.max() - xx.min() + 1,
                                 yy.max() - yy.min() + 1, fill=False, edgecolor="#ef4ea7", linewidth=1.2))
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(shown, ax=axis, fraction=0.04, pad=0.025)
    fig.suptitle("Fixed evaluation ROI shown in pink")
    fig.tight_layout()
    _save(fig, directory, "object_and_roi")

    sensitivity = np.zeros(data.truth.shape, dtype=float)
    power = np.abs(data.probe_truth) ** 2
    height, width = power.shape
    for y, x in np.asarray(data.top_left_px, dtype=int):
        sensitivity[y:y + height, x:x + width] += power
    n = len(sensitivity)
    central = sensitivity[n // 4:3 * n // 4, n // 4:3 * n // 4]
    reference = float(central.mean())
    relative = sensitivity / reference if reference else sensitivity
    fig, axis = plt.subplots(figsize=(6.2, 5.5))
    shown = axis.imshow(relative, cmap="magma", norm=LogNorm(vmin=max(float(relative.min()), 1e-5), vmax=max(float(relative.max()), 1)))
    axis.add_patch(Rectangle((xx.min() - 0.5, yy.min() - 0.5), xx.max() - xx.min() + 1,
                             yy.max() - yy.min() + 1, fill=False, edgecolor="#35ebd7", linewidth=1))
    axis.set_title(f"75% nominal overlap; {len(data.positions_px):,} patterns")
    axis.set_xticks([])
    axis.set_yticks([])
    fig.colorbar(shown, ax=axis, fraction=0.04, pad=0.025, label="Relative sensitivity")
    fig.suptitle("Sum of squared shifted true-probe magnitudes")
    fig.tight_layout()
    _save(fig, directory, "scan_sensitivity")
    table = pd.DataFrame([{
        "overlap_ratio": 0.75, "eta": 0.1, "batch_size": 27, "epochs": 2 if smoke else 3000,
        "patterns": len(data.positions_px), "object_size": n, "probe_size": height,
        "roi_pixels": int(data.roi_mask.sum()), "roi_side": int(xx.max() - xx.min() + 1),
        "truth_probe_squared_norm": float(np.sum(np.abs(truth) ** 2)),
        "initial_probe_squared_norm": float(np.sum(np.abs(initial_raw) ** 2)),
        "initial_probe_optimal_scalar_shape_nmse": shape_nmse,
        "initial_probe_phase_only_nmse": phase_only_nmse,
        "minimum_relative_sensitivity_in_roi": float(relative[data.roi_mask].min()),
        "next_centered_even_square_side": int(xx.max() - xx.min() + 3),
        "next_centered_even_square_min_relative_sensitivity": data.metadata.get("next_centered_even_square_min_relative_sensitivity"),
        "data_fingerprint": data.fingerprint,
    }])
    table.to_csv(directory / "setup_summary.csv", index=False)
    return directory, table
