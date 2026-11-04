"""Reproduce four synthetic figures from the archived parameter-sweep results.

Overlap labels use the original computational-window width. Batch labels give
both the number of scan positions and its percentage of all scan positions.
The epoch budgets are 3,000, 800, and 200 at the three overlaps, including noise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[2]
OUTPUT = REPOSITORY / "reproduced_manuscript_figures"
SUPPORTING = OUTPUT / "evidence"
sys.path.insert(0, str(REPOSITORY / "src"))
from magpie_publication.api import (METHODS, file_path,
    load_fields, load_index, read_history, select_cases)
from magpie_publication.figures import COLORS, LABELS, ROI, _show, _style_axis, _truth

OVERLAPS = (.75, .875, .9375)
STEPS = {.75: 32, .875: 16, .9375: 8}
POSITIONS = {.75: 841, .875: 3249, .9375: 12769}
BATCHES = (27, 108, 432)
SOURCE_HASHES = {}
MANIFEST = json.loads((REPOSITORY / "results/manifest.json").read_text())["files"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_source(path):
    path = Path(path).resolve()
    SOURCE_HASHES[str(path)] = sha(path)
    return path


def checked_selected(study, **filters):
    rows = select_cases(study, selected=True, root=REPOSITORY, **filters)
    if len(rows) != 4 or {row["method"] for row in rows} != set(METHODS):
        raise ValueError(f"Expected the four selected methods for {study}: {filters}")
    for row in rows:
        for kind in ("config", "result", "history"):
            path = file_path(row, kind, root=REPOSITORY)
            # Curated config/result JSON adds publication identifiers. Their
            # byte hashes belong to the publication manifest, whereas the
            # source_identity hashes describe the original unmodified files.
            expected = MANIFEST[str(path.relative_to(REPOSITORY))]["sha256"]
            if sha(path) != expected:
                raise ValueError(f"Changed publication record: {path}")
            record_source(path)
    return {row["method"]: row for row in rows}


def object_history(row):
    frame = read_history(row, root=REPOSITORY)
    points = frame[["epoch", "object_nmse"]].dropna()
    expected = np.arange(0, row["epochs"] + 1, 20)
    if (not np.array_equal(points.epoch.to_numpy(), expected)
            or not np.isfinite(points.to_numpy()).all()
            or (points.object_nmse <= 0).any()
            or points.object_nmse.iloc[-1] != row["object_nmse"]):
        raise ValueError(f"Invalid or incomplete saved object-error history: {row['id']}")
    return points


def window_width(truth):
    width = int(truth["probe_truth"].shape[-1])
    if truth["probe_truth"].shape != (128, 128):
        raise ValueError("The publication overlap convention requires the original 128-pixel probe window.")
    return dict(criterion="Original computational-window linear overlap",
                window_width_pixels=width,
                linear_overlap_rule="1 - nominal_step / computational_probe_window_width",
                cases=[dict(original_window_overlap=o, nominal_step_pixels=STEPS[o],
                            scan_positions=POSITIONS[o],
                            nominal_linear_overlap_percent=100*(1-STEPS[o]/width),
                            baseline_batch_size=27, baseline_batch_percent=100*27/POSITIONS[o])
                       for o in OVERLAPS])


def save(fig, stem, outputs, *, dpi=240):
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, facecolor="white")
        outputs.append(path)
    plt.close(fig)


def curve_grid(rows, titles, budgets, outputs, stem):
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5), sharey=True)
    for index, (axis, methods, title, epochs) in enumerate(zip(axes, rows, titles, budgets)):
        for method in METHODS:
            points = object_history(methods[method])
            if not points.object_nmse.between(10**-4.5, 1).all():
                raise ValueError("A trajectory exceeds the existing shared publication scale.")
            axis.plot(points.epoch, points.object_nmse, color=COLORS[method], label=LABELS[method],
                      linewidth=1.2, solid_capstyle="round")
        _style_axis(axis, (10**-4.5, 1))
        axis.set_xlim(0, epochs)
        axis.set_xticks({3000:[0,1000,2000,3000],
                        800:[0,200,400,600,800], 200:[0,50,100,150,200]}[epochs])
        axis.set_xlabel("Epoch")
        axis.set_title(f"({chr(97+index)}) {title}")
    axes[0].set_ylabel("Object error")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.53, .005),
               ncol=4, frameon=False, handlelength=2.2, columnspacing=1.4)
    fig.subplots_adjust(left=.105, right=.975, bottom=.28, top=.88, wspace=.16)
    save(fig, stem, outputs)


def noise_figure(overlap_labels, outputs):
    previous_style = plt.rcParams.copy()
    plt.rcParams.update({"font.size":8, "axes.titlesize":7.4,"axes.labelsize":8,
                        "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":8,"axes.linewidth":.55})
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.45), sharey=True)
    evidence = []
    for index, (axis, overlap) in enumerate(zip(axes, OVERLAPS)):
        selected = {eta: checked_selected("noise", overlap=overlap, eta=eta, batch_size=27) for eta in (.1,.2,.4)}
        for method in METHODS:
            values = [selected[eta][method]["object_nmse"] for eta in (.1,.2,.4)]
            axis.plot([.1,.2,.4], values, color=COLORS[method], label=LABELS[method], marker="o",
                      markersize=3.5, markeredgewidth=0, linewidth=1.2)
            evidence.extend(dict(case_id=selected[eta][method]["id"], method=method, overlap_ratio=overlap,
                                 eta=eta, epochs=selected[eta][method]["epochs"],
                                 object_nmse=selected[eta][method]["object_nmse"]) for eta in (.1,.2,.4))
        _style_axis(axis, (10**-4.5, .1))
        axis.set_xticks([.1,.2,.4])
        axis.set_xlim(.085,.415)
        axis.set_xlabel(r"Noise parameter $\eta$")
        axis.set_title(f"({chr(97+index)}) {overlap_labels[overlap]:g}% overlap",fontsize=8.3)
    axes[0].set_ylabel("Final object NMSE")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.53,.005),
               ncol=4,frameon=False,handlelength=2.2,columnspacing=1.4)
    fig.subplots_adjust(left=.105,right=.975,bottom=.28,top=.88,wspace=.16)
    save(fig,"numerical_noise_summary",outputs)
    plt.rcParams.update(previous_style)
    return evidence


def objects_figure(truth, selected, overlap_label, outputs):
    reference = truth["truth"][ROI]
    fields = [reference]
    for method in METHODS:
        candidate = load_fields(selected[method], root=REPOSITORY)
        record_source(file_path(selected[method], "reconstruction", root=REPOSITORY))
        if not np.array_equal(candidate["roi_mask"], truth["roi_mask"]):
            raise ValueError("A selected reconstruction has a different scoring region.")
        field = candidate["object_aligned"][ROI]
        nmse = float(np.sum(np.abs(reference-field)**2,dtype=np.float64)/np.sum(np.abs(reference)**2,dtype=np.float64))
        if abs(nmse-selected[method]["object_nmse"])>1e-12:
            raise ValueError("Saved object no longer reproduces its endpoint error.")
        fields.append(field)
    errors = [np.abs(field.astype(np.complex128)-reference.astype(np.complex128)) for field in fields]
    if max(np.abs(field).max() for field in fields)>2 or max(error.max() for error in errors)>.8:
        raise ValueError("An existing shared reconstruction display limit clips its data.")
    width,height,panel,rowgap=6.5,3.8382,.9594,.12
    fig=plt.figure(figsize=(width,height))
    cursor=height-.16
    fig.text(.501,(cursor-.015)/height,
             f"{overlap_label:g}% overlap · η = 0.1 · batch 27 ({100*27/POSITIONS[.875]:.2f}%) · 800 epochs",
             ha="center",va="top",fontsize=9)
    for row in range(3):
        y=cursor-.40-row*(panel+rowgap)-panel
        for col,(field,error) in enumerate(zip(fields,errors)):
            axis=fig.add_axes([(.112*width+col*(panel+.065))/width,y/height,panel/width,panel/height])
            if row==0:
                handle=_show(axis,np.abs(field),Normalize(0,1))
                axis.set_title("Ground truth" if col==0 else LABELS[METHODS[col-1]],
                               color="black" if col==0 else COLORS[METHODS[col-1]],fontsize=8.5,pad=4)
            elif row==1:
                handle=_show(axis,np.angle(field.astype(np.complex128)),Normalize(-np.pi,np.pi))
            else:
                handle=_show(axis,np.maximum(error,1e-3),LogNorm(1e-3,.8),"gray_r")
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("#999999" if col==0 else COLORS[METHODS[col-1]])
                spine.set_linewidth(.8)
            if col==0:
                axis.set_ylabel(("Magnitude","Phase","Abs. complex error")[row],fontsize=8.3,labelpad=7)
        ticks=([0,.5,1],[-np.pi,0,np.pi],[1e-3,1e-2,1e-1,.8])[row]
        labels=(None,[r"$-\pi$","0",r"$\pi$"],[r"$10^{-3}$",r"$10^{-2}$",r"$10^{-1}$","0.8"])[row]
        bar=fig.colorbar(handle,cax=fig.add_axes([.912,y/height,.014,panel/height]),ticks=ticks)
        bar.outline.set_linewidth(.45)
        bar.ax.tick_params(labelsize=7.3,length=2,width=.45,pad=2)
        bar.ax.yaxis.set_minor_locator(NullLocator())
        if labels:
            bar.set_ticklabels(labels)
    save(fig,"numerical_0875_baseline_objects",outputs,dpi=220)
    return dict(magnitude_limits=[0,1],phase_limits=[-np.pi,np.pi],error_limits=[1e-3,.8],
                magnitude_colormap="gray; same fixed [0,1] scale as Figure 2",
                actual_magnitude_max=[float(np.abs(field).max()) for field in fields],
                magnitude_fraction_above_display_max=[float((np.abs(field)>1).mean()) for field in fields],
                panel_frames=dict(ground_truth="#999999", methods=COLORS, linewidth_points=.8),
                error_transform="LogNorm",actual_max_error=float(max(error.max() for error in errors)),
                fields_unchanged=True, epochs=800, original_window_overlap=.875,
                nominal_linear_overlap_percent=overlap_label,batch_percent=100*27/POSITIONS[.875])


def main(argv=None):
    global OUTPUT, SUPPORTING
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-overlap",action="store_true",help="Generate unaffected noise, batch, and representative-object assets only")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args=parser.parse_args(argv)
    OUTPUT = args.output.expanduser().resolve()
    SUPPORTING = OUTPUT / "evidence"
    SUPPORTING.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    index=load_index(REPOSITORY)
    record_source(REPOSITORY/"results/manifest.json")
    record_source(REPOSITORY/"results/index.json")
    record_source(REPOSITORY/index["synthetic_truth"]["path"])
    truth=_truth(index,REPOSITORY)
    overlap_definition=window_width(truth)
    overlap_labels={case["original_window_overlap"]:case["nominal_linear_overlap_percent"] for case in overlap_definition["cases"]}
    baseline={o:checked_selected("overlap",overlap=o,eta=.1,batch_size=27) for o in OVERLAPS}
    batch={b:checked_selected("batch",overlap=.875,eta=.1,batch_size=b) for b in BATCHES}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"axes.titlesize":8.5,
                        "axes.labelsize":8.5,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8.5,
                        "axes.linewidth":.6,"pdf.fonttype":42,"ps.fonttype":42,
                        "image.interpolation":"none","image.resample":False})
    outputs=[]
    if not args.skip_overlap:
        curve_grid([baseline[o] for o in OVERLAPS],
                   [f"{overlap_labels[o]:g}% overlap" for o in OVERLAPS],
                   [3000,800,200],outputs,"numerical_overlap_convergence")
    curve_grid([batch[b] for b in BATCHES],[f"Batch {b} ({100*b/POSITIONS[.875]:.2f}%)" for b in BATCHES],
               [800]*3,outputs,"numerical_batch_convergence")
    noise_evidence=noise_figure(overlap_labels,outputs)
    object_evidence=objects_figure(truth,baseline[.875],overlap_labels[.875],outputs)
    if any(sha(path)!=digest for path,digest in SOURCE_HASHES.items()):
        raise ValueError("An input scientific record changed during plotting.")
    evidence=dict(status="complete",
                  source_archive_unchanged=True,source_files_sha256=SOURCE_HASHES,
                  display_overlap_definition=overlap_definition,
                  batch_percentage_definition="100 * batch_size / total retained scan positions; no computational batch sizes changed",
                  batch_875_percentages={str(b):100*b/POSITIONS[.875] for b in BATCHES},
                  overlap_figure_written=not args.skip_overlap,
                  noise_endpoints=noise_evidence,
                  noise_epoch_budgets=[3000,800,200],
                  other_overlap_noise_endpoints_unchanged=True,
                  representative_object_display=object_evidence,
                  trajectory_rule="Raw epoch-0 and every-20-epoch saved object scores; no smoothing or minimum-over-epochs selection",
                  script_sha256=sha(__file__),output_files_sha256={str(path):sha(path) for path in outputs})
    destination=SUPPORTING/"synthetic_figures.json"
    destination.write_text(json.dumps(evidence,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=evidence["status"],outputs=[str(path) for path in outputs],evidence=str(destination),
                         probe_window_width=overlap_definition["window_width_pixels"],overlap_labels=overlap_labels),indent=2))


if __name__=="__main__":
    main()
