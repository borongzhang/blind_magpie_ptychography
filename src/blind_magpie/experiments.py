"""Run the synthetic comparisons from their recorded parameter tables."""
from __future__ import annotations

from pathlib import Path

from .protocol import METHODS, OVERLAPS
from .results import endpoints, selected_cases


def experiment_plan(study, *, mode="selected", overlap=None, eta=None,
                    batch_size=None, method=None, epochs=None, root=None):
    """List exact settings without importing PyTorch or starting a reconstruction.

    ``selected`` repeats the paper's object-selected pairs. ``sweep`` repeats
    every searched pair for overlap or batch; noise keeps the baseline pair.
    Filters allow running one method and condition at a time.
    """
    if study not in ("overlap", "noise", "batch"):
        raise ValueError("Synthetic studies are overlap, noise, or batch.")
    if mode not in ("selected", "sweep") or (study == "noise" and mode != "selected"):
        raise ValueError("Choose selected, or sweep for the overlap/batch study.")
    if overlap is not None and overlap not in OVERLAPS:
        raise ValueError(f"overlap must be one of {OVERLAPS}")
    if method == "all":
        method = None
    if method is not None and method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if eta is not None and eta not in (.1, .2, .4):
        raise ValueError("eta must be 0.1, 0.2, or 0.4")
    if epochs is not None and (not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1):
        raise ValueError("epochs must be a positive integer")
    if mode == "selected":
        rows = selected_cases(study, root)
    else:
        rows = endpoints(root)
        if study == "overlap":
            rows = rows.loc[rows.study == "overlap"]
        else:
            rows = rows.loc[(rows.study == "batch") |
                            ((rows.study == "overlap") & (rows.overlap == .875))]
    for name, value in (("overlap", overlap), ("eta", eta),
                        ("batch_size", batch_size), ("method", method)):
        if value is not None:
            rows = rows.loc[rows[name] == value]
    if rows.empty:
        raise ValueError("These filters do not match a recorded synthetic configuration.")
    plan = []
    for _, row in rows.iterrows():
        names = ("damping", "sigma") if row.method == "lsqml" else ("object_alpha", "probe_alpha")
        plan.append(dict(method=row.method, overlap=float(row.overlap), eta=float(row.eta),
                         batch_size=int(row.batch_size), epochs=epochs or int(row.epochs),
                         parameters={key: float(row[key]) for key in names}))
    return plan


def run_experiments(study, *, mode="selected", output="runs", device="cuda",
                    overlap=None, eta=None, batch_size=None, method=None,
                    epochs=None, smoke=False, dry_run=False, root=None):
    """Repeat synthetic cases, writing new results under ``output``.

    Existing completed cases are reused by the runner. Numerical failures are
    recorded and the remaining pairs continue. ``smoke=True`` uses the tiny
    development dataset; its results are separate from the paper experiment.
    """
    import pandas as pd
    plan = experiment_plan(study, mode=mode, overlap=overlap, eta=eta,
                           batch_size=batch_size, method=method,
                           epochs=2 if smoke and epochs is None else epochs, root=root)
    if dry_run:
        return pd.DataFrame(plan)
    from .reconstruct import run_case
    output = Path(output).expanduser().resolve()
    results = []
    for index, case in enumerate(plan, 1):
        print(f"Case {index}/{len(plan)}: {case}", flush=True)
        result = run_case(**case, output=output, device=device, smoke=smoke)
        results.append(result)
    frame = pd.DataFrame(results)
    frame.to_csv(output / f"{study}_{mode}_summary.csv", index=False)
    return frame
