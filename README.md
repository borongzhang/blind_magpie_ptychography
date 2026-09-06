# MAGPIE for blind ptychography

Code and experiments for [Stochastic Multigrid Method for Blind Ptychographic
Phase Retrieval](https://arxiv.org/abs/2511.01793), by Borong Zhang, Junjing Deng,
Yi Jiang, and Zichao Wendy Di.

The notebooks compare rPIE, GM-rPIE, GM-MAGPIE, and LSQML using shared
source modules and Pty-Chi 1.4.0. All included experiments run on NVIDIA CUDA.

## Installation

Use Python 3.11. Install CUDA-enabled PyTorch 2.11 and torchvision 0.26 builds
for your GPU, then install the project:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks]"
python -m ipykernel install --sys-prefix --name python3 --display-name "Python (MAGPIE)"
jupyter lab
```

Start Jupyter from the repository root, select this environment's kernel,
then restart the kernel and run a notebook from top to bottom. The settings
near the top specify the seeds and all method parameters. CUDA is required;
the notebooks fail early when a GPU is unavailable.

## Experiments

| Study | Notebooks | Settings |
|---|---|---|
| Overlap | [examples/synthetic](examples/synthetic) | 62.5%, 75%, 87.5% overlap; eta=0.05; 3,000 epochs; batch 81 |
| Noise | [examples/synthetic_noise](examples/synthetic_noise), plus the [shared eta=0.05 baseline](examples/synthetic/synthetic_overlap_075_eta_0p05.ipynb) | 75% overlap; eta=0.05, 0.2, 0.4; 3,000 epochs; batch 81 |
| Real data | [examples/real_data](examples/real_data) | Full/half chip and full/one-third test pattern; refined scan positions |

The nine notebooks cover nine unique recorded runs. The 75%-overlap,
eta=0.05 CUDA run is shared by the overlap and noise comparisons and appears
only once in the notebook tree.

The synthetic studies use a 1,024 × 1,024 object, a 128 × 128 probe,
position/data/reconstruction seeds 42/42/21, and scaled-Poisson measurements
`Y = eta * Poisson(I / eta)`. Eta is not a noise percentage. All synthetic
studies use object/probe alphas (0.1, 0.1) for rPIE, (0.01, 0.01) for GM-rPIE,
and (0.05, 0.01) for GM-MAGPIE. Synthetic LSQML uses a Gaussian reconstruction
likelihood with sigma=0.5 and object/probe optimal-step multipliers 0.9/0.9;
its outer SGD step sizes are 1.0/1.0. Scan positions remain fixed in the
synthetic reconstructions.

| Real-data notebook | Patterns | Detector | Batch | Epochs |
|---|---:|---:|---:|---:|
| [chip_full.ipynb](examples/real_data/chip_full.ipynb) | 812 | 512 × 512 | 29 | 50 |
| [chip_half_scan.ipynb](examples/real_data/chip_half_scan.ipynb) | 406 | 512 × 512 | 29 | 100 |
| [test_pattern_full.ipynb](examples/real_data/test_pattern_full.ipynb) | 14,640 | 256 × 256 | 244 | 50 |
| [test_pattern_third_scan.ipynb](examples/real_data/test_pattern_third_scan.ipynb) | 4,880 | 256 × 256 | 244 | 100 |

The measured-data subsets are sampled across the full scan using seed 42.
The full test-pattern notebook retains 14,640 of 14,641 patterns to form
complete batches. All four real methods refine scan positions from the
first epoch (zero-based start 0), using gradient correction, step size 1,
a 50-pixel cap per coordinate per update, and a fixed position mean.
Gaussian LSQML uses sigma=0.5 and tied optimal-step multipliers 0.5/0.5,
with outer SGD step sizes 1.0/1.0. Snapshots include complex object/probe
fields and refined positions every 10 epochs. Complete method settings
remain in each notebook.

All notebooks support `SMOKE_TEST = True`. Synthetic smoke runs use one
epoch on the full generated dataset; real-data smoke runs use one epoch and
one minibatch. Smoke archives have separate `_smoke` directories.

## Data

The two synthetic input images are included in `assets/`. Supply the measured
HDF5 files separately in [assets/ptycho_real_data](assets/ptycho_real_data/README.md).
Large datasets and reconstruction arrays are excluded from this distribution.

## Recorded results

Full-precision summaries retain the original run IDs, backends, and settings:

- [overlap.csv](results/overlap.csv)
- [noise.csv](results/noise.csv)
- [real_data.csv](results/real_data.csv), including RMS and maximum position shifts

The notebooks and CSVs come from the completed server runs in
`magpie_from_pace_20260906_145649`. Notebook code and source implementations
were checked against their archived manifests. All original notebook outputs
are preserved, including synthetic metric/object/probe plots, real-data
snapshots, and final scan-position-correction plots. No reconstruction or
plot was rerun during cleanup. Server paths in text logs are replaced with
`<original-project>`. The synthetic LSQML defaults are written explicitly
in the notebooks with their recorded values.

Synthetic summaries distinguish frozen noisy/clean amplitude MSE from the
online pre-update residual. Real-data summaries report the online residual;
they do not provide a truth-based reconstruction error or measured resolution.
The recorded runs used an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU,
PyTorch 2.11.0 with CUDA 13.0, and CUDA synchronization around timed runs.
Synthetic timings include sampled frozen/truth metrics; real-data timings
include periodic snapshot saving and display. Fixed seeds do not guarantee
bitwise equality across accelerators or reruns.

Running a notebook creates an immutable archive under
`results/notebook_exports/<study>/`, including reconstruction arrays, curves,
settings, and checksums. Real-data archives include initial/refined positions
and periodic snapshots. Generated archives are ignored by Git.

## Layout

```text
src/          shared algorithms, simulation, data loading, metrics, and saving
examples/     five synthetic and four real-data notebooks with outputs
assets/       synthetic inputs and measured-data instructions
results/      three saved summary tables
```

No license was specified in the original project.
