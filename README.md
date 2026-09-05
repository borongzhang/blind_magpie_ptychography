# MAGPIE for blind ptychography

Code and experiments for [Stochastic Multigrid Method for Blind Ptychographic
Phase Retrieval](https://arxiv.org/abs/2511.01793), by Borong Zhang, Junjing Deng,
Yi Jiang, and Zichao Wendy Di.

The notebooks compare rPIE, GM-rPIE, GM-MAGPIE, and LSQML using one shared
implementation and Pty-Chi 1.4.0.

## Installation

Use Python 3.11. The experiments use PyTorch 2.11 and torchvision 0.26; on an
NVIDIA machine, install their CUDA-enabled builds for your GPU first.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks]"
python -m ipykernel install --sys-prefix --name python3 --display-name "Python (MAGPIE)"
jupyter lab
```

Start Jupyter from the repository root, select this environment's kernel,
then restart the kernel and run the selected notebook from top to bottom.
Notebook settings near the top specify the seeds and all method parameters.

## Experiments

| Study | Notebooks | Device | Settings |
|---|---|---|---|
| Overlap | [examples/synthetic](examples/synthetic) | Apple MPS | 50%, 62.5%, 75% overlap; eta=0.05; 3,000 epochs; batch 81 |
| Noise | [examples/synthetic_noise](examples/synthetic_noise) | NVIDIA CUDA | 75% overlap; eta=0.05, 0.5, 1; 3,000 epochs; batch 81 |
| Real data | [examples/real_data](examples/real_data) | NVIDIA CUDA | Full/half chip and full/quarter test pattern |

The overlap notebooks retain the local MPS setup; the noise and real-data
notebooks explicitly select CUDA. They fail if their requested device is
unavailable. In CUDA notebooks, `SMOKE_TEST = True` runs one epoch per method
before committing to the full experiment. Real-data smoke runs use one batch;
synthetic smoke runs retain the full generated dataset.

The synthetic studies use a 1,024 × 1,024 object, a 128 × 128 probe,
position/data/reconstruction seeds 42/42/21, and scaled-Poisson measurements
`Y = eta * Poisson(I / eta)`. Eta is not a noise percentage.

| Real-data notebook | Patterns | Detector | Batch | Epochs |
|---|---:|---:|---:|---:|
| [chip_full.ipynb](examples/real_data/chip_full.ipynb) | 812 | 512 × 512 | 29 | 200 |
| [chip_half_scan.ipynb](examples/real_data/chip_half_scan.ipynb) | 406 | 512 × 512 | 29 | 200 |
| [test_pattern_full.ipynb](examples/real_data/test_pattern_full.ipynb) | 14,592 | 256 × 256 | 64 | 100 |
| [test_pattern_quarter_scan.ipynb](examples/real_data/test_pattern_quarter_scan.ipynb) | 3,648 | 256 × 256 | 64 | 300 |

All four real comparisons use seed 42 and Gaussian LSQML with sigma=0.5 and
tied step-size scaler=0.5. Full experiment parameters remain in each notebook.

## Data

The two synthetic input images are included in `assets/`. Supply the measured
HDF5 files separately in [assets/ptycho_real_data](assets/ptycho_real_data/README.md).
Large datasets and reconstruction arrays are excluded from this distribution.

## Recorded results

The saved summary values are collected in three files:

- [overlap.csv](results/overlap.csv)
- [noise.csv](results/noise.csv)
- [real_data.csv](results/real_data.csv)

Each row retains the experiment, original run ID, backend, method, settings,
and full-precision summary values. These are existing results, not reruns of
the cleaned code. The eta=0.05 noise reference is the same MPS run as the 75%
overlap reference; the corresponding CUDA notebook prepares a future rerun.
There are nine unique recorded runs across the ten study configurations.

Synthetic summaries distinguish frozen noisy/clean amplitude MSE from the
online pre-update residual. Real summaries contain the online residual.
Timings span different hardware and measurement procedures: the original MPS
runs did not explicitly synchronize the accelerator, while CUDA runs did.
They should not be combined into a controlled runtime comparison. Fixed seeds
do not guarantee bitwise equality across accelerators or MPS reruns.

Running a notebook creates a new full archive under `results/notebook_exports/`,
including reconstruction arrays, curves, settings, and checksums. These generated
files are ignored by Git.

The notebooks include preserved outputs from the original runs, including plots
and printed metrics. They have not been rerun after code cleanup. Personal paths
in text logs are replaced with `<original-project>`.

The CUDA eta=0.05 notebook has no saved outputs; use the linked MPS baseline.
The quarter test-pattern notebook contains a partial saved history, stopping at
GM-MAGPIE epoch 200; its complete 300-epoch result is in `results/real_data.csv`.
The 62.5% overlap notebook's original source differs from its archived source
hash, although displayed summary values match the saved results. These cases
are also noted inside the relevant notebooks.

## Layout

```text
src/          shared algorithms, simulation, data loading, metrics, and saving
examples/     overlap, noise, and real-data notebooks
assets/       synthetic inputs and measured-data instructions
results/      three saved summary tables
```

This folder can replace the code in a checkout of
[blind_magpie_ptychography](https://github.com/borongzhang/blind_magpie_ptychography).
Preserve the checkout's `.git` directory and remove superseded example files
when copying. No license was specified in the original project.
