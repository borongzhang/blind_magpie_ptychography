# Geometric MAGPIE Experiments

This repository contains small, focused experiments comparing Pty-Chi rPIE with
the geometric MAGPIE update on synthetic and chip ptychography data.

## Layout

- `src/`: reusable Python code.
- `examples/synthetic/`: synthetic-data notebooks.
- `examples/chip/`: real-chip notebooks.
- `assets/`: small image/probe assets and local-only real data.

## Setup

From the repository root:

```bash
conda activate magpie
python -m pip install -e .
```

The notebooks also add `src/` to `sys.path`, so they can be run directly from a
fresh checkout.

## Examples

- `examples/synthetic/magpie_synthetic.ipynb` compares rPIE and geometric
  MAGPIE on synthetic data with known ground truth.
- `examples/synthetic/geometric_mean_test.ipynb` checks the aligned complex
  geometric mean used by MAGPIE.
- `examples/chip/magpie_real_chip.ipynb` compares rPIE and geometric MAGPIE on
  the chip dataset.

Real chip HDF5 files are kept under `assets/ptycho_real_data/` locally, but are
ignored by git because they are large.
