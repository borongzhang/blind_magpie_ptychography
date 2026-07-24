# MAGPIE for Blind Ptychography

Reference implementation and reproducible experiments for MAGPIE-based blind
ptychographic reconstruction.

This repository accompanies the forthcoming paper:

> **[Paper title]**  
> [Authors], [journal or arXiv], [year]  
> [Paper link / DOI]

The code compares five blind object-and-probe reconstruction methods in a
common [Pty-Chi](https://github.com/AdvancedPhotonSource/pty-chi) pipeline.
The final paper title, citation, and method capitalization will be inserted
after the revised manuscript is available.

## Algorithms

| Method | Local object proposal | Local probe proposal | Geometric mean and weighted synthesis |
|---|---|---|---|
| **rPIE** | rPIE | rPIE | No |
| **MAGPIE-O** | MAGPIE | rPIE | No |
| **GM-rPIE** | rPIE | rPIE | Yes |
| **GM-MAGPIE-O** | MAGPIE | rPIE | Yes |
| **GM-MAGPIE-OP** | MAGPIE | MAGPIE | Yes |

`MAGPIE-O` changes only the local object update in native rPIE. It retains the
native rPIE probe update and update application, without a geometric-mean step
or custom minibatch synthesis.

The GM methods geometrically average each current local estimate with its
updated proposal. For minibatches larger than one, the resulting local object
and probe estimates are combined with counterpart-intensity-weighted
synthesis. The probe synthesis uses the adjoint of the same fractional shift
operator used by the forward model.

All MAGPIE-based methods in the final experiments use every valid multigrid
level. The experiments use the native Pty-Chi fractional-position convention:
object patches are extracted at integer scan anchors and the fractional
offsets are applied to the probe.

## Reproducibility conventions

Within each notebook, the five methods share the dataset, initialization,
batch size, scan permutations, unit object/probe step sizes, and forward
model. A fresh random permutation of all retained scan positions is traversed
once per epoch, and the object and probe are updated in every minibatch.

The final notebooks enable object-probe ambiguity removal after every epoch.
The algorithm-specific object and probe `alpha` values are exposed near the
top of each notebook and are the only tuned reconstruction hyperparameters.
Arrays used by the reconstruction pipeline are single precision
(`float32`/`complex64`).

## Installation

Python 3.11 or newer is required. The current experiments were developed with
Pty-Chi 1.4.0.

```bash
git clone https://github.com/borongzhang/blind_magpie_ptychography.git
cd blind_magpie_ptychography

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks]"
```

Then start Jupyter from the repository root:

```bash
jupyter lab
```

## Experiments

| Notebook | Experiment | Retained positions |
|---|---|---:|
| [`examples/synthetic/synthetic_final.ipynb`](examples/synthetic/synthetic_final.ipynb) | Synthetic data with known object and probe | Generated in the notebook |
| [`examples/real_data/chip_final.ipynb`](examples/real_data/chip_final.ipynb) | Complete chip scan | 812 |
| [`examples/real_data/chip_partial_scan_final.ipynb`](examples/real_data/chip_partial_scan_final.ipynb) | Every fourth chip position | 203 of 812 |
| [`examples/real_data/test_pattern_final.ipynb`](examples/real_data/test_pattern_final.ipynb) | Complete test-pattern scan | 14,641 |
| [`examples/real_data/test_pattern_quarter_scan_final.ipynb`](examples/real_data/test_pattern_quarter_scan_final.ipynb) | Every fourth test-pattern position | 3,661 of 14,641 |

The synthetic notebook reports truth-based object and probe errors in addition
to the diffraction residual. Since ground truth is unavailable for the real
datasets, the real-data notebooks compare residual histories and reconstructed
object/probe visualizations.

Two focused validation notebooks are also provided:

- [`examples/tests/geometric_mean_product_test.ipynb`](examples/tests/geometric_mean_product_test.ipynb)
  checks the complex geometric-mean product and branch-selection properties.
- [`examples/tests/subpixel_shift_test.ipynb`](examples/tests/subpixel_shift_test.ipynb)
  checks the subpixel-shift convention against Pty-Chi's shift implementation.

## Repository layout

- `src/algorithms/`: rPIE, geometric-mean, and MAGPIE reconstruction code.
- `src/utils/`: shared simulation, data-loading, reconstruction, alignment, and
  plotting utilities.
- `examples/synthetic/`: the matched synthetic comparison.
- `examples/real_data/`: complete and subsampled real-data comparisons.
- `examples/tests/`: focused mathematical and forward-model checks.
- `assets/`: small synthetic assets and local real-data instructions.

The auxiliary LSQML wrapper in `src/algorithms/lsqml.py` is not one of the five
methods compared by the final notebooks.

## Real data

The large experimental HDF5 files are intentionally excluded from Git. Place
the following files in `assets/ptycho_real_data/` before running the real-data
notebooks:

- `Velo_18c3_comm_chip65nm_scan054_data_roi0_Ndp512_us2.hdf5`
- `Velo_18c3_comm_TP_scan119_data_roi0_Ndp256_dp.hdf5`

**Data availability:** [Add the facility, archive/download link, or access
instructions used by the paper.]

## Citation

If this repository contributes to your work, please cite the accompanying
paper:

```bibtex
@article{zhang_magpie_YEAR,
  title   = {[Paper title]},
  author  = {[Authors]},
  journal = {[Journal or arXiv]},
  year    = {[Year]},
  doi     = {[DOI]}
}
```

Please also cite Pty-Chi and the experimental data source where appropriate.

## License

**[Add the selected software license before the public release. Data may have
separate access and reuse conditions.]**
