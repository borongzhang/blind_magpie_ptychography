# Blind MAGPIE

Synthetic reproduction code and recorded results for **Stochastic Multigrid Framework for Blind Ptychography**, by Borong Zhang, Junjing Deng, Yi Jiang, and Zichao Wendy Di ([paper](https://arxiv.org/abs/2511.01793)). The methods are rPIE, LSQML, GM-rPIE, and GM-MAGPIE.

## View the results

These notebooks contain **executed cells, tables, convergence plots, and reconstruction images**. Open them directly on GitHub:

| Notebook | Contents |
|---|---|
| [1. Synthetic overlap](examples/01_overlap.ipynb) | Ground truth and initialization, parameter sweeps, object and probe errors, selected reconstructions |
| [2. Synthetic noise](examples/02_noise.ipynb) | Object and probe errors at η = 0.1, 0.2, and 0.4 |
| [3. Synthetic batch size](examples/03_batch.ipynb) | Retuned sweeps and convergence at three minibatch sizes |
| [4. Real reconstructions](examples/04_real_reconstructions.ipynb) | Test-pattern and integrated-circuit object/probe images and settings |

The real-data notebook displays the recorded reconstructions. Raw experimental measurements are not included.

## Run the notebooks locally

Use Python 3.11:

```bash
git clone --depth 1 https://github.com/borongzhang/blind_magpie_ptychography.git
cd blind_magpie_ptychography
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebooks]"
jupyter lab
```

The saved-result cells run on CPU. Synthetic reconstruction cells are optional and disabled by default.

## Reproduce synthetic reconstructions

Use Linux with an NVIDIA GPU for the full runs. Install a CUDA-enabled PyTorch build for your system, then install the reconstruction dependencies:

```bash
python -m pip install -e ".[reconstruction]"
```

The reconstruction extra pins [Pty-Chi](https://github.com/AdvancedPhotonSource/pty-chi) to commit `9bff0739cc91b9ec58f49341429bd955a767389d` and the recorded numerical package versions. Synthetic inputs are generated from the two included images and fixed random seeds.

Repeat the selected methods at 87.5% overlap:

```bash
python -m blind_magpie --study overlap --overlap 0.875
```

Repeat a parameter sweep, noise study, or batch-size study:

```bash
python -m blind_magpie --study overlap --overlap 0.875 --mode sweep
python -m blind_magpie --study noise --overlap 0.875
python -m blind_magpie --study batch --batch-size 108 --mode sweep
```

Add `--method gm_magpie` to run one algorithm or `--dry-run` to list the settings. New configurations, errors, and reconstruction arrays are written under `runs/`. Completed matching cases are reused. The Python entry points are `blind_magpie.run_case(...)` and `blind_magpie.run_experiments(...)`.

For a short CPU installation check on a small diagnostic dataset:

```bash
python -m blind_magpie --study overlap --overlap 0.875 --method gm_magpie --smoke --output smoke_runs
```

## Experiment settings

The synthetic object is 1024 × 1024 and the probe is 128 × 128. Object magnitude and phase come from the Baboon and Cameraman images. The true and initial probes use −700 µm and −400 µm defocus settings. Measurements use `d = η Poisson(d_noiseless / η)`.

| Overlap | Scan step | Scan positions | Epochs | Minibatch |
|---|---:|---:|---:|---:|
| 75% | 32 | 841 | 3,000 | 27 (3.21%) |
| 87.5% | 16 | 3,249 | 800 | 27 (0.83%) |
| 93.75% | 8 | 12,769 | 200 | 27 (0.21%) |

Overlap is `1 − scan step / probe array width`. The baseline noise parameter is η = 0.1. The batch study uses 27, 108, and 432 positions at 87.5% overlap, with parameters retuned at each size.

PIE methods search both alphas in `[0.004, 0.02, 0.1]`. LSQML searches damping `[0.04, 0.1, 0.2, 1]` and σ `[0.3, 0.4, 0.5, 0.6]`. Each method's pair minimizes its final **object NMSE**. Noise tests retain the baseline pair. Errors are recorded every 20 epochs; progress is printed every 300 epochs.

Object NMSE uses the common scanned region after complex-scale and affine-phase alignment. Probe NMSE uses the full probe with the paired ambiguity correction and remaining global phase removed. Real-data reconstruction quality is shown through the images because ground truth is unavailable.

## Files

```text
examples/          four executed notebooks
src/blind_magpie/  synthetic generator, reconstruction loop, result plots
results/           compact endpoint/history tables, settings, final images
```

`results/endpoints.csv` contains all 239 synthetic candidates, including nine failed candidates. `results/convergence.csv.gz` contains the recorded checkpoints for the 44 distinct selected synthetic cases; shared baselines are stored once. The notebooks plot these saved values. `results/protocol.json` records settings, selections, and input identities. Real-data images and their eight method configurations are included separately.

Distributed under the [MIT license](LICENSE). Please cite the paper when using this work.
