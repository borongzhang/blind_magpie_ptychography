# Experiment notebooks

All nine notebooks compare rPIE, LSQML, GM-rPIE, and GM-MAGPIE. Their default mode reads the included publication records, plots object-error histories for synthetic data, and displays the paper figures. This works on CPU.

| Study | Notebook | Epochs |
|---|---|---:|
| 75% overlap sweep | [overlap_075.ipynb](overlap/overlap_075.ipynb) | 3,000 |
| 87.5% overlap sweep | [overlap_0875.ipynb](overlap/overlap_0875.ipynb) | 800 |
| 93.75% overlap sweep | [overlap_09375.ipynb](overlap/overlap_09375.ipynb) | 200 |
| Noise at 75% overlap | [noise_075.ipynb](noise/noise_075.ipynb) | 3,000 |
| Noise at 87.5% overlap | [noise_0875.ipynb](noise/noise_0875.ipynb) | 800 |
| Noise at 93.75% overlap | [noise_09375.ipynb](noise/noise_09375.ipynb) | 200 |
| Batch sizes at 87.5% overlap | [batch_0875.ipynb](batch/batch_0875.ipynb) | 800 |
| Test-pattern data | [test_pattern.ipynb](real/test_pattern.ipynb) | 100 |
| Integrated-circuit data | [integrated_circuit.ipynb](real/integrated_circuit.ipynb) | 100 |

Start Jupyter from the repository root using the environment installed in [the main README](../README.md). The notebooks find the repository from their working directory and use `results/index.json` to locate portable records.

`RUN_RECONSTRUCTIONS = False` keeps inspection separate from reconstruction. Set it to `True` to run the optional rerun cell on a CUDA GPU. The output goes to the scientific runner's experiment folders; the inspection cells continue to show the archived publication results. Real-data reruns also need the [original HDF5 files](../assets/ptycho_real_data/README.md).

The η=0.1 noise baselines and batch-27 baseline reuse the overlap records. No shared reconstruction is counted twice in the result index.
