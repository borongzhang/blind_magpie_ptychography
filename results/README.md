# Published experiment records

`index.json` is the entry point. Paths in it are relative to the repository
root. Its `cases` list contains 247 distinct terminal runs:

| Physical study | Runs | Complete | Failed |
|---|---:|---:|---:|
| Overlap parameter sweeps | 129 | 121 | 8 |
| Additional noise levels | 24 | 24 | 0 |
| Additional batch sizes | 86 | 85 | 1 |
| Real datasets | 8 | 8 | 0 |

The synthetic sweeps use nine pairs for each PIE method and sixteen for LSQML.
Parameters are selected by the lowest **final** object NMSE among completed
candidates in each method and condition. Failures retain their reason and saved
partial history; they have no final reconstruction score and cannot be selected.

The `studies` mapping describes logical comparisons. The noise comparison reuses
the twelve selected overlap runs at noise parameter 0.1, giving 36 records in
`noise_endpoints.csv`. The batch comparison reuses all 43 overlap candidates at
87.5% and batch size 27, giving 129 records in `batch_endpoints.csv`. Reused runs
refer to the same case IDs and files. `selections` gives the selected case IDs
for each logical comparison. `selected.csv` contains the 52 distinct selected
runs, without duplicated baselines.

Each case has portable `config.json` and `result.json` files plus a byte-for-byte
copy of its original `history.csv` in `records/{study}/{id}/`. Source identity
hashes identify the original records; `manifest.json` hashes the exported files.
Object error is the aligned complex NMSE on the common synthetic evaluation
region. Real datasets have no ground truth, so their object-error fields are
empty; their saved residuals are diagnostics rather than quality rankings.

`reconstructions/` contains the four selected 87.5% baseline reconstructions and
the eight 100-epoch real-data reconstructions used in the paper. These twelve
NPZ files are lossless copies. `synthetic_truth.npz` contains only the four
original truth/initialization arrays needed to regenerate the figures, with
unchanged array hashes. Measurement caches and other reconstruction arrays are
not included. `figures/` contains only the seven current manuscript figures,
each as PDF and PNG.

Archived metadata aliases under `analysis/` preserve the original file hashes
required by the frozen rerun protocols. Use `index.json` and `records/` for
analysis and plotting.
