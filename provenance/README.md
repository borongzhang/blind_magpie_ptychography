# Experiment identities

The current overlap-selected pairs are in `short_noise_parameters.json`. The batch study is defined in `batch_0875_protocol.json`; `real_protocol.json` records the measured-data setup.

The scientific runners and the independent `studies/reference_075/src/` package retain their recorded source bytes. Follow-up runners verify the original baseline records under `results/analysis/` before using a selected pair. These files are input identities; the portable result index is `results/index.json`.

`selected_parameters.json` is a source-identity fixture referenced by the archived runner checks. Use `short_noise_parameters.json` and the result index for the current experimental selections.

The measured-data setup refers to the original notebooks in commit [`9baaf92`](https://github.com/borongzhang/blind_magpie_ptychography/tree/9baaf92c57f0ad067edf5005b9050a1bccf0dffa). That commit remains in this repository's history; the current measured-data reconstructions are the eight 100-epoch runs indexed in `results/index.json`.
