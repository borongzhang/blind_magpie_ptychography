# Measured data

Place these two files in this directory:

- `Velo_18c3_comm_chip65nm_scan054_data_roi0_Ndp512_us2.hdf5`
- `Velo_18c3_comm_TP_scan119_data_roi0_Ndp256_dp.hdf5`

The loader expects `dp` (intensities), `ppX` and `ppY` (scan positions in metres),
`dx` (pixel size in metres), and `lambda` (wavelength in metres).

These large files are excluded from the code distribution and ignored by Git.
No public download location was supplied; obtain them from the project authors
or data provider.
