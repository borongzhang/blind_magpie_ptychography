# Measured data

The included real-data notebooks display the recorded 100-epoch reconstructions without requiring the measured inputs. To repeat those reconstructions, place the original HDF5 files here:

| Dataset | File |
|---|---|
| Test pattern | `Velo_18c3_comm_TP_scan119_data_roi0_Ndp256_dp.hdf5` |
| Integrated circuit | `Velo_18c3_comm_chip65nm_scan054_data_roi0_Ndp512_us2.hdf5` |

The loader reads `dp` for intensities, `ppX` and `ppY` for scan positions, `dx` for pixel spacing, and `lambda` for wavelength. Position and length values use metres, as in the original files.

The measured files are not redistributed with the code. Obtain the original files from the authors or data provider and retain their applicable data-use terms and acknowledgments.

The experimental setup and integrated-circuit data are described in:

- Deng et al., [The Velociprobe: An ultrafast hard X-ray nanoprobe for high-resolution ptychographic imaging](https://doi.org/10.1063/1.5103173), 2019.
- Deng et al., [Nanoscale x-ray imaging of circuit features without wafer etching](https://doi.org/10.1103/PhysRevB.95.104111), 2017.

See the [repository README](../../README.md) for the reconstruction notebooks and method citation, and [LICENSE](../../LICENSE) for the code license.
