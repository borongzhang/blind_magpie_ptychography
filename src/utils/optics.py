from __future__ import annotations

import numpy as np

__all__ = ["fresnel_propagation", "generate_probe"]


def fresnel_propagation(
    wavefield: np.ndarray,
    pixel_size_m: float,
    propagation_distance_m: float,
    wavelength_m: float,
) -> np.ndarray:
    """Propagate a 2D field with the single-FFT Fresnel transform.

    The returned field is unnormalized because probe power is calibrated after
    propagation. Its natural output sampling is
    ``wavelength_m * propagation_distance_m / (N * pixel_size_m)``.
    """
    wavefield = np.asarray(wavefield, dtype=np.complex64)
    if wavefield.ndim != 2:
        raise ValueError("wavefield must be a two-dimensional array.")
    if not np.isfinite(pixel_size_m) or pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be finite and positive.")
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("wavelength_m must be finite and positive.")
    if not np.isfinite(propagation_distance_m) or propagation_distance_m <= 0:
        raise ValueError("propagation_distance_m must be finite and positive.")

    height, width = wavefield.shape
    pixel_size = np.float32(pixel_size_m)
    propagation_distance = np.float32(propagation_distance_m)
    wavelength = np.float32(wavelength_m)
    wave_number = np.float32(2 * np.pi / wavelength_m)
    imaginary_unit = np.complex64(1j)

    x_axis = (np.arange(width, dtype=np.float32) - width // 2) * pixel_size
    y_axis = (np.arange(height, dtype=np.float32) - height // 2) * pixel_size
    x, y = np.meshgrid(x_axis, y_axis)

    u_axis = (
        wavelength
        * propagation_distance
        * np.fft.fftfreq(width, d=pixel_size).astype(np.float32)
    ).astype(np.float32, copy=False)
    v_axis = (
        wavelength
        * propagation_distance
        * np.fft.fftfreq(height, d=pixel_size).astype(np.float32)
    ).astype(np.float32, copy=False)
    u, v = np.meshgrid(u_axis, v_axis)

    output_phase = np.exp(imaginary_unit * wave_number * propagation_distance)
    output_phase = output_phase * np.exp(
        imaginary_unit
        * wave_number
        * (u**2 + v**2)
        / (np.float32(2) * propagation_distance)
    )
    input_chirp = wavefield * np.exp(
        imaginary_unit
        * wave_number
        * (x**2 + y**2)
        / (np.float32(2) * propagation_distance)
    )
    propagated = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(input_chirp)) * output_phase
    )
    return propagated.astype(np.complex64)


def generate_probe(
    size: int,
    wavelength_m: float,
    sample_pixel_size_m: float,
    defocus_m: float,
) -> np.ndarray:
    """Generate an idealized Velociprobe FZP field at the sample plane."""
    if not isinstance(size, (int, np.integer)) or size < 2:
        raise ValueError("size must be an integer greater than one.")
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("wavelength_m must be finite and positive.")
    if not np.isfinite(sample_pixel_size_m) or sample_pixel_size_m <= 0:
        raise ValueError("sample_pixel_size_m must be finite and positive.")
    if not np.isfinite(defocus_m):
        raise ValueError("defocus_m must be finite.")
    wavelength = np.float32(wavelength_m)
    sample_pixel_size = np.float32(sample_pixel_size_m)
    outer_radius_m = np.float32(90e-6)
    outermost_zone_width_m = np.float32(50e-9)
    beamstop_diameter_m = np.float32(60e-6)
    focal_length_m = (
        np.float32(2) * outer_radius_m * outermost_zone_width_m / wavelength
    )
    propagation_distance = focal_length_m + np.float32(defocus_m)
    if propagation_distance <= 0:
        raise ValueError("defocus_m must place the sample downstream of the FZP.")

    fzp_pixel_size_m = (
        wavelength * propagation_distance / (np.float32(size) * sample_pixel_size)
    )
    axis = (np.arange(size, dtype=np.float32) - size // 2) * fzp_pixel_size_m
    x_fzp, y_fzp = np.meshgrid(axis, axis)
    radius = np.sqrt(x_fzp**2 + y_fzp**2)

    lens_phase = np.exp(
        np.complex64(-1j)
        * np.float32(2 * np.pi)
        / wavelength
        * radius**2
        / (np.float32(2) * focal_length_m)
    )
    transmitting_annulus = (radius <= outer_radius_m) & (
        radius >= beamstop_diameter_m / 2
    )
    wave_at_fzp = transmitting_annulus * lens_phase

    return fresnel_propagation(
        wave_at_fzp,
        fzp_pixel_size_m,
        propagation_distance,
        wavelength,
    )
