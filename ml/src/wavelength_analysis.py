import numpy as np


def band_to_wavelength(band_indices, total_bands=550,
                       start_nm=400, end_nm=950):

    band_indices = np.asarray(band_indices)

    if total_bands <= 1:
        raise ValueError("Total bands must be greater than 1")

    if start_nm >= end_nm:
        raise ValueError("Invalid wavelength range")

    if not np.issubdtype(band_indices.dtype, np.integer):
        raise ValueError("Band indices must be integers")

    if np.any(band_indices < 0) or np.any(band_indices >= total_bands):
        raise ValueError("Band index out of range")

    wavelengths = np.linspace(
        start_nm, end_nm, total_bands
    )

    return wavelengths[band_indices]


def rank_wavelength_differences(difference, top_k=10,
                                start_nm=400, end_nm=950):

    difference = np.asarray(difference)

    if difference.ndim != 1 or difference.size == 0:
        raise ValueError("Difference must be a non-empty 1D array")

    if not np.isfinite(difference).all():
        raise ValueError("Difference contains invalid values")

    if not 1 <= top_k <= len(difference):
        raise ValueError("Invalid top_k value")

    top_bands = np.argsort(difference)[::-1][:top_k]

    wavelengths = band_to_wavelength(
        top_bands,
        total_bands=len(difference),
        start_nm=start_nm,
        end_nm=end_nm
    )

    return {
        "top_bands": top_bands,
        "wavelengths_nm": wavelengths,
        "differences": difference[top_bands]
    }