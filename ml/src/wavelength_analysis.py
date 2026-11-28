import numpy as np


def band_to_wavelength(band_indices, total_bands=550,
                       start_nm=400, end_nm=950):

    band_indices = np.asarray(band_indices)

    if isinstance(total_bands, (bool, np.bool_)) or not isinstance(
        total_bands, (int, np.integer)
    ):
        raise ValueError("Total bands must be an integer")

    if total_bands <= 1:
        raise ValueError("Total bands must be greater than 1")

    if not np.isscalar(start_nm) or not np.isscalar(end_nm):
        raise ValueError("Wavelength limits must be numeric scalars")

    if isinstance(start_nm, (bool, np.bool_)) or isinstance(
        end_nm, (bool, np.bool_)
    ):
        raise ValueError("Wavelength limits must be numeric")

    if not np.isreal(start_nm) or not np.isreal(end_nm):
        raise ValueError("Wavelength limits must be real numbers")

    if not np.isfinite(start_nm) or not np.isfinite(end_nm):
        raise ValueError("Wavelength limits must be finite")

    if start_nm >= end_nm:
        raise ValueError("Invalid wavelength range")

    if band_indices.size == 0:
        raise ValueError("Band indices cannot be empty")

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

    if not np.issubdtype(difference.dtype, np.number):
        raise ValueError("Difference must contain numeric values")

    if np.iscomplexobj(difference):
        raise ValueError("Complex differences are not supported")

    if not np.isfinite(difference).all():
        raise ValueError("Difference contains invalid values")

    if isinstance(top_k, (bool, np.bool_)) or not isinstance(
        top_k, (int, np.integer)
    ):
        raise ValueError("top_k must be an integer")

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