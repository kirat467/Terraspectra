import numpy as np


def analyze_spectral_difference(healthy_cube, diseased_cube, top_k=10):
    healthy_cube = np.asarray(healthy_cube)
    diseased_cube = np.asarray(diseased_cube)

    if healthy_cube.ndim != 3 or diseased_cube.ndim != 3:
        raise ValueError("Both inputs must be 3D hyperspectral cubes")

    if healthy_cube.shape != diseased_cube.shape:
        raise ValueError("Both cubes must have the same shape")

    if not np.isfinite(healthy_cube).all():
        raise ValueError("Healthy cube contains invalid values")

    if not np.isfinite(diseased_cube).all():
        raise ValueError("Diseased cube contains invalid values")

    if not 1 <= top_k <= healthy_cube.shape[2]:
        raise ValueError("Invalid top_k value")

    healthy_mean = healthy_cube.mean(axis=(0, 1))
    diseased_mean = diseased_cube.mean(axis=(0, 1))

    difference = np.abs(healthy_mean - diseased_mean)

    top_bands = np.argsort(difference)[::-1][:top_k]

    return {
        "healthy_mean": healthy_mean,
        "diseased_mean": diseased_mean,
        "difference": difference,
        "top_bands": top_bands
    }