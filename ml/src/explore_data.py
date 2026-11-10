
import numpy as np


def explore_cube(cube):

    cube = np.asarray(cube)

    if cube.ndim != 3:
        raise ValueError("Image must have 3 dimensions: H, W, Bands")

    if cube.size == 0:
        raise ValueError("Image cube is empty")

    if not np.issubdtype(cube.dtype, np.number):
        raise ValueError("Image must contain numeric values")

    height, width, bands = cube.shape

    missing = np.isnan(cube).sum()
    infinite = np.isinf(cube).sum()
    valid = cube[np.isfinite(cube)]

    if valid.size == 0:
        raise ValueError("Image contains no valid pixel values")

    report = {
        "height": height,
        "width": width,
        "bands": bands,
        "dtype": str(cube.dtype),
        "total_values": int(cube.size),
        "missing_values": int(missing),
        "infinite_values": int(infinite),
        "minimum": float(valid.min()),
        "maximum": float(valid.max()),
        "mean": float(valid.mean()),
        "standard_deviation": float(valid.std())
    }

    print("\n--- TerraSpectra Dataset Report ---")

    for key, value in report.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    return report
