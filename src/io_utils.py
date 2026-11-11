import numpy as np
import rasterio


def load_cube(path):
    """Hyperspectral GeoTIFF load karo. Returns cube (bands,H,W), profile, wavelengths(nm) ya None."""
    with rasterio.open(path) as src:
        cube = src.read().astype("float32")
        profile = src.profile
        nodata = src.nodata
        wavelengths = None
        # Kai sensors wavelength tags me rakhte hain
        tags = src.tags()
        for key in ("wavelength", "WAVELENGTH", "wavelengths"):
            if key in tags:
                try:
                    wavelengths = np.array(
                        [float(x) for x in tags[key].strip("{}[]").split(",")]
                    )
                except ValueError:
                    pass
    if nodata is not None:
        cube[cube == nodata] = np.nan
    return cube, profile, wavelengths


def save_cube(path, cube, profile):
    profile = profile.copy()
    profile.update(count=cube.shape[0], dtype="float32", nodata=np.nan)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(cube.astype("float32"))


def describe(cube, wavelengths=None):
    print("Shape (bands,H,W):", cube.shape)
    print("NaN %:", round(float(np.isnan(cube).mean() * 100), 2))
    print("Min/Max:", np.nanmin(cube), np.nanmax(cube))
    if wavelengths is not None:
        print("Wavelength range:", wavelengths.min(), "-", wavelengths.max(), "nm")
