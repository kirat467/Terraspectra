import numpy as np


def make_patches(cube, labels=None, size=32, stride=32, max_nan_fraction=0.2):
    """cube (B,H,W), labels (H,W) optional.
    Returns X (N,B,size,size), y (N,) majority label, coords (N,2) top-left (row,col)."""
    _, H, W = cube.shape
    X, y, coords = [], [], []
    for i in range(0, H - size + 1, stride):
        for j in range(0, W - size + 1, stride):
            p = cube[:, i:i + size, j:j + size]
            if np.isnan(p).mean() > max_nan_fraction:
                continue
            if labels is not None:
                lab = labels[i:i + size, j:j + size]
                valid = lab[lab >= 0]
                if valid.size == 0:
                    continue
                y.append(np.bincount(valid.astype(int)).argmax())
            X.append(np.nan_to_num(p, nan=0.0))
            coords.append((i, j))
    X = np.stack(X) if X else np.empty((0, cube.shape[0], size, size))
    return X, (np.array(y) if labels is not None else None), np.array(coords)
