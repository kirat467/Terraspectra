import numpy as np


def spatial_block_split(coords, patch_size, block_size_patches=8,
                        ratios=(0.7, 0.15, 0.15), seed=42):
    """Patch ko uske spatial block ke hisaab se train/val/test me daalo.
    Isse neighbouring patches alag-alag set me nahi jaate (data leakage nahi)."""
    block_px = patch_size * block_size_patches
    block_ids = [(r // block_px, c // block_px) for r, c in coords]
    uniq = sorted(set(block_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr, n_va = int(n * ratios[0]), int(n * ratios[1])
    assign = {}
    for k, b in enumerate(uniq):
        assign[b] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")
    split = np.array([assign[b] for b in block_ids])
    return {s: np.where(split == s)[0] for s in ("train", "val", "test")}
