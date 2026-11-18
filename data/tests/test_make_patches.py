import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "ml" / "src"))
pytest.importorskip("h5py")
pytest.importorskip("sklearn")
from make_patches import assign_blocks, build, cut_patches, summarise, to_cnn_layout  # noqa: E402
from make_scene import build_scenes, write_hdf5  # noqa: E402

FARM = ROOT / "frontend" / "public" / "data"
BLOCK = 18


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("p") / "s.h5"
    meta, geo, owner, sc = build_scenes(FARM)
    write_hdf5(path, sc, meta)
    return build(path), sc


def test_every_split_has_patches_and_diseased_ones(built):
    out, _ = built
    s = summarise(out)
    for name in ("train", "val", "test"):
        assert s["splits"][name]["patches"] > 100
        assert s["splits"][name]["early_stress"] + s["splits"][name]["visible_stress"] > 0
    assert s["patch_shape"] == [9, 9, 20] and out["X"].dtype == np.float32 and np.isfinite(out["X"]).all()


def test_no_pixel_is_shared_between_splits(built):
    out, _ = built
    for date in range(6):
        owner = np.full((135, 135), -1, dtype=int)
        for (r, c), sp, di in zip(out["pos"], out["split"], out["date_index"]):
            if di != date:
                continue
            region = owner[r - 4:r + 5, c - 4:c + 5]
            assert ((region == -1) | (region == sp)).all()        # a pixel only ever belongs to one split
            owner[r - 4:r + 5, c - 4:c + 5] = sp


def test_same_place_is_in_the_same_split_on_every_date(built):
    out, _ = built
    seen = {}
    for pos, sp in zip(map(tuple, out["pos"]), out["split"]):
        assert seen.setdefault(pos, sp) == sp


def test_labels_match_the_centre_pixel_and_skip_non_crop(built):
    out, sc = built
    for k in np.random.default_rng(0).choice(len(out["y"]), 200, replace=False):
        r, c = out["pos"][k]
        assert out["y"][k] == sc[out["date_index"][k]]["label"][r, c] >= 0


def test_pca_uses_train_pixels_only(built):
    out, sc = built
    blocks = out["blocks"]
    train_mask = np.kron(blocks == 0, np.ones((BLOCK, BLOCK), bool))[:135, :135]
    mean = np.concatenate([s["cube"][train_mask] for s in sc]).mean(axis=0)
    assert np.allclose(out["pca"].mean_, mean, atol=1e-5)
    all_mean = np.concatenate([s["cube"].reshape(-1, 200) for s in sc]).mean(axis=0)
    assert not np.allclose(out["pca"].mean_, all_mean, atol=1e-6)


def test_block_assignment_is_reproducible_and_spreads_disease():
    labels = np.full((2, 54, 54), -1, dtype=np.int8)
    labels[:, :, :] = 0
    labels[:, 0:18, 0:18] = 2
    labels[:, 18:36, 18:36] = 1
    labels[:, 36:54, 36:54] = 1
    a, b = assign_blocks(labels, 18), assign_blocks(labels, 18)
    assert (a == b).all()
    assert {a[0, 0], a[1, 1], a[2, 2]} == {0, 1, 2}


def test_cut_patches_skips_tracks_and_returns_cnn_layout():
    cube = np.random.default_rng(0).random((36, 36, 4)).astype(np.float32)
    label = np.zeros((36, 36), dtype=np.int8)
    label[:, 10:14] = -1
    blocks = np.zeros((2, 2), dtype=int)
    X, y, sp, pos = cut_patches(cube, label, blocks, block=18, patch=9, stride=1, margin=2)
    assert len(X) == len(y) == len(sp) == len(pos) > 0 and X.shape[1:] == (9, 9, 4)
    assert (y >= 0).all()
    assert to_cnn_layout(X).shape == (len(X), 1, 4, 9, 9)
