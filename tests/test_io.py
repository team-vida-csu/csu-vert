import numpy as np
from pathlib import Path
from PIL import Image

import pytest


from vert.io import (
    get_image_files,
    load_image_rgb,
    tile_image,
    normalize_image,
    iter_tiles,
)

def _save_rgb(path: Path, arr: np.ndarray) -> None:
    """Save an HxWx3 uint8 array to disk as an RGB image."""
    Image.fromarray(arr).save(path)



def test_get_image_files_folder_and_glob(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    img1 = (np.random.rand(8, 8, 3) * 255).astype("uint8")
    img2 = (np.random.rand(8, 8, 3) * 255).astype("uint8")
    _save_rgb(tmp_path / "a" / "x.png", img1)
    _save_rgb(tmp_path / "b" / "y.jpg", img2)
    (tmp_path / "notes.txt").write_text("not an image")

    files_by_dir = get_image_files(tmp_path)
    assert len(files_by_dir) == 2
    files_by_glob = get_image_files(str(tmp_path / "a" / "*.png"))
    assert len(files_by_glob) == 1
    assert files_by_glob[0].name == "x.png"

def test_load_image_rgb_returns_float01(tmp_path: Path):
    img = (np.random.rand(10, 12, 3) * 255).astype("uint8")
    p = tmp_path / "im.png"
    _save_rgb(p, img)

    arr = load_image_rgb(p)
    assert arr.shape == (10, 12, 3)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0

def test_normalize_image_noop_and_with_stats():
    img = np.full((4, 5, 3), 0.5, dtype=np.float32)
    # No mean/std -> identical object (or at least identical values)
    out = normalize_image(img)
    np.testing.assert_allclose(out, img)

    # With mean/std -> (0.5-0.5)/0.25 = 0
    out2 = normalize_image(img, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
    np.testing.assert_allclose(out2, np.zeros_like(img))



# ---------- Tiling ----------

def test_tile_image_non_overlapping_no_pad(tmp_path: Path):
    img = (np.random.rand(16, 16, 3)).astype("float32")
    tiles = tile_image(img, tile_size=8, overlap=0, pad=False)
    assert len(tiles) == 4
    for t, y, x in tiles:
        assert t.shape == (8, 8, 3)
        assert y in (0, 8) and x in (0, 8)

def test_tile_image_with_overlap_and_pad_edges():
    img = (np.random.rand(20, 20, 3)).astype("float32")
    tiles = tile_image(img, tile_size=16, overlap=4, pad=True)
    assert len(tiles) == 4
    for t, y, x in tiles:
        assert t.shape == (16, 16, 3)
        assert y in (0, 4) and x in (0, 4)
        assert np.isfinite(t).all()

def test_tile_image_small_image_with_pad():
    img = (np.random.rand(10, 10, 3)).astype("float32")
    tiles = tile_image(img, tile_size=32, overlap=8, pad=True)
    assert len(tiles) == 1
    tile, y, x = tiles[0]
    assert tile.shape == (32, 32, 3)
    assert y == 0 and x == 0

def test_tile_image_small_image_no_pad():
    img = (np.random.rand(10, 10, 3)).astype("float32")
    tiles = tile_image(img, tile_size=32, overlap=8, pad=False)
    assert tiles == []

def test_iter_tiles_matches_tile_image():
    img = (np.random.rand(33, 35, 3)).astype("float32")
    a = tile_image(img, tile_size=16, overlap=4, pad=True)
    b = list(iter_tiles(img, tile_size=16, overlap=4, pad=True))
    assert len(a) == len(b)
    for (ta, ya, xa), (tb, yb, xb) in zip(a, b):
        assert ya == yb and xa == xb
        np.testing.assert_allclose(ta, tb)

# ---------- Edge / error cases ----------

def test_tile_image_rejects_non_rgb():
    gray = np.random.rand(8, 8, 1).astype("float32")
    with pytest.raises(ValueError):
        tile_image(gray, 8, 0, pad=True)

def test_tile_image_rejects_bad_tile():
    img = np.random.rand(8, 8, 3).astype("float32")
    with pytest.raises(ValueError):
        tile_image(img, tile_size=0, overlap=0, pad=True)