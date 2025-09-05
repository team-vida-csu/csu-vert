from __future__ import annotations

from pathlib import Path
from typing import  Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import glob


#---------------------------------------------
#Files are located and loaded into memory.
#---------------------------------------------

IMAGE_EXTENTIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

def get_image_files(inp: Union[str, Path]) -> List[Path]:
    """
    Given path to folder return a sorted list of image files. 
    """
    p = Path(inp)
    if p.exists() and p.is_dir():
        return sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_EXTENTIONS)
    
    if p.exists() and p.is_file():
        return [p] if p.suffix.lower() in IMAGE_EXTENTIONS else []
    
    matches = [Path(s) for s in glob.glob(str(inp), recursive=True)]
    return sorted(
        q for q in matches
        if q.suffix.lower() in IMAGE_EXTENTIONS and q.is_file()
    )

def load_image_rgb(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image as RGB float32 in [0, 1], shape (H, W, 3).
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr

def normalize_image(img: np.ndarray, mean: Optional[Sequence[float]] = None, std: Optional[Sequence[float]] = None, ) -> np.ndarray:
    """
    Normalize image in-place-ish: (img - mean) / std, where img is (H, W, 3) in [0,1].
    """
    if mean is None or std is None:
        return img
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    return (img - mean) / std

#---------------------------------------------
#Handle image titling
#---------------------------------------------

def _compute_grid(length: int, tile: int, overlap: int) -> List[int]:
    """
    Compute top-left coordinates along one axis so that tiles cover the dimension.
    Ensures the last tile aligns with the end (uses max with negative to handle small images).
    """
    if tile <= 0:
        raise ValueError("tile must be positive")
    stride = max(1, tile - overlap)
    if length <= tile:
        return [0]
    coords = list(range(0, length -tile + 1, stride))
    last = length - tile
    if coords[-1] != last:
        coords.append(last)
    return coords

def _pad_reflect(arr: np.ndarray, top: int, left: int, h: int, w: int) -> np.ndarray:
    """
    Extract a (h,w,3) patch from arr (H,W,3) allowing out-of-bounds by padding with reflect.
    top/left are the intended top-left in the original array (can be <0 or beyond bounds).
    """
    H, W, C = arr.shape
    y0, x0 = top, left
    y1, x1 = top + h, left + w

    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bottom = max(0, y1 - H)
    pad_right = max(0, x1 - W)

    cy0 = max(0, y0)
    cx0 = max(0, x0)
    cy1 = min(H, y1)
    cx1 = min(W, x1)

    patch = arr[cy0:cy1, cx0:cx1, :]

    if any(v > 0 for v in (pad_top, pad_bottom, pad_left, pad_right)):
        pad_spec = (
            (pad_top, pad_bottom),
            (pad_left, pad_right),
            (0, 0),
        )
        patch = np.pad(patch, pad_spec, mode="reflect")

    if patch.shape[0] != h or patch.shape[1] != w:
        ph, pw = patch.shape[:2]
        oh, ow = max(h, ph), max(w, pw)
        patch = np.pad(
            patch,
            ((0, oh - ph), (0, ow - pw), (0, 0)),
            mode="reflect",
        )
        patch = patch[:h, :w, :]

    return patch.astype(np.float32)


def tile_image(img: np.ndarray, tile_size: int = 512, overlap: int = 64, pad: bool = True,) -> List[Tuple[np.ndarray, int, int]]:
    """
    Produce tiles of shape (tile_size, tile_size, 3) with given overlap.

    Returns:
        List of (tile, y, x), where (y,x) is the top-left in the original image.
        If pad=True, edge tiles are padded using reflect to reach full tile size.
        If pad=False, only full in-bounds tiles are returned (images smaller than
        tile_size will yield an empty list).
    """
    H, W, C = img.shape
    if C != 3:
        raise ValueError("Expected 3-channel RGB image")

    ys = _compute_grid(H, tile_size, overlap)
    xs = _compute_grid(W, tile_size, overlap)

    tiles: List[Tuple[np.ndarray, int, int]] = []
    for y in ys:
        for x in xs:
            if pad:
                tile = _pad_reflect(img, y, x, tile_size, tile_size)
                tiles.append((tile, y, x))
            else:
                if y + tile_size <= H and x + tile_size <= W:
                    tile = img[y : y + tile_size, x : x + tile_size, :]
                    tiles.append((tile.copy(), y, x))
    return tiles

def iter_tiles(img: np.ndarray, tile_size: int = 512, overlap: int = 64, pad: bool = True,) -> Iterator[Tuple[np.ndarray, int, int]]:
    """
    Generator version of tile_image().
    """
    for tile, y, x in tile_image(img, tile_size, overlap, pad):
        yield tile, y, x