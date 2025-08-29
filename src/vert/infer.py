

import glob
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from vert.io import (
    get_image_files,
    load_image_rgb,
    normalize_image,
    tile_image,
)

def has_torch():
    """Return the torch module if available, otherwise None."""
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None
    

def require_torch(why: str = "TorchScript (.pt) inference"):
    """Raise a friendly error if torch isn't available."""
    torch = has_torch()
    if torch is None:
        raise RuntimeError(
            f"PyTorch is required for {why} but isn't installed.\n"
            "Install it for your platform:\n"
            "  • Apple Silicon/macOS:   pip install torch torchvision torchaudio\n"
            "  • Linux/Windows (CPU):   pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cpu\n"
            "Then re-run your command."
        )
    return torch

def _resolve_device(requested: str, prefer_torch: bool) -> str:
    """
    Resolve 'auto' to a specific device.
    If prefer_torch=False (e.g., ONNX path), don't force importing torch.
    """
    if requested != "auto":
        return requested

    if not prefer_torch:
        # ONNX path: we can just default to CPU without needing torch
        return "cpu"

    torch = has_torch()
    if torch is None:
        return "cpu"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
# -----------------------
# Entry point
# -----------------------

def run_folder(
    input: str,
    output: str,
    weights: str,
    device: str = "auto",
    tile_size: int = 512,
    overlap: int = 64,
    batch_size: int = 4,
    amp: bool = False,
    mean: Optional[Tuple[float, float, float]] = (0.485, 0.456, 0.406),
    std: Optional[Tuple[float, float, float]] = (0.229, 0.224, 0.225),
    palette: Optional[List[Tuple[int, int, int]]] = None,
    ext: str = "png",
) -> None:
    """
    Run UNet inference over a folder/single-file/glob and save masks.

    Parameters
    ----------
    input : str
        Folder, single image path, or glob (absolute or relative).
    output : str
        Output folder for predicted masks.
    weights : str
        Path to TorchScript .pt or ONNX .onnx file.
    device :  {"auto","cpu","cuda","mps"}
    tile_size : int
    overlap : int
    batch_size : int
    amp : bool
        Mixed precision (CUDA only).
    mean, std : Optional[tuple]
        If provided, images are normalized (x-mean)/std with inputs in [0,1].
    palette : Optional[List[RGB]]
        Optional color palette for saved indexed masks.
    ext : str
        Output extension: "png" (default) or "tif".
    """
    files = get_image_files(input)
    if not files:
        raise FileNotFoundError(f"No input images found for: {input}")

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    wpath = Path(weights)
    if not wpath.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    is_onnx = wpath.suffix.lower() == ".onnx"

    device = _resolve_device(device, prefer_torch=not is_onnx)

    model = _load_model(wpath, device, is_onnx)



    for f in tqdm(files, desc="Inferring"):
        img = load_image_rgb(f)  # (H,W,3), float32 in [0,1]
        if mean is not None and std is not None:
            img = normalize_image(img, mean=mean, std=std)

        logits = _predict_image(
            img_np=img,
            model=model,
            is_onnx=is_onnx,
            device=device,
            tile_size=tile_size,
            overlap=overlap,
            batch_size=batch_size,
            amp=amp,
        )  # (C,H,W) float32

        print(
            f"{Path(f).name}: C={logits.shape[0]}, logits min/max="
            f"{float(logits.min()):.3f}/{float(logits.max()):.3f}"
        )
        unique, counts = np.unique(np.argmax(logits, axis=0), return_counts=True)
        print("pred histogram:", dict(zip(unique.tolist(), counts.tolist())))

        palette = [(0,0,0), (255,0,0), (0,255,0), (0,0,255)] 
        mask_idx = np.argmax(logits, axis=0).astype(np.uint8)  # (H,W)
        _save_indexed_mask(
            mask_idx,
            out_path=out_dir / f"{Path(f).stem}.{ext}",
            palette=palette,
        )

# -----------------------
# Core prediction
# -----------------------

def _load_model(wpath: Path, device: str, is_onnx: bool):
    if is_onnx:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        # Note: on mac Apple Silicon, ORT GPU is separate from torch; CPU provider works broadly.
        return ort.InferenceSession(str(wpath), providers=providers)

    # TorchScript path: require torch at runtime, not import-time
    torch = require_torch("loading a TorchScript (.pt) model")
    m = torch.jit.load(str(wpath), map_location=device)
    m.eval()
    return m


def _predict_image(img_np: np.ndarray, model, is_onnx: bool, device: str, tile_size: int, overlap: int, batch_size: int, amp: bool,) -> np.ndarray:
    """
    Tile with overlap, run model, and stitch with Gaussian blending.
    Returns blended logits as (C,H,W) float32.
    """
    H, W, C3 = img_np.shape
    if C3 != 3:
        raise ValueError("Expected 3-channel RGB inputs.")

    tiles = tile_image(img_np, tile_size=tile_size, overlap=overlap, pad=True) 

    first_tile = tiles[0][0]
    C = _forward_batch([first_tile], model, is_onnx, device, amp)[0].shape[0]

    acc = np.zeros((C, H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)
    wmask = _gaussian_weight(tile_size, tile_size)

    batch_tiles: List[np.ndarray] = []
    batch_coords: List[Tuple[int, int]] = []

    for tile, y, x in tiles:
        batch_tiles.append(tile)
        batch_coords.append((y, x))
        if len(batch_tiles) == batch_size:
            _accumulate_batch(batch_tiles, batch_coords, model, is_onnx, device, amp, wmask, acc, wsum, tile_size)
            batch_tiles.clear()
            batch_coords.clear()

    if batch_tiles:
        _accumulate_batch(batch_tiles, batch_coords, model, is_onnx, device, amp, wmask, acc, wsum, tile_size)

    np.maximum(wsum, 1e-8, out=wsum)
    acc /= wsum[None, :, :]
    return acc

def _accumulate_batch(
    batch_tiles: List[np.ndarray],
    batch_coords: List[Tuple[int, int]],
    model,
    is_onnx: bool,
    device: str,
    amp: bool,
    wmask: np.ndarray,              # (tile,tile)
    acc: np.ndarray,                # (C,H,W)
    wsum: np.ndarray,               # (H,W)
    tile_size: int,) -> None:
    """
    Run a batch forward and blend into accumulator.
    """
    logits_b = _forward_batch(batch_tiles, model, is_onnx, device, amp)  # (B,C,tile,tile)

    for (y, x), lg in zip(batch_coords, logits_b):
        c, h, w = lg.shape
        hh = min(tile_size, acc.shape[1] - y)
        ww = min(tile_size, acc.shape[2] - x)
        acc[:, y:y+hh, x:x+ww] += lg[:, :hh, :ww] * wmask[:hh, :ww]
        wsum[y:y+hh, x:x+ww] += wmask[:hh, :ww]

def _forward_batch(tiles: List[np.ndarray],
    model,
    is_onnx: bool,
    device: str,
    amp: bool,) -> np.ndarray:
    """
    tiles: list of (tile,tile,3) float32 in whatever normalized scale you used.
    Returns (B,C,tile,tile) float32 logits.
    """
    batch = np.stack(tiles, axis=0).transpose(0, 3, 1, 2).astype(np.float32)

    if is_onnx:
        inputs = {"image": batch}
        outputs = model.run(None, inputs)[0] 
        return outputs.astype(np.float32)

     # TorchScript path (lazy import)
    torch = require_torch("TorchScript forward pass")
    x = torch.from_numpy(batch).to(device)


    x = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        if amp and device == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                y = model(x)
        else:
            y = model(x)
    return y.detach().cpu().numpy().astype(np.float32)

# -----------------------
# Utilities
# -----------------------

def _gaussian_weight(h: int, w: int, sigma_frac: float = 0.125) -> np.ndarray:
    """
    2D Gaussian weight mask (peaks center). sigma = size * sigma_frac.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    sigma_y = max(1.0, h * sigma_frac)
    sigma_x = max(1.0, w * sigma_frac)
    wy = np.exp(-0.5 * ((yy - cy) / sigma_y) ** 2)
    wx = np.exp(-0.5 * ((xx - cx) / sigma_x) ** 2)
    w2d = wy * wx
    w2d /= w2d.max()
    return w2d.astype(np.float32)

def _save_indexed_mask(mask_idx: np.ndarray, out_path: Path, palette: Optional[List[Tuple[int, int, int]]]) -> None:
    """
    Save a (H,W) uint8 index image as PNG/TIFF with optional palette.
    """
    out = Image.fromarray(mask_idx, mode="P")
    if palette:
        flat = [v for rgb in palette for v in rgb]
        flat += [0] * max(0, 256 * 3 - len(flat))
        out.putpalette(flat, rawmode="RGB")
    out.save(out_path)