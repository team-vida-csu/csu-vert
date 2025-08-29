# tests/test_infer.py
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from vert.infer import run_folder


# -----------------------
# Helpers
# -----------------------

def _save_rgb(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)

def make_dummy_torchscript(path: Path, in_ch: int = 3, out_ch: int = 2) -> None:
    """
    Create a tiny TorchScript model that maps 3->2 channels with a 1x1 conv.
    Deterministic weights for stable tests.
    """
    import torch  # import locally so torch isn't required at import time for this file

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
            torch.manual_seed(0)
            with torch.no_grad():
                self.conv.weight.copy_(torch.randn(out_ch, in_ch, 1, 1) * 0.01)
                self.conv.bias.zero_()

        def forward(self, x):
            return self.conv(x)

    m = Tiny().eval()
    scripted = torch.jit.script(m)
    scripted.save(str(path))


def make_dummy_onnx(ts_weights: Path, onnx_path: Path, h: int = 32, width: int = 32) -> None:
    """
    Export a tiny eager model to ONNX (dynamic H/W).
    We intentionally avoid wrapping a ScriptModule to keep tracing simple.
    """
    import torch  # local import

    class Tiny(torch.nn.Module):
        def __init__(self, in_ch=3, out_ch=2):
            super().__init__()
            self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
            torch.manual_seed(0)
            with torch.no_grad():
                self.conv.weight.copy_(torch.randn(out_ch, in_ch, 1, 1) * 0.01)
                self.conv.bias.zero_()

        def forward(self, x):
            return self.conv(x)

    model = Tiny().eval()

    dummy = torch.randn(1, 3, h, width)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "N", 2: "H", 3: "W"},
            "logits": {0: "N", 2: "H", 3: "W"},
        },
    )



# -----------------------
# Tests (TorchScript)
# -----------------------

def test_run_single_file_torchscript(tmp_path: Path):
    w = tmp_path / "dummy.pt"
    make_dummy_torchscript(w)

    img = (np.random.rand(37, 41, 3) * 255).astype("uint8")  # odd dims to hit padding/tiling
    img_path = tmp_path / "in.png"
    _save_rgb(img_path, img)

    outdir = tmp_path / "out"
    run_folder(
        input=str(img_path),
        output=str(outdir),
        weights=str(w),
        device="auto",
        tile_size=32,
        overlap=8,
        batch_size=2,
        amp=False,
    )

    outs = list(outdir.glob("*.png"))
    assert len(outs) == 1
    out_im = Image.open(outs[0])
    assert out_im.size == (img.shape[1], img.shape[0])  # (W,H)
    assert out_im.mode == "P"


def test_run_glob_torchscript(tmp_path: Path):
    w = tmp_path / "dummy.pt"
    make_dummy_torchscript(w)

    a = (np.random.rand(48, 48, 3) * 255).astype("uint8")
    b = (np.random.rand(64, 40, 3) * 255).astype("uint8")
    d = tmp_path / "imgs"
    d.mkdir()
    _save_rgb(d / "a.png", a)
    _save_rgb(d / "b.png", b)

    outdir = tmp_path / "out"
    run_folder(
        input=str(d / "*.png"),
        output=str(outdir),
        weights=str(w),
        device="cpu",
        tile_size=32,
        overlap=0,
        batch_size=4,
        amp=False,
    )

    outs = sorted(outdir.glob("*.png"))
    assert [p.stem for p in outs] == ["a", "b"]
    # verify sizes
    sizes = [Image.open(p).size for p in outs]
    assert sizes[0] == (48, 48)
    assert sizes[1] == (40, 64)


def test_run_directory_torchscript_with_palette(tmp_path: Path):
    w = tmp_path / "dummy.pt"
    make_dummy_torchscript(w)

    root = tmp_path / "data"
    sub = root / "sub"
    sub.mkdir(parents=True)
    _save_rgb(root / "x.jpg", (np.random.rand(30, 30, 3) * 255).astype("uint8"))
    _save_rgb(sub / "y.png", (np.random.rand(35, 22, 3) * 255).astype("uint8"))

    palette = [(0, 0, 0), (255, 0, 0)]  # two classes

    outdir = tmp_path / "pred"
    run_folder(
        input=str(root),
        output=str(outdir),
        weights=str(w),
        device="cpu",
        tile_size=16,
        overlap=4,
        batch_size=2,
        amp=False,
        palette=palette,
    )

    outs = sorted(outdir.glob("*.png"))
    assert len(outs) == 2
    for p in outs:
        im = Image.open(p)
        assert im.mode == "P"
        # Palette applied (at least first 2 colors present)
        pal = im.getpalette()
        assert pal is not None and len(pal) >= 6  # 2*3 RGB entries


# -----------------------
# Optional ONNX tests
# -----------------------

def test_run_folder_onnx(tmp_path: Path):
    onnxruntime = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

    ts = tmp_path / "dummy.pt"
    make_dummy_torchscript(ts)
    onnx_path = tmp_path / "dummy.onnx"
    make_dummy_onnx(ts, onnx_path, h=32, width=32)

    img = (np.random.rand(45, 50, 3) * 255).astype("uint8")
    img_path = tmp_path / "im.png"
    _save_rgb(img_path, img)

    outdir = tmp_path / "out"
    run_folder(
        input=str(img_path),
        output=str(outdir),
        weights=str(onnx_path),
        device="cpu",
        tile_size=16,
        overlap=4,
        batch_size=3,
        amp=False,
    )

    outs = list(outdir.glob("*.png"))
    assert len(outs) == 1
    im = Image.open(outs[0])
    assert im.size == (img.shape[1], img.shape[0])
