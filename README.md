# csu-vert 🌿
### (Convolutional Segmentation for Understory VEgetation Recognition and Typing)

This project provides a lightweight CLI (`csu-vert`) to run semantic segmentation on images using trained UNet models (TorchScript or ONNX). 
It supports tiling with overlap, Gaussian blending, palette-based mask outputs, and per-image vegetation statistics exported to CSV.

## Installation

> ⚠️ PyTorch wheels are platform-specific. By default `csu-vert` installs without `torch` to keep it portable.  
> Choose an install path depending on your platform and whether you want Torch or ONNX runtime.

### Core install (no inference backend)

```bash
# in folder with downloaded source code.
pip install -e ".[onnx]"
```
## Usage
### Basic example
~~~
csu-vert \
  --input path/to/images/*.png \
  --output out_masks/ \
  --weights unet.onnx \
  --device auto
~~~
- **--input** can be a folder, single file, or glob pattern
- **--weights** can be TorchScript (.pt) or ONNX (.onnx)
- **--device** can be auto, cpu, cuda, or mps
### With vegetation stats (CSV)
~~~
csu-vert \
  --input images/ \
  --output preds/ \
  --weights unet.onnx \
  --csv stats.csv \
  --class-names background,forb,graminoid,woody \
  --suppress-noise \
  --min-class-percent 0.5
~~~
This will:
- Save indexed masks to **preds/**
- Write a CSV with per-class pixel counts and percentages for each image
- Suppress classes that cover ≤0.5% of pixels, remapping them to background

## Options

| Flag                  | Description                                                      |
| --------------------- | ---------------------------------------------------------------- |
| `-i, --input`         | Input folder, file, or glob                                      |
| `-o, --output`        | Output folder for masks                                          |
| `-w, --weights`       | Path to `.pt` or `.onnx` model                                   |
| `--device`            | `auto`, `cpu`, `cuda`, or `mps`                                  |
| `--tile-size`         | Size of image tiles (default 512)                                |
| `--overlap`           | Overlap between tiles (default 64)                               |
| `--batch-size`        | Number of tiles per forward pass                                 |
| `--mean` / `--std`    | Normalization values (default: ImageNet)                         |
| `--csv`               | Path to CSV for per-image stats                                  |
| `--class-names`       | Comma-separated names (default: background,forb,graminoid,woody) |
| `--suppress-noise`    | If set, remap tiny classes to background                         |
| `--min-class-percent` | Threshold % for noise suppression                                |
| `--ext`               | Output extension: `png` (default), `jpg`, `tif`                  |

## Development
### Clone & install locally
~~~
git clone https://github.com/USERNAME/csu-vert.git
cd csu-vert
pip install -e ".[dev,all]"
~~~
### Run tests
~~~
pytest
~~~
## License
This project is licensed under CC0
