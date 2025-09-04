# csu-vert 🌿
### (Convolutional Segmentation for Understory VEgetation Recognition and Typing)

This project provides a lightweight CLI (`csu-vert`) to run semantic segmentation on images using trained UNet models (TorchScript or ONNX). 
It supports tiling with overlap, Gaussian blending, palette-based mask outputs, and per-image vegetation statistics exported to CSV.

![segmented image](https://github.com/team-vida-csu/csu-vert/blob/main/src/vert/images/githubdemo_photo.png)

## Installation

> ⚠️ PyTorch wheels are platform-specific. By default `csu-vert` installs without `torch` to keep it portable.  
> Choose an install path depending on your platform and whether you want Torch or ONNX runtime.

### Core install (no inference backend)

```bash
# Latest main
pip install "csu-vert[onnx] @ git+https://github.com/team-vida-csu/csu-vert.git"

# Specific tag
pip install "csu-vert[torch] @ git+https://github.com/team-vida-csu/csu-vert.git@v0.1.0"
```
### If you want to run TorchScript (.pt) models instead of ONNX, install with torch:
~~~
pip install torch torchvision torchaudio
~~~
## Usage
The CLI has subcommands:
- **infer** -> run inference on a folder / file / glob
- **list-models** -> list registered downladable models
- **precache** -> download & cache weights/config without running inference
### Run inference with auto-downloaded weights
~~~
csu-vert infer \
  --input images/*.png \
  --output preds/ \
  --weights auto:unet-r34-4c@v0.0.2 --format onnx
~~~
This will:
- Download the weights + config file if missing (cached under **~/.cache/vert/…**)
- Run inference using the Onnx model
- Save masks to **preds/**
### Run inference with local weights
~~~
csu-vert infer \
  --input images/ \
  --output preds/ \
  --weights ./models/vert-unet_resnet34_4cls_scripted.pt
~~~
### With vegetation stats (CSV)
~~~
csu-vert infer \
  --input images/ \
  --output preds/ \
  --weights auto:unet-r34-4c@v0.0.2 \
  --csv stats.csv \
  --suppress-noise \
  --min-class-percent 0.5
~~~
This will:
- Save indexed masks to **preds/**
- Write a CSV with per-class pixel counts and percentages for each image
- Suppress classes that cover ≤0.5% of pixels, remapping them to background
### List available models
~~~
csu-vert list-models
~~~
### Pre-cache weights before inference
~~~
csu-vert precache \
  --weights auto:unet-r34-4c@v0.0.2 --format onnx
~~~

## Options
### Inference flags

| Flag                  | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| `-i, --input`         | Input folder, file, or glob                                    |
| `-o, --output`        | Output folder for masks                                        |
| `-w, --weights`       | Local path to `.pt`/`.onnx` or remote spec (`auto:…` / `gh:…`) |
| `--format`            | Preferred format when using `auto:` (`pt` or `onnx`)           |
| `--device`            | `auto`, `cpu`, `cuda`, or `mps`                                |
| `--tile-size`         | Size of image tiles (default 512)                              |
| `--overlap`           | Overlap between tiles (default 64)                             |
| `--batch-size`        | Number of tiles per forward pass                               |
| `--amp`               | Enable mixed precision (CUDA/MPS only)                         |
| `--mean` / `--std`    | Normalization values (default: ImageNet or from YAML)          |
| `--csv`               | Path to CSV for per-image stats                                |
| `--class-names`       | Comma-separated names (default or from YAML)                   |
| `--suppress-noise`    | Remap small classes to background                              |
| `--min-class-percent` | Threshold % for noise suppression                              |
| `--ext`               | Output extension: `png` (default), `jpg`, `tif`                |
| `--side-by-side`      | Save original + mask comparison with legend                    |
| `--overlay`           | Save overlay image (mask blended over original)                |
| `--overlay-alpha`     | Alpha for overlay (0–255, default 112)                         |


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
