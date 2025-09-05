import argparse
import sys
from typing import Optional, List

from . import infer
from .weights import resolve_weights, list_models, precache  # <- wire weights.py

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="csu-vert", description="UNet inference CLI")

    sub = p.add_subparsers(dest="cmd")

    # infer (default) ---------------------------------------------------------
    a = sub.add_parser("infer", help="Run inference")

    a.add_argument("--input", "-i", required=True,
        help="Path to an input folder, single file, or glob (e.g. images/*.png)")

    a.add_argument("--output", "-o", required=True,
        help="Folder where output masks will be saved")

    a.add_argument("--weights", "-w", required=True,
        help="Local path to .pt/.onnx OR spec (auto:<id>@<ver> | gh:<owner>/<repo>@<tag>/<file>)")

    a.add_argument("--format", choices=["auto", "pt", "onnx"], default="auto",
        help="Preferred format when using auto:<id>@<ver> (ignored for local paths)")

    a.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for inference (default: auto-detect)")

    a.add_argument("--tile-size", type=int, default=512, help="Tile size (pixels)")
    a.add_argument("--overlap", type=int, default=64, help="Tile overlap (pixels)")
    a.add_argument("--batch-size", type=int, default=4, help="Tiles per batch")
    a.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA/MPS")

    a.add_argument("--ext", default="png", choices=["png", "jpg", "tif", "tiff"],
        help="Output mask file extension")

    a.add_argument("--mean", nargs=3, type=float, default=[0.485, 0.456, 0.406],
        help="Normalization mean override (e.g. 0.485 0.456 0.406). If omitted, uses YAML if available.")

    a.add_argument("--std", nargs=3, type=float, default=[0.229, 0.224, 0.225],
        help="Normalization std override (e.g. 0.229 0.224 0.225). If omitted, uses YAML if available.")

    a.add_argument("--csv", dest="csv_path", default=None,
        help="If set, write per-image stats to this CSV file")

    a.add_argument("--min-class-percent", type=float, default=0.0,
        help="Treat classes with <= this percent area as noise (e.g., 0.5 for 0.5%%)")

    a.add_argument("--suppress-noise", action="store_true",
        help="If set, classes under --min-class-percent are remapped to background (0)")

    a.add_argument("--class-names", default="background,forb,graminoid,woody",
        help='Comma-separated class names (e.g. "background,forb,graminoid,woody"). '
             "If omitted, uses YAML if available.")
    a.add_argument("--save-mask", action="store_true",
                   help="Save the predicted mask.")
    a.add_argument("--side-by-side", action="store_true", default=True,
        help="Save side-by-side original|mask comparison (default: on)")
    a.add_argument("--no-side-by-side", dest="side_by_side", action="store_false",
        help="Disable side-by-side output")

    a.add_argument("--overlay", action="store_true",
        help="Also save image with prediction overlayed")

    a.add_argument("--overlay-alpha", type=int, default=112,
        help="Overlay alpha (0..255, default 112)")

    # utility subcommands -----------------------------------------------------
    sub.add_parser("list-models", help="List downloadable model IDs/versions")

    pc = sub.add_parser("precache", help="Download/cache weights without running inference")
    pc.add_argument("--weights", "-w", required=True,
        help="auto:<id>@<ver> or gh:<owner>/<repo>@<tag>/<file>")
    pc.add_argument("--format", choices=["auto", "pt", "onnx"], default="auto")

    return p

def _merge_yaml(cfg: dict,
                user_mean: Optional[List[float]],
                user_std: Optional[List[float]],
                user_classes: Optional[str]):
    mean = tuple(user_mean) if user_mean else None
    std = tuple(user_std) if user_std else None
    class_names = user_classes.split(",") if user_classes else None
    palette = None

    if cfg:
        # Normalization from YAML if not overridden
        if mean is None and cfg.get("normalize_mean"):
            m = cfg.get("normalize_mean")
            if isinstance(m, (list, tuple)) and len(m) == 3:
                mean = tuple(float(x) for x in m)
        if std is None and cfg.get("normalize_std"):
            s = cfg.get("normalize_std")
            if isinstance(s, (list, tuple)) and len(s) == 3:
                std = tuple(float(x) for x in s)

        # Class names from YAML if not overridden
        if class_names is None and cfg.get("class_names") is not None:
            cn = cfg["class_names"]
            if isinstance(cn, dict) and "num_classes" in cfg:
                class_names = [cn.get(i, f"class_{i}") for i in range(cfg["num_classes"])]
            elif isinstance(cn, list):
                class_names = cn

        # Palette
        if cfg.get("palette") is not None and "num_classes" in cfg:
            try:
                palette = [tuple(cfg["palette"][i]) for i in range(cfg["num_classes"])]
            except Exception:
                palette = None

    return mean, std, class_names, palette

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "list-models":
        for line in list_models():
            print(line)
        return 0

    if args.cmd == "precache":
        w, y = precache(args.weights, preferred_format=args.format)
        print(f"Cached weights: {w}")
        if y: 
            print(f"Cached config: {y}")
        return 0

    # default: infer
    w_path, y_path, cfg, fmt = resolve_weights(args.weights, preferred_format=args.format)

    # Merge YAML-driven defaults unless user overrides
    mean, std, class_names, palette = _merge_yaml(cfg, args.mean, args.std, args.class_names)

    # Log a little context
    print(f"Using weights: {w_path}")
    if y_path:
        print(f"Using config : {y_path}")
    if mean and std:
        print(f"Normalization: mean={mean}, std={std}")
    if class_names:
        print(f"Classes     : {class_names}")
    if palette:
        print(f"Palette     : {len(palette)} colors")
    
    infer.run_folder(
        input=args.input,
        output=args.output,
        weights=str(w_path),     # resolved local file path
        device=args.device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        amp=args.amp,
        mean=mean,
        std=std,
        ext=args.ext,
        csv_path=args.csv_path,
        min_class_percent=args.min_class_percent,
        suppress_noise=args.suppress_noise,
        class_names=class_names,
        palette=palette,
        save_overlay=args.overlay,
        overlay_alpha=args.overlay_alpha,
        save_side_by_side=args.side_by_side,
        save_mask=args.save_mask,
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
