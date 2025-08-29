import argparse
import sys
from . import infer

def main(argv=None):
    parser = argparse.ArgumentParser(prog="csu-vert")

    parser.add_argument('--input', "-i", required=True,
        help="Path to an input folder or glob (e.g. images/*.png)"
    )

    parser.add_argument(
        "--output", "-o", required=True,
        help="Folder where output masks will be saved"
    )

    parser.add_argument(
        "--weights", "-w", required=True,
        help="Path to model weights (.pt/.onnx) or remote spec (e.g. hf:repo@file.pt)"
    )
    parser.add_argument(
            "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
            help="Device to use for inference (default: auto-detect GPU)"
    )

    parser.add_argument(
        "--ext", default="png", choices=["png", "jpg", "tif", "tiff"],
        help="File extension of the images"
    )

    parser.add_argument(
        "--mean",
        nargs=3,
        type=float,
        default=[0.485, 0.456, 0.406],
        help="Normalization mean (default: ImageNet mean 0.485 0.456 0.406)",
    )

    parser.add_argument(
        "--std",
        nargs=3,
        type=float,
        default=[0.229, 0.224, 0.225],
        help="Normalization std (default: ImageNet std 0.229 0.224 0.225)",
    )
    
    parser.add_argument(
        "--csv", default=None,
        help="If set, write per-image stats to this CSV file."
    )

    parser.add_argument(
        "--min-class-percent", type=float, default=0.0,
        help="Treat classes with <= this percent area as noise (e.g., 0.5 for 0.5%%)."
    )

    parser.add_argument(
        "--suppress-noise", action="store_true",
        help="If set, classes under --min-class-percent are remapped to background (0) in the saved mask."
    )

    parser.add_argument(
        "--class-names", default="background,forb,graminoid,woody",
        help='Comma-separated class names (e.g. "background,forb,graminoid,woody"). '
            "Used for CSV headers; falls back to class_0..class_{C-1}."
    )

    args = parser.parse_args(argv)

    
    infer.run_folder(
        input=args.input,
        output=args.output,
        weights=args.weights,
        device=args.device,
        mean=args.mean,
        std=args.std,
        ext=args.ext,
        csv_path=args.csv,
        min_class_percent=args.min_class_percent,
        suppress_noise=args.suppress_noise,
        class_names=args.class_names.split(",") if args.class_names else None,
    )

if __name__ == "__main__":
    sys.exit(main())