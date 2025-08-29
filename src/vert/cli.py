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

    args = parser.parse_args(argv)

    
    infer.run_folder(
        input=args.input,
        output=args.output,
        weights=args.weights,
        device=args.device,
        mean=args.mean,
        std=args.std,
        ext=args.ext,
    )

if __name__ == "__main__":
    sys.exit(main())