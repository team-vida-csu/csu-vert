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
            "--device", default="auto", choices=["auto", "cpu", "cuda"],
            help="Device to use for inference (default: auto-detect GPU)"
    )

    parser.add_argument(
        "--ext", default="png", choices=[".png", ".jpg"],
        help="File extension of the images"
    )

    args = parser.parse_args(argv)

    
    infer.run_folder(
        input=args.input,
        output=args.output,
        weights=args.weights,
        device=args.device,
        ext=args.ext,
    )

if __name__ == "__main__":
    sys.exit(main())