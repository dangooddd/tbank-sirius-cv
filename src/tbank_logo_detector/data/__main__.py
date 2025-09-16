import argparse
from pathlib import Path
from .annotate import annotate
from .detect import detect
from .split import split


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # annotate command
    annotate_parser = subparsers.add_parser("annotate")
    annotate_parser.add_argument("--input-dir", type=Path, default="data/raw/boxes")
    annotate_parser.add_argument("--output-dir", type=Path, default="data/raw/labels")
    annotate_parser.add_argument("--reference-dir", type=Path, default="data/reference")
    annotate_parser.add_argument("--conf", type=float, default=0.8)
    annotate_parser.add_argument("--model-name", type=str, default="ViT-bigG-14")
    annotate_parser.add_argument("--pretrained", type=str, default="laion2b_s39b_b160k")

    # detect command
    detect_parser = subparsers.add_parser("detect")
    detect_parser.add_argument("--input-dir", type=Path, default="data/raw/images")
    detect_parser.add_argument("--output-dir", type=Path, default="data/raw/boxes")
    detect_parser.add_argument("--box-threshold", type=float, default=0.3)
    detect_parser.add_argument("--text-threshold", type=float, default=0.25)

    # split command
    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--src", type=Path, default="data/raw")
    split_parser.add_argument("--dst", type=Path, default="data/processed")
    split_parser.add_argument("--val-size", type=float, default=0.2)
    split_parser.add_argument("--seed", type=int, default=34)

    args = parser.parse_args()

    if args.command == "annotate":
        annotate(
            boxes_dir=args.input_dir,
            output_dir=args.output_dir,
            reference_dir=args.reference_dir,
            conf=args.conf,
            model_name=args.model_name,
            pretrained=args.pretrained,
        )
    elif args.command == "detect":
        detect(
            images_dir=args.input_dir,
            output_dir=args.output_dir,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
    elif args.command == "split":
        split(
            dataset_dir=args.src,
            split_dst=args.dst,
            val_size=args.val_size,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
