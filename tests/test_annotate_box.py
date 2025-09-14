import argparse
from pathlib import Path

from annotation.annotate import Annotator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate a single detected box directory"
    )
    parser.add_argument(
        "--box-dir",
        type=str,
        required=True,
        help="Path to the directory with detected box images and metadata.json",
    )
    args = parser.parse_args()

    annotator = Annotator()
    box_path = Path(args.box_dir)

    if not box_path.exists():
        print(f"Directory '{box_path}' does not exists")
        exit(1)

    if not box_path.is_dir():
        print(f"'{box_path}' is not a directory")
        exit(1)

    annotation = annotator.annotate_image_from_boxes(box_path)
    print(annotation)
