from pathlib import Path
from sklearn.model_selection import train_test_split
from rich.progress import track
import shutil
import argparse


def copy_images_with_labels(
    images_list: list[Path],
    labels_src: Path,
    images_dst: Path,
    labels_dst: Path,
    description: str = None,
):
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    for image in track(images_list, description=description):
        label = labels_src / f"{image.stem}.txt"
        if not image.is_file() or not label.is_file():
            continue

        try:
            shutil.copyfile(image, images_dst / image.name)
            shutil.copyfile(label, labels_dst / label.name)
        except Exception:
            continue


def split_dataset_with_labels(dataset_dir, split_dst, val_size=0.2, seed=34):
    dataset_dir = Path(dataset_dir)
    split_dst = Path(split_dst)
    images_src = dataset_dir / "images"
    labels_src = dataset_dir / "labels"

    if not images_src.is_dir() or not labels_src.is_dir():
        return False

    images = list(images_src.iterdir())
    train_images, val_images = train_test_split(
        images, test_size=val_size, random_state=seed
    )

    copy_images_with_labels(
        train_images,
        labels_src,
        split_dst / "images" / "train",
        split_dst / "labels" / "train",
        "Creating train dataset...",
    )

    copy_images_with_labels(
        val_images,
        labels_src,
        split_dst / "images" / "val",
        split_dst / "labels" / "val",
        "Creating val dataset...",
    )

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset with labels")
    parser.add_argument(
        "--src", type=str, default="data/raw", help="Source dataset directory"
    )
    parser.add_argument(
        "--dst", type=str, default="data/processed", help="Destination directory"
    )
    parser.add_argument(
        "--val-size", type=float, default=0.2, help="Validation set size ratio"
    )
    parser.add_argument("--seed", type=int, default=34, help="Random seed")

    args = parser.parse_args()

    split_dataset_with_labels(args.src, args.dst, args.val_size, args.seed)
