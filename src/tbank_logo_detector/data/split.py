from pathlib import Path
from sklearn.model_selection import train_test_split
from rich.progress import track
import shutil


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


def split(dataset_dir: Path, split_dst: Path, val_size: float, seed=34):
    """
    Сделать test/val разделение данного датасета

    Args:
        dataset_dir: путь к датасету в raw формате
        split_dst: пусть к новому датасету в подходящем для YOLO формате
        val_size: относительный размер валидационной выборки
        seed: сид разделения
    """
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
