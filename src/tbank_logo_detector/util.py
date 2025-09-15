from pathlib import Path
from PIL import ImageDraw

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(file_path):
    """Helper function to determine is file image or not"""
    file_path = Path(file_path)
    return file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS


def draw_bounding_box(image, x1, y1, x2, y2, color="red", width=2):
    """Draw borser around block with angles in (x1, y1) and (x2, y2)"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    return img_copy
