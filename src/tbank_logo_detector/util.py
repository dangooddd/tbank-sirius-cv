from pathlib import Path
from PIL import ImageDraw

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(file_path):
    file_path = Path(file_path)
    return file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS


def draw_bounding_boxes(image, boxes, color="red", width=2):
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    for box in boxes:
        draw.rectangle(box, outline=color, width=width)

    return img_copy
