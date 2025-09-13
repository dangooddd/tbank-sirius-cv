from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(file_path):
    """Helper function to determine is file image or not"""
    file_path = Path(file_path)
    return file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
