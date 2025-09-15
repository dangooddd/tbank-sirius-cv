from groundingdino.util.inference import load_model, load_image, predict
from PIL import Image
from pathlib import Path
from rich.progress import track
from tbank_logo_detector.util import is_image
import json


class Detector:
    def __init__(
        self,
        config_path="configs/GroundingDINO_SwinB_cfg.py",
        weights_path="models/groundingdino/groundingdino_swinb_cogcoor.pth",
    ):
        self.model = load_model(config_path, weights_path)

    def __call__(
        self,
        image_path,
        text_prompt="logo",
        box_threshold=0.30,
        text_threshold=0.25,
    ):
        image, image_tensor = load_image(image_path)
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        return image, boxes, logits, phrases

    def extract_boxes(self, image, boxes):
        height, width, _ = image.shape
        extracted_boxes = []

        for box in boxes:
            x_center, y_center, w, h = box
            x1 = max(0, int((x_center - w / 2) * width))
            y1 = max(0, int((y_center - h / 2) * height))
            x2 = min(width, int((x_center + w / 2) * width))
            y2 = min(height, int((y_center + h / 2) * height))

            extracted_boxes.append(
                (
                    Image.fromarray(image[y1:y2, x1:x2, :]),
                    f"{x_center} {y_center} {w} {h}",
                )
            )

        return extracted_boxes

    def save_boxes(self, extracted_boxes, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        json_dict = {}

        for i, (image, coords) in enumerate(extracted_boxes):
            image_path = save_dir / f"{i}.jpg"
            image.save(image_path)
            json_dict[i] = coords

        try:
            with open(save_dir / "metadata.json", "w") as file:
                json.dump(json_dict, file)
        except Exception as err:
            print(f"Ошибка при записи json: {err}")


def detect(images_dir: Path, output_dir: Path):
    """
    Detects and saves all logo boxes found in images dir
    If follow directory is given:
        images_dir/
            img1.jpg
            img2.jpg
    There will be created directory:
        output_dir/
            img1/
                0.jpg
                1.jpg
                metadata.json
            img2/
                ...
    """
    detector = Detector()

    for file_path in track(
        list(images_dir.iterdir()), description="Extracting logos..."
    ):
        if is_image(file_path):
            save_dir = output_dir / file_path.stem
            image, boxes, _, _ = detector(file_path)
            extracted_boxes = detector.extract_boxes(image=image, boxes=boxes)
            detector.save_boxes(extracted_boxes, save_dir)
