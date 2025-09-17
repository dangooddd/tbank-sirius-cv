import open_clip
import torch
from PIL import Image
from pathlib import Path
from ..util import is_image
from rich.progress import track
import json


positive_prompts = [
    "логотип Т-Банк",
    "значок Т-Банк",
    "эмблема Т-Банк со щитом",
    "Т-Банк",
    "Символ Т-Банка",
    "T-Bank logo with shield",
    "T-Bank logo",
    "Официальный логотип Т-Банка",
    "Логотип Т-Банка на белом фоне",
    "желтый логотип со щитом",
    "белый логотип со щитом",
    "щитом с буквой Т",
]

negative_prompts = [
    "мобильный оператор",
    "добывающая компания",
    "самолет",
    "american bank",
    "unknown",
    "unknown logo",
    "неизвестное лого",
    "телепередача",
    "mobile operator",
    "зеленый логотип",
    "красный логотип",
    "синий логотип",
    "cиний",
    "красный",
    "зеленый",
    "круг",
    "круглое лого",
    "circle",
    "green logo",
    "red logo",
    "blue logo",
    "red",
    "green",
    "blue",
    "текст",
    "text",
    "банкомат",
    "логотип банкомата",
    "Tinkoff logo",
    "Deutsche logo",
    "логотип Тинькофф",
    "эмблема Тинькофф",
    "логотип Мегафон",
    "логотип МТС",
    "логотип Билайн",
    "логотип Теле-2",
    "логотип Газпром",
    "логотип Россельхоз",
    "логотип Сбербанк",
    "логотип Роснефть",
    "логотип Тинькофф",
    "логотип VK",
    "логотип Лукойл",
    "логотип Авито",
    "логотип ВТБ",
    "логотип Яндекс",
    "логотип Райффайзен",
    "логотип Google",
    "логотип Касперский",
    "логотип Microsoft",
    "логотип Windows",
    "Windows",
    "Microsoft",
    "Тинькофф",
    "Тинькофф",
    "Мегафон",
    "МТС",
    "Билайн",
    "Теле-2",
    "Газпром",
    "Россельхоз",
    "Сбербанк",
    "Роснефть",
    "Тинькофф",
    "VK",
    "Лукойл",
    "Авито",
    "ВТБ",
    "Яндекс",
    "Райффайзен",
    "Google",
    "Касперский",
]


class LegacyAnnotator:
    @torch.no_grad()
    def __init__(
        self,
        model_name="ViT-H-14",
        pretrained="laion2b_s32b_b79k",
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        device="cuda",
    ):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.positive_prompts = positive_prompts
        self.negative_prompts = negative_prompts

        # calculate prompts features
        self.positive_prompts_features = self.model.encode_text(
            self.tokenizer(self.positive_prompts).to(device)
        )
        self.positive_prompts_features /= self.positive_prompts_features.norm(
            dim=-1, keepdim=True
        )
        self.negative_prompts_features = self.model.encode_text(
            self.tokenizer(self.negative_prompts).to(device)
        )
        self.negative_prompts_features /= self.negative_prompts_features.norm(
            dim=-1, keepdim=True
        )

    @torch.no_grad
    def _get_image_features(self, image):
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    @torch.no_grad
    def _get_scores(self, image_features, text_features):
        similarity = image_features @ text_features.T
        return similarity

    def annotate_image_from_boxes(self, boxes_path):
        boxes_path = Path(boxes_path)
        best_box = -1
        best_box_score = 0

        try:
            with open(boxes_path / Path("metadata.json"), "r") as file:
                metadata = json.load(file)
        except Exception:
            return

        for file_path in boxes_path.iterdir():
            if is_image(file_path):
                num = file_path.stem
                image = (
                    self.preprocess(Image.open(file_path)).unsqueeze(0).to(self.device)
                )
                image_features = self._get_image_features(image)
                positive_scores = self._get_scores(
                    image_features, self.positive_prompts_features
                )
                negative_scores = self._get_scores(
                    image_features, self.negative_prompts_features
                )

                if positive_scores.max() > negative_scores.max():
                    if best_box_score < positive_scores.max():
                        best_box_score = positive_scores.max()
                        best_box = num

        if best_box == -1:
            return ""
        else:
            return f"0 {metadata[best_box]}"

    def create_label_file(self, label_path, annotation):
        if annotation is None:
            return

        label_path = Path(label_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, "w") as file:
            file.write(annotation)


class Annotator:
    @torch.no_grad()
    def __init__(
        self,
        model_name: str = "ViT-bigG-14",
        pretrained: str = "laion2b_s39b_b160k",
        reference_dir: Path = Path("assets/reference"),
        conf: float = 0.8,
        device: str = "cuda",
    ):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
        )
        self.conf = conf

        images = []
        for image_path in reference_dir.iterdir():
            if is_image(image_path):
                images.append(self.preprocess(Image.open(image_path)).to(self.device))

        self.features = self._get_image_features(torch.stack(images))

    @torch.no_grad
    def _get_image_features(self, image):
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    @torch.no_grad
    def _cosine_similarity(self, first_features, second_features):
        similarity = first_features @ second_features.T
        return similarity

    def annotate_image_from_boxes(self, boxes_path):
        boxes_path = Path(boxes_path)
        boxes = []

        try:
            with open(boxes_path / Path("metadata.json"), "r") as file:
                metadata = json.load(file)
        except Exception:
            return

        for file_path in boxes_path.iterdir():
            if is_image(file_path):
                box_num = file_path.stem
                image = (
                    self.preprocess(Image.open(file_path)).unsqueeze(0).to(self.device)
                )
                image_features = self._get_image_features(image)
                scores = self._cosine_similarity(image_features, self.features)

                if scores.max() > self.conf:
                    boxes.append(box_num)

        annotation = ""
        for box_num in boxes:
            annotation += f"0 {metadata[box_num]}\n"

        return annotation

    def create_label_file(self, label_path, annotation):
        if annotation is None:
            return

        label_path = Path(label_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, "w") as file:
            file.write(annotation)


def legacy_annotate(
    boxes_dir: Path,
    output_dir: Path,
    model_name: str,
    pretrained: str,
):
    """Annotates images from boxes that was defined in detect.py"""
    annotator = Annotator(model_name=model_name, pretrained=pretrained)

    for boxes_path in track(list(boxes_dir.iterdir()), "Annotating images..."):
        if boxes_path.is_dir():
            label_path = output_dir / f"{boxes_path.stem}.txt"
            annotation = annotator.annotate_image_from_boxes(boxes_path)
            annotator.create_label_file(label_path, annotation)


def annotate(
    boxes_dir: Path,
    output_dir: Path,
    reference_dir: Path,
    conf: float,
    model_name: str,
    pretrained: str,
):
    """Annotates images from boxes that was defined in detect.py"""
    annotator = Annotator(
        model_name=model_name,
        pretrained=pretrained,
        reference_dir=reference_dir,
        conf=conf,
    )

    for boxes_path in track(list(boxes_dir.iterdir()), "Annotating images..."):
        if boxes_path.is_dir():
            label_path = output_dir / f"{boxes_path.stem}.txt"
            annotation = annotator.annotate_image_from_boxes(boxes_path)
            annotator.create_label_file(label_path, annotation)
