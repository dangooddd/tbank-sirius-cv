import open_clip
import torch
from PIL import Image
from pathlib import Path
from utils import is_image


class Annotator:
    @torch.no_grad()
    def __init__(
        self,
        model_name="ViT-H-14",
        model_weights="laion2b_s32b_b79k",
        positive_prompts=[
            "логотип Т-Банк",
            "эмблема Т-Банк",
            "значок Т-Банк на белом фоне",
            "значок Т-Банк на желтом фоне",
            "shield T-Bank logo",
            "эмблема Т-Банк со щитом",
            "Т-Банк",
            "T-Bank logo",
        ],
        negative_prompts=[
            "логотип Тинькофф",
            "эмблема Тинькофф",
            "логотип не Т-Банка",
            "логотип неизвестной компании",
            "иконка без текста Т-Банк",
            "не Т-Банк",
            "Тинькофф",
            "Tinkoff",
            "random logo",
        ],
        device="cuda",
    ):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=model_weights,
            device=device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.count_positive = len(positive_prompts)
        self.prompts = positive_prompts + negative_prompts

        # calculate prompts features
        self.prompts_features = self.model.encode_text(
            self.tokenizer(self.prompts).to(device)
        )
        self.prompts_features /= self.prompts_features.norm(dim=-1, keepdim=True)

    @torch.no_grad
    def _get_image_features(self, image):
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    @torch.no_grad
    def _get_scores(self, image_features, text_features):
        similarity = image_features @ text_features.T
        return similarity.softmax(dim=-1).squeeze()

    @torch.no_grad
    def annotate_image_from_boxes(self, boxes_path):
        boxes_path = Path(boxes_path)
        best_box = -1
        best_box_score = 0

        for file_path in boxes_path.iterdir():
            if is_image(file_path):
                num = file_path.stem
                image = (
                    self.preprocess(Image.open(file_path)).unsqueeze(0).to(self.device)
                )
                image_features = self._get_image_features(image)
                scores = self._get_scores(image_features, self.prompts_features)
                print(scores)

                if scores.argmax() < self.count_positive:
                    if best_box_score < scores.max():
                        best_box_score = scores.max()
                        best_box = num

        return best_box


if __name__ == "__main__":
    annotator = Annotator()
    print(
        annotator.annotate_image_from_boxes(
            "/home/dangooddd/Dev/sirius-ml/data/raw/boxes/1e9e23475bf3281ec1651e4676de8da8"
        )
    )
