import open_clip
import torch
import argparse
from pathlib import Path
from PIL import Image


@torch.no_grad()
def get_image_features(image, model):
    image_feature = model.encode_image(image)
    image_feature /= image_feature.norm(dim=-1, keepdim=True)
    return image_feature


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="ViT-bigG-14")
    parser.add_argument("--pretrained", type=str, default="laion2b_s39b_b160k")
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=device,
    )

    first = preprocess(Image.open(args.first)).unsqueeze(0).to(device)
    second = preprocess(Image.open(args.second)).unsqueeze(0).to(device)

    first_features = get_image_features(first, model)
    second_features = get_image_features(second, model)
    similarity = (first_features @ second_features.T).item()
    print(similarity)
