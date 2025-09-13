import open_clip
import torch
from PIL import Image

torch.set_printoptions(precision=4, sci_mode=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
image_path = "data/sample/tbank_blue.png"
prompts = [
    "logo of T-Bank company",
    "logo of Tinkoff company",
    "neither T-Bank nor Tinkoff",
]
model_name = "ViT-H-14"
model_weights = "laion2b_s32b_b79k"

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name=model_name,
    pretrained=model_weights,
    device=device,
)
tokenizer = open_clip.get_tokenizer(model_name)

image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
text = tokenizer(prompts).to(device)

with torch.no_grad(), torch.amp.autocast("cuda"):
    image_embedding = model.encode_image(image)
    text_embedding = model.encode_text(text)

    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

    similarity = (image_embedding @ text_embedding.T) * 100
    probs = similarity.softmax(dim=-1)

print(probs)
