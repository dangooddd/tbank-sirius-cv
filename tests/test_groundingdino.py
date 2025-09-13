from groundingdino.util.inference import load_model, load_image, predict

image_path = "data/sample/tbank_blue.png"
image_source, image = load_image(image_path)
prompt = "logo"

model = load_model(
    "configs/GroundingDINO_SwinB_cfg.py",
    "models/groundingdino/groundingdino_swinb_cogcoor.pth",
    device="cuda",
)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=prompt,
    box_threshold=0.20,
    text_threshold=0.20,
)

for box, logit, phrase in zip(boxes, logits, phrases):
    print(f"Found: {phrase} with confidence {logit:.2f}")
    print(f"Bounding box: {box.tolist()}")

print(type(image_source))
print(image_source.shape)
