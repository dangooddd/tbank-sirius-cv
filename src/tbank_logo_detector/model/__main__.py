import argparse
from pathlib import Path
from PIL import Image
from . import load_model
from ..util import draw_bounding_boxes


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train command
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model-name", type=str, default="yolo")
    train_parser.add_argument(
        "--weights-path", type=Path, default="models/yolo/yolov8l.pt"
    )
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch", type=int, default=16)
    train_parser.add_argument("--nbs", type=int, default=64)
    train_parser.add_argument("--lr0", type=float, default=0.01)
    train_parser.add_argument("--resume", type=bool, default=False)

    # predict command
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model-name", type=str, default="yolo")
    predict_parser.add_argument(
        "--weights-path", type=Path, default="models/yolo/yolov8l.pt"
    )
    predict_parser.add_argument("--image-path", type=Path, required=True)
    predict_parser.add_argument("--conf", type=float, default=0.25)

    args = parser.parse_args()

    if args.command == "train":
        train(
            model_name=args.model_name,
            weights_path=args.weights_path,
            epochs=args.epochs,
            batch=args.batch,
            lr0=args.lr0,
            nbs=args.nbs,
            resume=args.resume,
        )
    elif args.command == "predict":
        predict(
            model_name=args.model_name,
            weights_path=args.weights_path,
            image_path=args.image_path,
            conf=args.conf,
        )


def train(model_name: str, weights_path: Path, epochs, batch, nbs, lr0, resume):
    model = load_model(model_name, weights_path)
    model.train(epochs=epochs, batch=batch, lr0=lr0, nbs=nbs, resume=resume)


def predict(model_name: str, weights_path: Path, image_path: Path, conf: float):
    model = load_model(model_name=model_name, weights_path=weights_path)
    result = model.predict(image=image_path, conf=conf)

    if len(result) == 0:
        print("Логотип Т-Банка не найден")
    else:
        print("Логотип Т-Банк найден:", result)
        image = Image.open(image_path)
        image = draw_bounding_boxes(image, result)
        image.show()


if __name__ == "__main__":
    main()
