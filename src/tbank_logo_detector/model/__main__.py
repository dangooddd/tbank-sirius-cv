import argparse
from pathlib import Path
from . import load_model


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
        )
    elif args.command == "predict":
        predict(
            model_name=args.model_name,
            weights_path=args.weights_path,
            image_path=args.image_path,
            conf=args.conf,
        )


def train(model_name: str, weights_path: Path, epochs=50, batch=16):
    model = load_model(model_name, weights_path)
    model.train(epochs=epochs, batch=batch)


def predict(model_name: str, weights_path: Path, image_path: Path, conf: float):
    model = load_model(model_name, weights_path)
    return model.predict(image_path=image_path, conf=conf)


if __name__ == "__main__":
    main()
