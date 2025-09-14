from tbank_logo_detector.model.yolo import YOLOModel
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--weights", type=str, default="models/yolo/yolov8l.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    model = YOLOModel(args.weights)
    model.train(epochs=args.epochs, batch=args.batch)
