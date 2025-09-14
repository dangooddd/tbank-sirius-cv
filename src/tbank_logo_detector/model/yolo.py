from ultralytics import YOLO


class YOLOModel:
    def __init__(self, weights: str = "models/yolo/yolov8l.pt"):
        self.model = YOLO(weights)
        self.data_yaml = "configs/yolo_dataset.yaml"

    def train(self, epochs: int = 50, batch: int = 16, imgsz: int = 640):
        return self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
        )

    def validate(self):
        return self.model.val()

    def predict(self, image_path: str, conf: float = 0.25):
        return self.model.predict(image_path, conf=conf)
