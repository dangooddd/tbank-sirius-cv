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
        results = self.model.predict(image_path, conf=conf)

        if not results or len(results[0].boxes) == 0:
            return None

        # Get the box with highest confidence
        boxes = results[0].boxes
        best_box_idx = boxes.conf.argmax()

        # Get coordinates in xyxy format (x1, y1, x2, y2)
        box_coords = boxes.xyxy[best_box_idx].cpu().numpy()
        x1, y1, x2, y2 = box_coords.astype(int)

        return (x1, y1, x2, y2)
