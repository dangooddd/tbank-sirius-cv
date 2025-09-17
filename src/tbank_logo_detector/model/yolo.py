from ultralytics import YOLO
from PIL import Image


class YOLOModel:
    def __init__(self, weights: str = "weights/yolo/yolov8l.pt"):
        self.model = YOLO(weights)
        self.cfg = "configs/yolo_config.yaml"

    def train(
        self,
        epochs: int = 50,
        batch: int = 16,
        imgsz: int = 640,
        nbs: int = 32,
        lr0: float = 0.01,
        resume: bool = False,
    ):
        return self.model.train(
            cfg=self.cfg,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            nbs=nbs,
            lr0=lr0,
        )

    def validate(self):
        return self.model.val()

    def predict(self, image: str | Image.Image, conf: float = 0.4):
        results = self.model.predict(image, conf=conf)

        # Преобразование действительных box в целые
        boxes = results[0].boxes
        box_list = []

        for box in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box.astype(int)
            box_list.append((x1, y1, x2, y2))

        return box_list
