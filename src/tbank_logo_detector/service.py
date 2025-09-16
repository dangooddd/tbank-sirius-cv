from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
from .model import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model(config["model_name"], config["weights_path"])
    yield


config = {
    "model_name": "yolo",
    "weights_path": "models/yolo/yolov8l_best.pt",
    "conf": 0.6,
}

model = None

app = FastAPI(
    title="T-Bank Logo Detector",
    description="API для детекции логотипа Т-банка на изображениях",
    lifespan=lifespan,
)


class BoundingBox(BaseModel):
    """Абсолютные координаты BoundingBox"""

    x_min: int = Field(..., description="Левая координата", ge=0)
    y_min: int = Field(..., description="Верхняя координата", ge=0)
    x_max: int = Field(..., description="Правая координата", ge=0)
    y_max: int = Field(..., description="Нижняя координата", ge=0)


class Detection(BaseModel):
    """Результат детекции одного логотипа"""

    bbox: BoundingBox = Field(..., description="Результат детекции")


class DetectionResponse(BaseModel):
    """Ответ API с результатами детекции"""

    detections: List[Detection] = Field(..., description="Список найденных логотипов")


class ErrorResponse(BaseModel):
    """Ответ при ошибке"""

    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Дополнительная информация")


@app.post(
    "/detect",
    response_model=DetectionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении

    Args:
        file: Загружаемое изображение (JPEG, PNG, BMP, WEBP)

    Returns:
        DetectionResponse: Результаты детекции с координатами найденных логотипов
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail={"error": "Uploaded file is not an image"}
        )
