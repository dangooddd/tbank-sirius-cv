import io
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from .model import load_model

config = {
    "model_name": "yolo",
    "weights_path": "weights/weights.pt",
    "conf": 0.6,
}

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model(config["model_name"], config["weights_path"])
    yield


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


class ServiceError(Exception):
    def __init__(self, status_code: int, error_response: ErrorResponse):
        self.status_code = status_code
        self.error_response = error_response


@app.exception_handler(ServiceError)
async def http_exception_handler(request: Request, exc: ServiceError):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.error_response.model_dump(),
    )


@app.post("/detect", response_model=DetectionResponse)
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении

    Args:
        file: Загружаемое изображение (JPEG, PNG, BMP, WEBP)

    Returns:
        DetectionResponse: Результаты детекции с координатами найденных логотипов
    """
    if not file.content_type.startswith("image/"):
        raise ServiceError(
            status_code=400,
            error_response=ErrorResponse(error="Uploaded file is not an image"),
        )

    image_bytes = await file.read()
    image_stream = io.BytesIO(image_bytes)

    try:
        # Без .copy обьект image остается связанным с image_stream.
        # В таком случае после image.verify() указатель image_stream сместится
        # и повторно использовать изображение не выйдет
        image = Image.open(image_stream).copy()
        image.verify()
    except UnidentifiedImageError as e:
        raise ServiceError(
            status_code=400,
            error_response=ErrorResponse(error="Invalid image file", detail=str(e)),
        )
    except Exception as e:
        raise ServiceError(
            status_code=500,
            error_response=ErrorResponse(error="Error processing image", detail=str(e)),
        )

    global model
    try:
        boxes = model.predict(image=image, conf=config["conf"])
    except Exception as e:
        raise ServiceError(
            status_code=500,
            error_response=ErrorResponse(error="Error while predicting", detail=str(e)),
        )

    detections = []
    for box in boxes:
        detections.append(
            Detection(
                bbox=BoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3])
            )
        )

    return DetectionResponse(detections=detections)
