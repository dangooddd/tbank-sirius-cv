from .yolo import YOLOModel


MODELS = {
    "yolo": YOLOModel,
}


def load_model(model_name: str, weights_path: str):
    return MODELS[model_name](weights_path)


__all__ = ["YOLOModel", "load_model"]
