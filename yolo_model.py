"""Image-level pothole predictions using the supplied pretrained checkpoint."""
import os
os.environ['YOLO_AUTOINSTALL'] = 'false'
from ultralytics import YOLO
from PIL import Image
import torch

def load_yolo(path):
    torch.set_num_threads(1)
    model = YOLO(str(path))
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    classes = [int(i) for i, name in items if str(name).strip().lower() in {'pothole', 'potholes'}]
    if model.task != 'detect' or not classes:
        raise ValueError(f'Expected pothole detection weights; classes: {names!r}')
    return model, classes

def pothole_present(loaded, image, threshold=0.35):
    model, classes = loaded
    image = image.convert('RGB').copy()
    image.thumbnail((416, 416), Image.Resampling.LANCZOS)
    result = model.predict(source=image, conf=threshold, classes=classes, device='cpu', imgsz=416, verbose=False, save=False)[0]
    return bool(result.boxes is not None and len(result.boxes))
