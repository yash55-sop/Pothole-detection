import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn

SIZE = 128
GRID = 8

class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a, b, stride):
            return nn.Sequential(nn.Conv2d(a, b, 3, stride, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU())
        self.net = nn.Sequential(block(3, 16, 2), block(16, 32, 2), block(32, 48, 2), block(48, 64, 2), block(64, 96, 1), block(96, 96, 1), nn.Conv2d(96, 5, 1))
    def forward(self, x):
        return self.net(x).permute(0, 2, 3, 1)

def tensor(image):
    image = image.convert('RGB').resize((SIZE, SIZE))
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255

def boxes_from_labels(path):
    result = []
    for line in path.read_text().splitlines():
        cls, x, y, w, h = map(float, line.split())
        if cls == 0:
            result.append((x, y, w, h))
    return result

def target(boxes):
    t = torch.zeros(GRID, GRID, 5)
    for x, y, w, h in boxes:
        col, row = min(int(x * GRID), GRID - 1), min(int(y * GRID), GRID - 1)
        t[row, col] = torch.tensor([1, x * GRID - col, y * GRID - row, w, h])
    return t

def loss(output, t):
    obj = nn.functional.binary_cross_entropy_with_logits(output[..., 0], t[..., 0], pos_weight=output.new_tensor(8.0))
    mask = t[..., 0].bool()
    box = nn.functional.smooth_l1_loss(output[..., 1:][mask].sigmoid(), t[..., 1:][mask]) if mask.any() else output.sum() * 0
    return obj + 5 * box

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    intersection = max(0, min(ax + aw/2, bx + bw/2) - max(ax - aw/2, bx - bw/2)) * max(0, min(ay + ah/2, by + bh/2) - max(ay - ah/2, by - bh/2))
    return intersection / max(aw * ah + bw * bh - intersection, 1e-9)

def predict(model, image, threshold=0.35):
    with torch.no_grad():
        p = model(tensor(image).unsqueeze(0))[0].sigmoid()
    candidates = []
    for row in range(GRID):
        for col in range(GRID):
            score, x, y, w, h = p[row, col].tolist()
            if score >= threshold:
                candidates.append((score, ((col+x)/GRID, (row+y)/GRID, w, h)))
    candidates.sort(reverse=True, key=lambda item: item[0])
    kept = []
    for candidate in candidates:
        if all(iou(candidate[1], other[1]) < 0.5 for other in kept): kept.append(candidate)
    return kept

def annotate(image, predictions):
    image = image.copy().convert('RGB'); draw = ImageDraw.Draw(image)
    for score, (x, y, w, h) in predictions:
        xy = (max(0, (x-w/2)*image.width), max(0, (y-h/2)*image.height), min(image.width-1, (x+w/2)*image.width), min(image.height-1, (y+h/2)*image.height))
        draw.rectangle(xy, outline='red', width=max(2, image.width//300))
        draw.text((xy[0], xy[1]), f'Pothole {score:.2f}', fill='red')
    return image
