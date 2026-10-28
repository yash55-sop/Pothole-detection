"""Evaluate pretrained YOLO on the same held-out images as the custom model."""
import hashlib, json
from pathlib import Path
from PIL import Image, ImageOps
from tiny_detector import boxes_from_labels
from yolo_model import load_yolo, pothole_present

ROOT = Path(__file__).resolve().parent

def main():
    custom = json.loads((ROOT/'evaluation_results/tiny_test.json').read_text())
    path = ROOT/'models/best.pt'
    model = load_yolo(path)
    threshold = 0.35
    tp = fp = fn = tn = 0
    rows = []
    for name in custom['splits']['test']:
        with Image.open(ROOT/'expanded_data/images'/name) as image:
            predicted = pothole_present(model, ImageOps.exif_transpose(image), threshold)
        actual = bool(boxes_from_labels(ROOT/'expanded_data/labels'/(Path(name).stem+'.txt')))
        tp += int(actual and predicted); fp += int(not actual and predicted)
        fn += int(actual and not predicted); tn += int(not actual and not predicted)
        rows.append({'image': name, 'actual': actual, 'predicted': predicted})
        print(f'{name}: actual={actual}, predicted={predicted}', flush=True)
    report = {'model': 'Pretrained YOLOv8', 'evaluation_type': 'image-level pothole presence',
              'splits': custom['splits'], 'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'confidence_threshold': threshold, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
              'precision': tp/(tp+fp) if tp+fp else None, 'recall': tp/(tp+fn) if tp+fn else None,
              'accuracy': (tp+tn)/(tp+fp+fn+tn), 'images': rows,
              'limitation': 'Small held-out sample; overlap with pretrained training data is unknown.'}
    (ROOT/'evaluation_results/yolo_test.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__': main()
