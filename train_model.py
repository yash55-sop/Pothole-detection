import hashlib, json, random
from pathlib import Path
from PIL import Image
import torch
from tiny_detector import TinyDetector, boxes_from_labels, tensor, target, loss, predict

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'expanded_data'

def evaluate(model, paths, threshold):
    tp = fp = fn = tn = 0
    rows = []
    for image_path in paths:
        with Image.open(image_path) as image:
            predicted = bool(predict(model, image, threshold))
        actual = bool(boxes_from_labels(DATA / 'labels' / (image_path.stem + '.txt')))
        tp += int(actual and predicted)
        fp += int(not actual and predicted)
        fn += int(actual and not predicted)
        tn += int(not actual and not predicted)
        rows.append({'image': image_path.name, 'actual': actual, 'predicted': predicted})
    total = tp + fp + fn + tn
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'accuracy': (tp + tn) / total if total else None,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'specificity': tn / (tn + fp) if tn + fp else None, 'images': rows}


def main():
    torch.set_num_threads(2); torch.manual_seed(42)
    images = sorted((DATA / 'images').glob('*.jpg'))
    if len(images) < 20: raise ValueError('Expected at least 20 labeled images in expanded_data')
    random.Random(42).shuffle(images)
    a, b = int(len(images)*0.7), int(len(images)*0.85)
    splits = {'train': images[:a], 'val': images[a:b], 'test': images[b:]}
    def sample(path):
        labels = boxes_from_labels(DATA / 'labels' / (path.stem + '.txt'))
        with Image.open(path) as image: x = tensor(image)
        return x, target(labels), target([(1-x, y, w, h) for x,y,w,h in labels])
    training = [sample(p) for p in splits['train']]; validation = [sample(p) for p in splits['val']]
    model = TinyDetector(); optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best = float('inf'); best_state = None; best_epoch = 0
    rng = random.Random(42)
    for epoch in range(30):
        model.train(); rng.shuffle(training)
        for start in range(0, len(training), 8):
            batch = training[start:start+8]; xs=[]; ys=[]
            for x, y, flipped in batch:
                if rng.random() < 0.5: xs.append(x.flip(-1)); ys.append(flipped)
                else: xs.append(x); ys.append(y)
            optimizer.zero_grad(); l = loss(model(torch.stack(xs)), torch.stack(ys)); l.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): val = sum(loss(model(x.unsqueeze(0)), y.unsqueeze(0)).item() for x,y,_ in validation) / len(validation)
        if val < best:
            best = val; best_epoch = epoch + 1; best_state = {k: v.clone() for k,v in model.state_dict().items()}
        print(f'Epoch {epoch+1}/30, validation loss {val:.4f}', flush=True)
    model.load_state_dict(best_state); model.eval()
    # Fixed threshold: retain the existing model's detection setting.
    threshold = 0.25
    validation_metrics = evaluate(model, splits['val'], threshold)
    metrics = evaluate(model, splits['test'], threshold)
    path = ROOT / 'models/tiny_detector.pt'; path.parent.mkdir(exist_ok=True)
    torch.save(best_state, path)
    report = {'model': 'Custom CNN trained from scratch', 'split_seed': 42, 'splits': {k: [p.name for p in v] for k,v in splits.items()}, 'epochs': 30, 'best_epoch': best_epoch, 'parameters': sum(p.numel() for p in model.parameters()), 'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'confidence_threshold': threshold, 'threshold_selection': 'fixed at 0.25', 'evaluation_type': 'image-level pothole presence', 'validation': validation_metrics, **metrics, 'limitation': 'Small test set from one source; negative images may be absent.'}
    out = ROOT / 'evaluation_results/tiny_test.json'; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__': main()
