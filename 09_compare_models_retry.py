"""Evaluate existing YOLO26n/s and trained Torchvision CNN models on the SAME test splits.
Usage: python 09_compare_models.py --task both
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

from common import to_square
from  importlib import import_module

ROOT = Path(__file__).resolve().parent
cnn = import_module('08_train_cnn')
MODELS = ('yolo26n', 'yolo26s', 'resnet18', 'mobilenet_v3_small')
CLASSES = {'eyes': ('Closed', 'Open'), 'yawn': ('no_yawn', 'yawn')}


def metrics(ys, ps, positive):
    n = len(ys)
    tp = sum(y == p == positive for y, p in zip(ys, ps))
    fp = sum(y != positive and p == positive for y, p in zip(ys, ps))
    fn = sum(y == positive and p != positive for y, p in zip(ys, ps))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {'accuracy': sum(y == p for y, p in zip(ys, ps)) / n, 'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}


def test_model(task, model_name, device, data_root, benchmark_reps):
    classes = CLASSES[task]
    test_folder = ROOT / data_root / task / 'test'
    ds = datasets.ImageFolder(test_folder)
    if ds.classes != list(classes):
        raise RuntimeError(f'Unexpected class names: {ds.classes}; expected {classes}')
    if model_name.startswith('yolo26'):
        from ultralytics import YOLO
        file = ROOT / 'runs' / f'{task}_{model_name}-cls' / 'weights' / 'best.pt'
        if not file.exists():
            return None, str(file)
        model = YOLO(str(file))
        if set(model.names.values()) != set(classes):
            raise RuntimeError(f'Model classes do not match: {model.names}')
        def predict(img):
            # All timing below includes common preprocessing and model preprocessing.
            result = model.predict(to_square(img), imgsz=224, device=device, verbose=False)[0]
            return result.names[result.probs.top1]
    else:
        file = ROOT / 'runs' / f'{task}_{model_name}' / 'best.pt'
        if not file.exists():
            return None, str(file)
        ckpt = torch.load(file, map_location='cpu', weights_only=True)
        if ckpt.get('classes') != list(classes):
            raise RuntimeError(f'Checkpoint classes mismatch: {ckpt.get("classes")}')
        model = cnn.make_model(model_name, pretrained=False)
        model.load_state_dict(ckpt['model'])
        torch_device = torch.device(device)
        model.to(torch_device).eval()
        tf = cnn.get_transforms(False)
        def predict(img):
            x = tf(to_square(img)).unsqueeze(0).to(torch_device)
            with torch.inference_mode():
                return classes[int(model(x).argmax(1).item())]
    # Warmup to avoid counting one-time setup costs.
    with Image.open(ds.samples[0][0]) as im:
        img = im.convert('RGB')
    for _ in range(3):
        predict(img)
    ys, ps, ms = [], [], []
    for path, idx in ds.samples:
        with Image.open(path) as im:
            img = im.convert('RGB')
        t0 = time.perf_counter()
        label = predict(img)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        ys.append(classes[idx]); ps.append(label)
        ms.append(elapsed_ms)
    m = metrics(ys, ps, 'Closed' if task == 'eyes' else 'yawn')
    # Dedicated repeated timing of one image to reduce variance; includes PIL-to-tensor/crop + model, excludes file IO.
    times = []
    for _ in range(benchmark_reps):
        t0 = time.perf_counter()
        predict(img)
        times.append((time.perf_counter() - t0) * 1000)
    m.update({'task': task, 'model': model_name, 'test_count': len(ys), 'model_mb': file.stat().st_size / (1024 ** 2),
              'latency_ms_median': statistics.median(times), 'latency_ms_test_mean': statistics.mean(ms)})
    return m, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['eyes', 'yawn', 'both'], default='both')
    ap.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    ap.add_argument('--data', default='data/processed')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Use cpu for comparable CPU benchmark')
    ap.add_argument('--reps', type=int, default=30)
    a = ap.parse_args()
    if a.reps <= 0: ap.error('--reps must be positive')
    if a.device == 'cuda' and not torch.cuda.is_available(): ap.error('CUDA not available')
    out = ROOT / 'reports'
    out.mkdir(exist_ok=True)
    results = []
    for task in (('eyes', 'yawn') if a.task == 'both' else (a.task,)):
        for model_name in a.models:
            print(f'Checking {task} / {model_name} ...', flush=True)
            result, missing = test_model(task, model_name, a.device, a.data, a.reps)
            if missing:
                print(f'  SKIPPED (weights not found): {missing}')
                continue
            results.append(result)
            print(f'  acc={result["accuracy"]:.2%}  recall={result["recall"]:.2%}  F1={result["f1"]:.3f}  median={result["latency_ms_median"]:.1f}ms')
    if not results:
        raise SystemExit('No trained model weights found')
    filepath = out / 'model_comparison.csv'
    with filepath.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print('\nSaved:', filepath)
    print('NOTE: CNN and YOLO use different ImageNet-compatible inference normalization/training recipes.\n'
          'Same data split and hardware, but not a controlled architecture-only ablation.\n'
          'Test data must not be used to tune models or decide preprocessing repeatedly.')


if __name__ == '__main__':
    main()
