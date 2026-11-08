"""Compare existing yawn classifiers on the SAME Roboflow external images.

Modes:
  face  = original labeled full-face image (preprocessed as in original training).
  mouth = the EXACT mouth ROI images cached by 12_external_eval.py --save-roi.

Models (when checkpoint exists):
  yolo26n, yolo26s, resnet18, mobilenet_v3_small, team_resnet18 (cached 12 eval).

No training and no edits to models or datasets. Model comparisons at 0.50 are primary;
0.10/0.15 are exploratory dev-set operating points, not verified deployment settings.

Usage:
  python 31_yawn_external_compare.py --check
  python 31_yawn_external_compare.py
  python 31_yawn_external_compare.py --models yolo26n mobilenet_v3_small
"""
from __future__ import annotations

import argparse
import csv
import importlib
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
LABELS = ('no_yawn', 'yawn')
ALL_MODELS = ('yolo26n', 'yolo26s', 'resnet18', 'mobilenet_v3_small', 'team_resnet18')


def image_index(data_root: Path):
    """Order exactly matches 12_external_eval.samples_in(..., 'yawn')."""
    root = data_root / 'yawn'
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset missing: {root}')
    entries = []
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix.lower() in EXTS and p.parent.name in LABELS:
            parts = p.relative_to(root).parts
            condition = parts[-3] if len(parts) >= 3 else 'normal'
            entries.append((p, condition, p.parent.name))
    if not entries:
        raise RuntimeError(f'No labeled images under {root}: expect yawn/<condition>/<no_yawn|yawn>/*.jpg')
    return entries


def path_key(path):
    # Reconcile absolute paths serialized on the same Windows machine.
    return str(Path(path).resolve()).replace('\\', '/').casefold()


def read_original(csv_file: Path):
    if not csv_file.is_file():
        raise FileNotFoundError(
            f'Original Roboflow baseline CSV missing: {csv_file}\n'
            'Run: python 12_external_eval.py --task yawn --data data/external_driver '
            '--team-dir teammate_yawn --output reports/yawn_roboflow --save-roi'
        )
    with csv_file.open('r', newline='', encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f)
                if r.get('task') == 'yawn' and r.get('model') == 'team_resnet18']
    if not rows:
        raise RuntimeError('Missing team_resnet18 yawn rows in external_predictions.csv')
    lookup = {}
    for r in rows:
        k = path_key(r['image'])
        if k in lookup:
            raise RuntimeError(f'Duplicate baseline CSV image: {r["image"]}')
        lookup[k] = r
    return lookup


def validate_cache(entries, original_rows, output_root):
    """Ensure ALL original images match; no accidentally skipped or mispaired ROIs."""
    paths = {path_key(p) for p, _, _ in entries}
    if paths != set(original_rows):
        missing = list(paths - set(original_rows))[:3]
        extra = list(set(original_rows) - paths)[:3]
        raise RuntimeError(
            f'Dataset vs saved CSV mismatch! Found {len(paths)} files, '
            f'{len(original_rows)} saved rows. Missing={missing}, extra={extra}.\n'
            'Use EXACTLY the same Roboflow dataset folder as the original 12_external_eval run.'
        )
    cached = {}
    for idx, (p, cond, actual) in enumerate(entries):
        r = original_rows[path_key(p)]
        if r['actual'] != actual or r['condition'] != cond:
            raise RuntimeError(f'Original CSV label/condition mismatch for {p}')
        roi = output_root / 'roi' / 'yawn' / cond / actual / f'{idx:05d}.png'
        ok = r.get('status') == 'OK' and r.get('positive_probability', '').strip() != ''
        if ok and not roi.is_file():
            raise FileNotFoundError(
                f'Saved mouth ROI missing: {roi}\n'
                'Run original 12_external_eval.py --task yawn --data data/external_driver '
                '--team-dir teammate_yawn --output reports/yawn_roboflow --save-roi'
            )
        if not ok and roi.is_file():
            print(f'[WARN] ROI exists but original status not OK: {p.name}; not using it.')
        cached[path_key(p)] = roi if ok else None
    return cached


def model_file(name):
    if name.startswith('yolo26'):
        return ROOT / 'runs' / f'yawn_{name}-cls' / 'weights' / 'best.pt'
    return ROOT / 'runs' / f'yawn_{name}' / 'best.pt'


class Predictor:
    def __init__(self, name, device='cpu'):
        self.name = name
        self.device = device
        self.file = model_file(name)
        if not self.file.is_file():
            raise FileNotFoundError(self.file)
        self.kind = 'yolo' if name.startswith('yolo26') else 'cnn'
        if self.kind == 'yolo':
            from ultralytics import YOLO
            self.net = YOLO(str(self.file))
            names = self.net.names
            mapped = {int(k): str(v).strip().lower() for k, v in names.items()}
            if sorted(mapped.values()) != sorted(LABELS):
                raise ValueError(f'Unexpected YOLO classes in {self.file}: {mapped}')
            self.positive_idx = next(idx for idx, label in mapped.items() if label == 'yawn')
        else:
            import torch
            self.torch = torch
            cnn = importlib.import_module('08_train_cnn')
            ckpt = torch.load(self.file, map_location='cpu', weights_only=True)
            names = list(ckpt.get('classes', []))
            if names != list(LABELS):
                raise ValueError(f'Unexpected CNN class order {names} in {self.file}')
            self.net = cnn.make_model(name, pretrained=False)
            self.net.load_state_dict(ckpt['model'])
            self.net.to(torch.device(device)).eval()
            self.transform = cnn.get_transforms(False)

    def probability(self, image):
        from common import to_square
        image = to_square(image, 224)
        if self.kind == 'yolo':
            result = self.net.predict(image, imgsz=224, device=self.device, verbose=False)[0]
            return float(result.probs.data[self.positive_idx].item())
        x = self.transform(image).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            return float(self.net(x).softmax(dim=1)[0, 1].item())


def read_rgb(path: Path, max_side=1280):
    from PIL import Image, ImageOps
    with Image.open(path) as f:
        img = ImageOps.exif_transpose(f).convert('RGB')
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        return img


def metrics(items, threshold):
    tp = tn = fp = fn = errors = roi_fail = 0
    latencies = []
    for row in items:
        prob = row['probability']
        pred = ('UNKNOWN' if prob is None else
                ('yawn' if prob >= threshold else 'no_yawn'))
        actual = row['actual']
        if row['status'] == 'ROI_FAIL':
            roi_fail += 1
        elif row['status'] == 'ERROR':
            errors += 1
        if pred == 'yawn':
            if actual == 'yawn':
                tp += 1
            else:
                fp += 1
        elif pred == 'no_yawn':
            if actual == 'no_yawn':
                tn += 1
            else:
                fn += 1
        elif actual == 'yawn':
            fn += 1
        if row.get('latency_ms') is not None:
            latencies.append(row['latency_ms'])
    n = len(items)
    positives = sum(x['actual'] == 'yawn' for x in items)
    negatives = n - positives
    valid = n - roi_fail - errors
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(total=n, positive_count=positives, negative_count=negatives,
                valid=valid, roi_fail=roi_fail, errors=errors, tp=tp, tn=tn,
                fp=fp, fn=fn, accuracy=(tp + tn) / n if n else 0.0,
                valid_accuracy=(tp + tn) / valid if valid else 0.0,
                recall=recall, precision=precision, f1=f1,
                fpr=fp / negatives if negatives else 0.0,
                latency_ms_median=statistics.median(latencies) if latencies else None)


def evaluate(args, entries, original_rows, rois):
    rows = []
    # Same dataset, SAME ROI file, identical failed ROI handling for all mouth-mode models.
    for model in args.models:
        if model == 'team_resnet18':
            for img, cond, actual in entries:
                r = original_rows[path_key(img)]
                prob = (float(r['positive_probability']) if
                        r.get('status') == 'OK' and r.get('positive_probability', '').strip() else None)
                rows.append(dict(image=str(img), condition=cond, actual=actual,
                                 model=model, input='mouth', status='OK' if prob is not None else r['status'],
                                 probability=prob, latency_ms=None, error=r.get('error', '')))
            continue
        path = model_file(model)
        if not path.is_file():
            print(f'[SKIP] {model}: weights absent: {path}')
            continue
        print(f'\n[MODEL] {model} | {path}', flush=True)
        try:
            predictor = Predictor(model, device=args.device)
        except Exception as exc:
            raise RuntimeError(f'Failed loading {model}: {exc}') from exc
        for input_type in args.inputs:
            print(f'[INPUT] {input_type} | images={len(entries)}', flush=True)
            # Make at least 1 warmup inference per mode outside benchmark.
            candidates = []
            for img, _, _ in entries:
                use = img if input_type == 'face' else rois[path_key(img)]
                if use is not None:
                    candidates.append(use)
                    break
            if candidates:
                predictor.probability(read_rgb(candidates[0], args.max_side))
            for i, (img, cond, actual) in enumerate(entries):
                use = img if input_type == 'face' else rois[path_key(img)]
                if use is None:
                    rows.append(dict(image=str(img), condition=cond, actual=actual,
                                     model=model, input=input_type, status='ROI_FAIL',
                                     probability=None, latency_ms=None, error='Original MediaPipe ROI failure'))
                    continue
                try:
                    image = read_rgb(use, args.max_side)
                    start = time.perf_counter()
                    prob = predictor.probability(image)
                    latency_ms = 1000.0 * (time.perf_counter() - start)
                    if not math.isfinite(prob) or not 0 <= prob <= 1:
                        raise ValueError(f'Invalid probability: {prob}')
                    rows.append(dict(image=str(img), condition=cond, actual=actual,
                                     model=model, input=input_type, status='OK',
                                     probability=prob, latency_ms=latency_ms, error=''))
                except Exception as exc:
                    rows.append(dict(image=str(img), condition=cond, actual=actual,
                                     model=model, input=input_type, status='ERROR',
                                     probability=None, latency_ms=None,
                                     error=f'{type(exc).__name__}: {exc}'))
                if (i + 1) % 100 == 0 or i + 1 == len(entries):
                    print(f'  {i + 1}/{len(entries)}', flush=True)
    return rows


def save_csv(filename, records):
    if not records:
        return
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description='Same Roboflow dataset comparison: YOLO26n/s, ResNet18, MobileNetV3-Small and team ResNet18')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/external_driver')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'reports/yawn_roboflow/external_predictions.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/yawn_external_model_compare')
    parser.add_argument('--models', nargs='+', choices=ALL_MODELS, default=list(ALL_MODELS))
    parser.add_argument('--inputs', nargs='+', choices=('face', 'mouth'), default=['face', 'mouth'])
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.50, 0.15, 0.10])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--max-side', type=int, default=1280)
    parser.add_argument('--check', action='store_true', help='Check data, weights and cached ROI; no inference')
    args = parser.parse_args()
    if args.max_side < 224 or not args.thresholds or any(not (0 < t < 1) for t in args.thresholds):
        parser.error('max-side must be >=224, thresholds must be within (0,1)')
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            parser.error('CUDA unavailable')

    entries = image_index(args.data.resolve())
    original_rows = read_original(args.baseline.resolve())
    rois = validate_cache(entries, original_rows, args.baseline.resolve().parent)
    from collections import Counter
    stats = Counter(label for _, _, label in entries)
    roi_count = sum(v is not None for v in rois.values())
    print(f'[DATA] {len(entries)} external photos; labels={dict(stats)}; same saved mouth ROIs={roi_count}/{len(entries)}')
    print('[CHECK] Matching original CSV and mouth ROI index: PASS')
    for name in args.models:
        if name == 'team_resnet18':
            print('[MODEL] team_resnet18: original 12_external_eval CSV (mouth ROI, no re-inference)')
        else:
            print(f'[MODEL] {name}: {"FOUND" if model_file(name).is_file() else "MISSING"}  {model_file(name)}')
    if args.check:
        return

    rows = evaluate(args, entries, original_rows, rois)
    if not rows:
        raise SystemExit('No model outputs produced')
    args.output.mkdir(parents=True, exist_ok=True)
    out_rows = [{k: (f'{v:.7f}' if k == 'probability' and v is not None else
                     f'{v:.3f}' if k == 'latency_ms' and v is not None else
                     '' if v is None else v)
                 for k, v in r.items()} for r in rows]
    save_csv(args.output / 'predictions.csv', out_rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r['model'], r['input'])].append(r)
    summaries = []
    print(f'\n=== SAME EXTERNAL DATA: {len(entries)} PHOTOS ===')
    print('MODEL                 INPUT   THR     ACC    RECALL   PREC     F1      FP  FN  ROI_FAIL')
    for (model, input_type), records in sorted(groups.items()):
        if len(records) != len(entries):
            raise RuntimeError(f'Expected {len(entries)} predictions from {model}/{input_type}, got {len(records)}')
        for threshold in sorted(set(args.thresholds), reverse=True):
            m = metrics(records, threshold)
            summaries.append(dict(model=model, input=input_type, threshold=threshold,
                                  **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}))
            print(f'{model:21} {input_type:6} {threshold:.2f}  '
                  f'{m["accuracy"]:6.2%}  {m["recall"]:6.2%}  {m["precision"]:6.2%}  '
                  f'{m["f1"]:6.2%}  {m["fp"]:3} {m["fn"]:3}  {m["roi_fail"]:3}')
    save_csv(args.output / 'summary.csv', summaries)

    team = next((x for x in summaries if x['model']=='team_resnet18' and x['input']=='mouth' and abs(x['threshold']-0.5)<1e-8), None)
    if team and team['total'] == 393:
        if (team['tp'] + team['tn']) != 258:
            print('[WARN] Original team benchmark expected 258/393 at 0.50. Please inspect CSV versions.')
        else:
            print('[VERIFY] Team original 0.50 benchmark reproduced: 258/393.')
    with (args.output / 'README_results.txt').open('w', encoding='utf-8') as f:
        f.write('Roboflow yawn external comparison\n'
                f'Images: {len(entries)}; saved MediaPipe mouth ROI: {roi_count}; ROI failures: {len(entries)-roi_count}\n'
                'All face-mode models use the full original face, with no MediaPipe restriction.\n'
                'All mouth-mode models use EXACTLY the same cached mouth ROI from 12_external_eval.\n'
                'Team ResNet18 mouth results reuse original prediction CSV, not a new inference.\n'
                'All thresholds are operating points on the already-used development set.\n'
                'Primary architecture comparison is threshold=0.50 within the same input mode;\n'
                'face-vs-mouth comparisons measure input pipeline changes, NOT only architecture.\n'
                'All unknowns count as incorrect. Timings exclude I/O and MediaPipe ROI extraction.\n'
                'Do not claim these results are performance on an untouched independent final test set.\n')
    print('\nSaved:', args.output / 'predictions.csv')
    print('Saved:', args.output / 'summary.csv')
    print('NOTE: 0.10/0.15 are exploratory thresholds tuned on Roboflow. Final independent evaluation needed.')
    print('NOTE: Face models trained on original faces may degrade when evaluated on mouth crops; view modes separately.')


if __name__ == '__main__':
    main()
