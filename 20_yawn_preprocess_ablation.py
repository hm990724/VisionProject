"""Evaluate 4 mouth-ROI preprocessing variants with the existing team ResNet18.

Input: reports/yawn_roboflow/external_predictions.csv and its saved ROI images.
Outputs: predictions.csv, summary.csv, example_variants.jpg.
No training, dataset edits, MediaPipe re-extraction or threshold tuning.
"""
import argparse
import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

ROOT = Path(__file__).resolve().parent
EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
VARIANTS = ('baseline', 'tight_85', 'soft_color', 'grayscale')


def load_yawn_model(team_dir):
    source = team_dir / '03_yawn_common.py'
    weights = team_dir / 'runs/03_yawn_resnet18/weights/03_best.pt'
    for p in (source, weights):
        if not p.is_file():
            raise FileNotFoundError(f'Team model file not found: {p}')
    spec = importlib.util.spec_from_file_location('team_yawn_ablation', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    device = module.choose_device('cpu')
    model, checkpoint = module.load_checkpoint(weights, device)
    class_names = checkpoint.get('class_names')
    if class_names is not None and list(class_names) != ['no_yawn', 'yawn']:
        raise ValueError(f'Unexpected class order: {class_names}')
    model.eval()
    return model, module.make_transform(training=False), device


def source_index(data):
    base = data.resolve() / 'yawn'
    files = sorted(p for p in base.rglob('*') if p.is_file()
                   and p.suffix.lower() in EXTS
                   and p.parent.name in ('yawn', 'no_yawn')) if base.is_dir() else []
    return {str(p.resolve()).casefold(): i for i, p in enumerate(files)}


def variants_of(image):
    original = image.convert('RGB')
    width, height = original.size
    dx, dy = round(width * 0.075), round(height * 0.075)
    tight = original.crop((dx, dy, width - dx, height - dy))
    # Tests whether reduced lip/chin/neck margins improve recognition.
    # Soft color version reduces saturation/contrast sensitivity, NOT true color calibration.
    softened = ImageEnhance.Color(original).enhance(0.65)
    softened = ImageEnhance.Contrast(softened).enhance(0.88)
    gray = ImageOps.grayscale(original).convert('RGB')
    return dict(baseline=original, tight_85=tight,
                soft_color=softened, grayscale=gray)


def read_input(path):
    with path.open(newline='', encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if r.get('task') == 'yawn' and r.get('model') == 'team_resnet18']


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records, threshold):
    groups = defaultdict(list)
    for r in records:
        groups[r['variant']].append(r)
    results = []
    for variant in VARIANTS:
        items = groups[variant]
        tp = sum(r['actual'] == 'yawn' and r['predicted'] == 'yawn' for r in items)
        tn = sum(r['actual'] == 'no_yawn' and r['predicted'] == 'no_yawn' for r in items)
        fp = sum(r['actual'] == 'no_yawn' and r['predicted'] == 'yawn' for r in items)
        fn = sum(r['actual'] == 'yawn' and r['predicted'] != 'yawn' for r in items)
        unknown = sum(r['predicted'] == 'UNKNOWN' for r in items)
        errors = sum(r['status'] == 'ERROR' for r in items)
        n_yawn = sum(r['actual'] == 'yawn' for r in items)
        n_normal = sum(r['actual'] == 'no_yawn' for r in items)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / n_yawn if n_yawn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        valid = len(items) - unknown
        d = dict(variant=variant, threshold=threshold, total=len(items), valid=valid,
                 unknown=unknown, errors=errors, tp=tp, tn=tn, fp=fp,
                 fn_including_unknown=fn, overall_accuracy=round((tp + tn) / len(items), 5),
                 valid_accuracy=round((tp + tn) / valid, 5) if valid else '',
                 yawn_recall=round(recall, 5), precision=round(precision, 5),
                 f1=round(f1, 5), open_false_alarm_rate=round(fp / n_normal, 5) if n_normal else '')
        results.append(d)
        print(f"{variant:12} acc={d['overall_accuracy']:.2%} "
              f"yawn_recall={recall:.2%} precision={precision:.2%} "
              f"F1={f1:.2%} FP={fp} FN={fn} UNKNOWN={unknown}")
    return results


def make_examples(cases, output, max_cases=12):
    if not cases:
        return
    width, rowh = 950, 176
    sheet = Image.new('RGB', (width, rowh * min(len(cases), max_cases)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (filename, pics, probs) in enumerate(cases[:max_cases]):
        y = i * rowh
        draw.text((8, y + 5), filename[:110], fill='black')
        for j, variant in enumerate(VARIANTS):
            im = pics[variant].copy()
            im.thumbnail((220, 117))
            x = 8 + j * 235
            sheet.paste(im, (x + (220-im.width)//2, y + 27 + (117-im.height)//2))
            draw.text((x, y + 150), f'{variant}: {probs[variant]:.3f}', fill='#223344')
    sheet.save(output, quality=90)


def main():
    ap = argparse.ArgumentParser(description='Compare existing team yawn ResNet18 across mouth ROI preprocessing')
    ap.add_argument('--input', type=Path, default=ROOT / 'reports/yawn_roboflow/external_predictions.csv')
    ap.add_argument('--roi-root', type=Path, default=ROOT / 'reports/yawn_roboflow/roi')
    ap.add_argument('--data', type=Path, default=ROOT / 'data/external_driver')
    ap.add_argument('--team-dir', type=Path, default=ROOT / 'teammate_yawn')
    ap.add_argument('--output', type=Path, default=ROOT / 'reports/yawn_roboflow/preprocess_ablation')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--limit', type=int, default=0, help='Test first N images, 0 for all')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--examples', type=int, default=12)
    args = ap.parse_args()
    if not args.input.is_file():
        ap.error(f'Missing CSV: {args.input}')
    if not 0 < args.threshold < 1 or args.batch < 1 or args.limit < 0:
        ap.error('threshold must be between 0 and 1, batch >= 1, limit >= 0')
    records = read_input(args.input)
    if not records:
        ap.error('No team_resnet18 yawn records')
    if args.limit:
        records = records[:args.limit]
    indices = source_index(args.data)
    if not indices:
        ap.error('No source images under --data/yawn; check dataset path')
    model, transform, device = load_yawn_model(args.team_dir.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    outputs, examples = [], []
    mismatch = 0
    missing = 0
    for k, row in enumerate(records):
        path = Path(row['image'])
        idx = indices.get(str(path.resolve()).casefold())
        roi = (args.roi_root / 'yawn' / row['condition'] / row['actual'] /
               f'{idx:05d}.png') if idx is not None else None
        original_status = row.get('status', '')
        predictions = {}
        if original_status == 'OK' and roi is not None and roi.is_file():
            with Image.open(roi) as img:
                pics = variants_of(img.copy())
            names, tensors = [], []
            for variant in VARIANTS:
                names.append(variant)
                tensors.append(transform(pics[variant]))
            probabilities = []
            with torch.inference_mode():
                for start in range(0, len(tensors), args.batch):
                    x = torch.stack(tensors[start:start + args.batch]).to(device)
                    probabilities.extend(model(x).softmax(dim=1)[:, 1].cpu().tolist())
            predictions = dict(zip(names, probabilities))
            previous = row.get('positive_probability', '')
            if previous and abs(predictions['baseline'] - float(previous)) > 0.02:
                mismatch += 1
            if len(examples) < args.examples and (row['actual'] != row['predicted']):
                examples.append((path.name, pics, predictions))
        elif original_status == 'OK':
            missing += 1
        for variant in VARIANTS:
            p = predictions.get(variant)
            pred = ('UNKNOWN' if p is None else
                    'yawn' if p >= args.threshold else 'no_yawn')
            outputs.append(dict(image=str(path), actual=row['actual'], condition=row['condition'],
                                variant=variant, threshold=args.threshold, predicted=pred,
                                yawn_probability=round(p, 6) if p is not None else '',
                                status=('OK' if p is not None else
                                        'MISSING_ROI' if original_status == 'OK' else original_status),
                                roi_path=str(roi) if roi is not None and roi.is_file() else ''))
        if (k + 1) % 100 == 0:
            print(f'Processed {k+1}/{len(records)}')
    write_csv(args.output / 'predictions.csv', outputs)
    metrics = summarize(outputs, args.threshold)
    write_csv(args.output / 'summary.csv', metrics)
    make_examples(examples, args.output / 'example_variants.jpg', args.examples)
    print('Saved:', args.output)
    if mismatch:
        print(f'WARNING: baseline prediction differed >0.02 from 12_eval on {mismatch} images. Check checkpoint/transform.')
    if missing:
        print(f'WARNING: {missing} saved ROI images missing; treated as UNKNOWN. Run 12_eval with --save-roi.')
    print('NOTE: Saved mouth ROIs reused; MediaPipe was NOT rerun, so UNKNOWNs are unchanged.')
    print('NOTE: Roboflow is a development set; choose settings only after testing on new, unused data.')


if __name__ == '__main__':
    main()
