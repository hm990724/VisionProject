"""Audit external yawn classification errors with original faces and saved mouth ROIs.

Requires only Pillow and Python standard library. No inference or retraining.
The ROI filenames are reconstructed from the SAME sorting convention as
12_external_eval.py: sorted(data/yawn.rglob('*')), indexed from 0.
"""
import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
GROUPS = ('missed_yawn', 'false_alarm', 'roi_fail', 'error', 'correct')


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if r.get('task') == 'yawn' and r.get('model') == 'team_resnet18']


def index_source_images(data):
    """Map normalized absolute source paths to 12_external_eval.py row indices."""
    source = data.resolve() / 'yawn'
    paths = sorted(
        p for p in source.rglob('*')
        if p.is_file() and p.suffix.lower() in EXT
        and p.parent.name in ('yawn', 'no_yawn')
    ) if source.is_dir() else []
    return {str(p.resolve()).casefold(): i for i, p in enumerate(paths)}


def category(r):
    status = r.get('status', '')
    if status == 'ROI_FAIL':
        return 'roi_fail'
    if status != 'OK':
        return 'error'
    actual, pred = r.get('actual'), r.get('predicted')
    if actual == 'yawn' and pred == 'no_yawn':
        return 'missed_yawn'
    if actual == 'no_yawn' and pred == 'yawn':
        return 'false_alarm'
    if actual == pred:
        return 'correct'
    return 'error'


def square_thumbnail(path, width=256, height=190):
    canvas = Image.new('RGB', (width, height), '#edf0f3')
    if not path or not Path(path).is_file():
        return canvas
    try:
        with Image.open(path) as obj:
            img = ImageOps.exif_transpose(obj).convert('RGB')
            img.thumbnail((width - 12, height - 12), Image.Resampling.LANCZOS)
        canvas.paste(img, ((width-img.width)//2, (height-img.height)//2))
    except (OSError, ValueError):
        pass
    return canvas


def render_pages(rows, group_name, out, per_page):
    if not rows:
        return
    width, height = 600, 254
    font = ImageFont.load_default()
    for start in range(0, len(rows), per_page):
        batch = rows[start:start + per_page]
        sheet = Image.new('RGB', (width, height * len(batch)), 'white')
        draw = ImageDraw.Draw(sheet)
        for pos, r in enumerate(batch):
            y = pos * height
            draw.rectangle((0, y, width - 1, y + height - 1), outline='#c7d5df', width=2)
            name = Path(r['image']).name
            draw.text((10, y + 8), name[:87], fill='#162f46', font=font)
            sheet.paste(square_thumbnail(r['image']), (10, y + 30))
            sheet.paste(square_thumbnail(r['roi_image']), (315, y + 30))
            draw.text((10, y + 222), 'FULL FACE', fill='#526273', font=font)
            draw.text((315, y + 222), 'MOUTH ROI', fill='#526273', font=font)
            desc = f"GT={r['actual']} | Pred={r['predicted']} | P(yawn)={r['positive_probability'] or 'N/A'} | {r['status']}"
            draw.text((10, y + 237), desc[:95], fill='#273c4d', font=font)
        filename = out / f'{group_name}_page_{start // per_page + 1:02d}.jpg'
        sheet.save(filename, quality=90)
        print('  sheet:', filename)


def main():
    p = argparse.ArgumentParser(description='Yawn external error analysis and face/ROI sheets')
    p.add_argument('--data', type=Path, default=ROOT / 'data/external_driver',
                   help='The SAME --data used with 12_external_eval.py')
    p.add_argument('--input', type=Path, default=ROOT / 'reports/yawn_roboflow/external_predictions.csv')
    p.add_argument('--roi-root', type=Path, default=ROOT / 'reports/yawn_roboflow/roi')
    p.add_argument('--output', type=Path, default=ROOT / 'reports/yawn_roboflow/error_audit')
    p.add_argument('--max-examples', type=int, default=24, help='Maximum preview samples per category')
    p.add_argument('--per-page', type=int, default=12)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if not args.input.is_file():
        p.error(f'Predictions CSV not found: {args.input}')
    if args.max_examples < 0 or args.per_page < 1:
        p.error('--max-examples must be >= 0 and --per-page >= 1')

    rows = read_rows(args.input)
    if not rows:
        p.error('No yawn/team_resnet18 predictions found in CSV')
    path_index = index_source_images(args.data)
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    groups = {g: [] for g in GROUPS}
    for r in rows:
        group = category(r)
        source = Path(r['image'])
        index = path_index.get(str(source.resolve()).casefold())
        roi = (args.roi_root / 'yawn' / r['condition'] / r['actual'] / f'{index:05d}.png') if index is not None else None
        record = {
            'group': group,
            'image': r['image'],
            'roi_image': str(roi) if roi and roi.exists() else '',
            'actual': r['actual'],
            'predicted': r['predicted'],
            'positive_probability': r.get('positive_probability', ''),
            'status': r.get('status', ''),
            'condition': r.get('condition', ''),
            'error': r.get('error', ''),
        }
        records.append(record)
        groups[group].append(record)

    out_csv = args.output / 'all_cases.csv'
    with out_csv.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    tally = Counter(r['group'] for r in records)
    actuals = Counter(r['actual'] for r in records)
    roi_yawn = sum(r['actual'] == 'yawn' for r in groups['roi_fail'])
    roi_normal = sum(r['actual'] == 'no_yawn' for r in groups['roi_fail'])
    print(f"Images: {len(rows)} | yawn={actuals['yawn']} | no_yawn={actuals['no_yawn']}")
    for group in GROUPS:
        print(f'{group}: {tally[group]}')
    print(f'ROI failures: yawn={roi_yawn}, no_yawn={roi_normal}')
    print('ROI files linked:', sum(bool(r['roi_image']) for r in records))

    stats = [{'group': g, 'count': tally[g]} for g in GROUPS]
    with (args.output / 'breakdown.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['group', 'count'])
        writer.writeheader()
        writer.writerows(stats)

    rng = random.Random(args.seed)
    for group in ('missed_yawn', 'false_alarm', 'roi_fail'):
        candidates = groups[group][:]
        rng.shuffle(candidates)
        if args.max_examples:
            render_pages(candidates[:args.max_examples], group, args.output, args.per_page)

    if not path_index:
        print('WARNING: No labeled images found at --data/yawn; ROI previews may be missing.')
    elif any(not r['roi_image'] for r in groups['missed_yawn'] + groups['false_alarm']):
        print('WARNING: Some model-error ROI images missing. Verify --data and --roi-root match your 12 run.')

    print('Saved:', out_csv)
    print('Saved:', args.output / 'breakdown.csv')
    print('NOTE: Review image labels and source overlap before drawing conclusions.')


if __name__ == '__main__':
    main()
