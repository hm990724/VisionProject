"""CEW external error audit, no training and no model inference.

Reads reports/cew/external_predictions.csv made by 12_external_eval.py.
Shows failure overlap, classification confusion, ROI failures and contact sheets.
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
MODELS = ('yolo26n', 'resnet18', 'mobilenet_v3_small')


def load_rows(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def result_type(row):
    status = row.get('status', '')
    if status == 'ROI_FAIL':
        return 'ROI_FAIL'
    if status == 'ERROR':
        return 'ERROR'
    actual = row['actual']
    predicted = row['predicted']
    if actual == predicted:
        return 'CORRECT'
    if actual == 'Closed' and predicted == 'Open':
        return 'FN_CLOSED_AS_OPEN'
    if actual == 'Open' and predicted == 'Closed':
        return 'FP_OPEN_AS_CLOSED'
    return 'OTHER'


def save_csv(path, records, columns):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def thumb(path, size):
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert('RGB')
    image.thumbnail(size)
    canvas = Image.new('RGB', size, '#f0f4f8')
    canvas.paste(image, ((size[0]-image.width)//2, (size[1]-image.height)//2))
    return canvas


def make_sheet(records, output, roi_folder, max_items, index_by_path):
    if not records:
        return False
    records = records[:max_items]
    cell_w, cell_h = 620, 220
    sheet = Image.new('RGB', (cell_w, cell_h * len(records)), '#e8eef4')
    draw = ImageDraw.Draw(sheet)
    for j, entry in enumerate(records):
        img_path = Path(entry['image'])
        yy = j * cell_h
        draw.rectangle((0, yy, cell_w, yy + cell_h - 2), fill='#ffffff')
        try:
            sheet.paste(thumb(img_path, (245, 175)), (7, yy + 33))
        except (OSError, ValueError):
            draw.text((15, yy+65), 'IMAGE LOAD FAILED', fill='#a21d28')
        # ROI names assigned by 12_external_eval.py via sorted input ordering
        idx = index_by_path.get(entry['image'])
        roi = roi_folder / f'{idx:05d}_0.png' if idx is not None else None
        if roi and roi.is_file():
            try:
                sheet.paste(thumb(roi, (245, 175)), (261, yy + 33))
            except (OSError, ValueError):
                pass
        draw.text((8, yy + 8), img_path.name[:88], fill='#10283a')
        predicted = ', '.join(f'{model}={row["predicted"]}({result_type(row)})' for model, row in entry['models'].items())
        draw.text((10, yy + 209 - 12), ('Truth=' + entry['actual'] + '  ' + predicted)[:101], fill='#202c37')
    sheet.save(output)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=Path, default=ROOT / 'reports/cew/external_predictions.csv')
    p.add_argument('--output', type=Path, default=ROOT / 'reports/cew/error_audit')
    p.add_argument('--max-examples', type=int, default=24)
    p.add_argument('--roi-root', type=Path, default=ROOT / 'reports/cew/roi/eyes/cew')
    a = p.parse_args()
    if not a.csv.is_file():
        p.error(f'CSV not found: {a.csv}')
    if a.max_examples < 1:
        p.error('--max-examples must be >= 1')
    a.output.mkdir(parents=True, exist_ok=True)

    rows = [r for r in load_rows(a.csv) if r.get('task') == 'eyes']
    if not rows:
        p.error('No eyes records in CSV')
    by_image = {}
    for r in rows:
        key = r['image']
        if key not in by_image:
            by_image[key] = {'image': key, 'actual': r['actual'], 'models': {}}
        by_image[key]['models'][r['model']] = r
    # 12_external_eval indexes all images in sorted file-name order globally
    index_by_path = {key: i for i, key in enumerate(sorted(by_image))}

    print(f'Images: {len(by_image)}; predictions: {len(rows)}')
    model_summary = []
    for model in MODELS:
        rset = [r for r in rows if r['model'] == model]
        if not rset:
            continue
        c = Counter(result_type(r) for r in rset)
        summary = {'model': model, 'total': len(rset), 'correct': c['CORRECT'],
                   'roi_fail': c['ROI_FAIL'], 'error': c['ERROR'],
                   'closed_as_open': c['FN_CLOSED_AS_OPEN'],
                   'open_as_closed': c['FP_OPEN_AS_CLOSED'],
                   'other': c['OTHER']}
        model_summary.append(summary)
        print(model, summary)
    save_csv(a.output / 'model_error_breakdown.csv', model_summary,
             ['model', 'total', 'correct', 'roi_fail', 'error', 'closed_as_open', 'open_as_closed', 'other'])

    groups = defaultdict(list)
    detailed = []
    for entry in by_image.values():
        ms = entry['models']
        y = ms.get('yolo26n')
        r = ms.get('resnet18')
        if not y or not r:
            continue
        ty, tr = result_type(y), result_type(r)
        y_bad = ty != 'CORRECT'
        r_bad = tr != 'CORRECT'
        if y_bad and r_bad:
            label = 'both_fail'
        elif y_bad:
            label = 'yolo_only_fail'
        elif r_bad:
            label = 'resnet_only_fail'
        else:
            label = 'both_correct'
        groups[label].append(entry)
        detailed.append({'image': entry['image'], 'actual': entry['actual'],
                         'group': label, 'yolo_pred': y['predicted'],
                         'yolo_type': ty, 'resnet_pred': r['predicted'], 'resnet_type': tr,
                         'roi_status': y.get('status', '')})
    save_csv(a.output / 'pairwise_images.csv', detailed,
             ['image', 'actual', 'group', 'yolo_pred', 'yolo_type', 'resnet_pred', 'resnet_type', 'roi_status'])

    for name in ('both_fail', 'yolo_only_fail', 'resnet_only_fail', 'both_correct'):
        items = groups[name]
        print(f'{name}: {len(items)}')
        if name == 'both_correct':
            continue
        # Put ROI_FAIL first; make easier to spot upstream failures.
        ordered = sorted(items, key=lambda x: (x['models']['resnet18'].get('status') != 'ROI_FAIL', x['image']))
        # ROI is saved by class, and the subset condition folder name is 'cew'.
        # Make separate sheets for Closed/Open so ROI search uses correct class folder.
        for actual in ('Closed', 'Open'):
            subset = [x for x in ordered if x['actual'] == actual]
            roi_dir = a.roi_root / actual
            out = a.output / f'{name}_{actual}.png'
            if make_sheet(subset, out, roi_dir, a.max_examples, index_by_path):
                print('  sheet:', out)
    print('Saved:', a.output)
    print('NOTE: CSV accuracy includes ROI failures. Inspect contact sheets and ROI crops before claiming root causes.')


if __name__ == '__main__':
    main()
