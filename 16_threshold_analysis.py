"""16_threshold_analysis.py
Existing prediction CSV threshold sweep. No model inference or retraining.
Run: python 16_threshold_analysis.py
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_rows(path):
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records, threshold, model, variant):
    n_closed = sum(r['actual'] == 'Closed' for r in records)
    n_open = sum(r['actual'] == 'Open' for r in records)
    tp = tn = fp = fn = roi_fail = errors = 0
    for r in records:
        actual = r['actual']
        if r['status'] != 'OK' or not r['closed_probability']:
            if r['status'] == 'ROI_FAIL':
                roi_fail += 1
            else:
                errors += 1
            if actual == 'Closed':
                fn += 1  # Unable to detect a true closed eye
            continue  # UNKNOWN on Open is not a correct classification
        closed = float(r['closed_probability']) >= threshold
        if actual == 'Closed':
            tp += int(closed)
            fn += int(not closed)
        else:
            fp += int(closed)
            tn += int(not closed)
    total = len(records)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_closed if n_closed else 0.0
    return {
        'model': model, 'variant': variant, 'threshold': f'{threshold:.3f}',
        'total': total, 'closed_total': n_closed, 'open_total': n_open,
        'accuracy': round((tp + tn) / total, 6),
        'closed_recall': round(recall, 6),
        'open_false_alarm_rate': round(fp / n_open, 6) if n_open else '',
        'closed_precision': round(precision, 6),
        'closed_f1': round(2 * precision * recall / (precision + recall), 6)
                     if precision + recall else 0.0,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn_including_unknown': fn,
        'roi_fail': roi_fail, 'errors': errors,
    }


def main():
    p = argparse.ArgumentParser(description='Sweep Closed thresholds using existing 15 ROI ablation predictions')
    p.add_argument('--input', type=Path, default=ROOT/'reports/cew/roi_ablation/predictions.csv')
    p.add_argument('--output', type=Path, default=ROOT/'reports/cew/threshold_analysis')
    p.add_argument('--model', default='yolo26n', help='Model name, or all')
    p.add_argument('--min', dest='minimum', type=float, default=0.05)
    p.add_argument('--max', dest='maximum', type=float, default=0.95)
    p.add_argument('--step', type=float, default=0.05)
    p.add_argument('--target-recall', type=float, default=0.90)
    args = p.parse_args()
    if not args.input.is_file():
        p.error(f'Prediction CSV not found: {args.input}')
    if not (0 < args.minimum <= args.maximum < 1 and args.step > 0):
        p.error('Expected 0 < --min <= --max < 1 and --step > 0')
    if not 0 <= args.target_recall <= 1:
        p.error('--target-recall must be between 0 and 1')

    records = read_rows(args.input)
    groups = defaultdict(list)
    for r in records:
        if r.get('actual') not in ('Closed', 'Open'):
            continue
        if args.model != 'all' and r.get('model') != args.model:
            continue
        groups[(r['model'], r['variant'])].append(r)
    if not groups:
        p.error('No matching prediction rows; check --model and CSV contents')

    thresholds = []
    value = args.minimum
    while value <= args.maximum + 1e-9:
        thresholds.append(round(value, 6))
        value += args.step
    if 0.5 not in thresholds:
        thresholds.append(0.5)
    thresholds = sorted(set(thresholds))
    rows = []
    for (model, variant), group in sorted(groups.items()):
        for threshold in thresholds:
            rows.append(summarize(group, threshold, model, variant))
    output = args.output / 'threshold_sweep.csv'
    save_rows(output, rows)

    print('CEW threshold comparison (ROI failures are counted as incorrect)')
    print(f"{'model':20} {'variant':9} {'thr':>5} {'acc':>8} {'recall':>8} {'open_FPR':>8} {'FP':>5} {'FN':>5}")
    for row in rows:
        if row['threshold'] in ('0.250', '0.300', '0.400', '0.500'):
            print(f"{row['model']:20} {row['variant']:9} {float(row['threshold']):5.2f} "
                  f"{row['accuracy']:8.2%} {row['closed_recall']:8.2%} "
                  f"{row['open_false_alarm_rate']:8.2%} {row['fp']:5} {row['fn_including_unknown']:5}")

    print(f'\nDevelopment candidates at Closed Recall >= {args.target_recall:.0%}:')
    for group in sorted(groups):
        qualified = [r for r in rows if (r['model'], r['variant']) == group
                     and r['closed_recall'] >= args.target_recall]
        if qualified:
            best = min(qualified, key=lambda r: (r['open_false_alarm_rate'], -r['accuracy']))
            print(f"{group[0]:20} {group[1]:9} threshold={best['threshold']} "
                  f"recall={best['closed_recall']:.2%} open_FPR={best['open_false_alarm_rate']:.2%}")
        else:
            print(f'{group}: no candidate in sweep')
    print('\nSaved:', output)
    print('CAUTION: This selects candidate settings on CEW. Validate on untouched, independent data before final deployment.')


if __name__ == '__main__':
    main()
