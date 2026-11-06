"""Compare mouth ROI preprocessing x decision thresholds using experiment 20 CSV.

No inference, no training. UNKNOWN is counted as failure in end-to-end accuracy;
UNKNOWN yawn samples count as missed yawns in recall. The results are for
exploration on Roboflow data, NOT a final untouched test score.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = ('baseline', 'tight_85', 'soft_color', 'grayscale')


def div(a, b):
    return a / b if b else 0.0


def read_predictions(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    required = {'image', 'actual', 'variant', 'status', 'yawn_probability'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'CSV 열을 확인해줘. 필요한 열: {sorted(required)}')
    groups = defaultdict(dict)
    for r in rows:
        variant = r['variant']
        if variant not in VARIANTS:
            continue
        if r['actual'] not in ('yawn', 'no_yawn'):
            raise ValueError(f"예상하지 못한 라벨: {r['actual']}")
        # Unique image+variant: prevent accidental duplicate inflation.
        if variant in groups[r['image']]:
            raise ValueError(f"중복 행: {r['image']}, {variant}")
        groups[r['image']][variant] = r
    if not groups:
        raise ValueError('평가할 예측 행이 없어.')
    for image, records in groups.items():
        if set(records) != set(VARIANTS):
            raise ValueError(f'전처리별 행이 누락됨: {image} ({sorted(records)})')
        if len({r['actual'] for r in records.values()}) != 1:
            raise ValueError(f'전처리별 실제 라벨 불일치: {image}')
    return groups


def measure(records, threshold, variant):
    tp = tn = fp = fn = unknown_yawn = unknown_normal = 0
    valid = 0
    for image_rows in records.values():
        row = image_rows[variant]
        actual = row['actual']
        score = row['yawn_probability'].strip()
        if row['status'] != 'OK' or not score:
            if actual == 'yawn':
                unknown_yawn += 1
                fn += 1
            else:
                unknown_normal += 1
            continue
        probability = float(score)
        if not 0 <= probability <= 1:
            raise ValueError(f'확률 범위 오류: {probability}')
        valid += 1
        is_yawn = probability >= threshold
        if actual == 'yawn':
            if is_yawn: tp += 1
            else: fn += 1
        elif is_yawn:
            fp += 1
        else:
            tn += 1

    n_yawn = tp + fn
    n_normal = tn + fp + unknown_normal
    n_all = n_yawn + n_normal
    precision = div(tp, tp + fp)
    recall = div(tp, n_yawn)
    return {
        'variant': variant,
        'threshold': f'{threshold:.2f}',
        'total': n_all,
        'valid': valid,
        'unknown': unknown_yawn + unknown_normal,
        'unknown_yawn': unknown_yawn,
        'unknown_no_yawn': unknown_normal,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn_including_unknown': fn,
        'accuracy': round(div(tp + tn, n_all), 6),
        'valid_accuracy': round(div(tp + tn, valid), 6),
        'yawn_recall': round(recall, 6),
        'precision': round(precision, 6),
        'f1': round(div(2 * precision * recall, precision + recall), 6),
        'no_yawn_fpr': round(div(fp, n_normal), 6),
    }


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def show(title, rows):
    print('\n' + title)
    print(f"{'variant':13} {'thr':>5} {'acc':>8} {'recall':>8} {'precision':>10} {'F1':>8} {'FPR':>8} {'FP':>4} {'FN':>4} {'UNK':>4}")
    for r in rows:
        print(f"{r['variant']:13} {r['threshold']:>5} {r['accuracy']:>7.2%} "
              f"{r['yawn_recall']:>7.2%} {r['precision']:>9.2%} "
              f"{r['f1']:>7.2%} {r['no_yawn_fpr']:>7.2%} "
              f"{r['fp']:>4} {r['fn_including_unknown']:>4} {r['unknown']:>4}")


def main():
    ap = argparse.ArgumentParser(description='Experiment 21: yawn preprocessing x threshold sweep')
    ap.add_argument('--input', type=Path,
                    default=ROOT / 'reports/yawn_roboflow/preprocess_ablation/predictions.csv')
    ap.add_argument('--output', type=Path,
                    default=ROOT / 'reports/yawn_roboflow/combined_threshold')
    ap.add_argument('--max-fpr', type=float, default=0.15,
                    help='Exploratory candidate false-positive-rate cap (default 0.15)')
    ap.add_argument('--min-recall', type=float, default=0.60,
                    help='Exploratory candidate minimum yawn recall (default 0.60)')
    args = ap.parse_args()
    if not args.input.is_file():
        ap.error(f'파일을 찾을 수 없어: {args.input}')
    if not (0 <= args.max_fpr <= 1 and 0 <= args.min_recall <= 1):
        ap.error('제한값은 0~1 범위여야 해.')

    groups = read_predictions(args.input)
    results = [measure(groups, step / 100, variant)
               for variant in VARIANTS for step in range(5, 96, 5)]
    output_csv = args.output / 'combined_threshold_sweep.csv'
    save(output_csv, results)
    show('0.50 기준 결과 (20번 재현 확인)', [r for r in results if r['threshold'] == '0.50'])
    show('RGB vs Grayscale - 주요 임계값',
         [r for r in results if r['variant'] in ('baseline', 'grayscale')
          and r['threshold'] in ('0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.50')])

    # Optimize exploratory F1 with constraints, without claiming external generalization.
    candidates = [r for r in results
                  if r['no_yawn_fpr'] <= args.max_fpr and r['yawn_recall'] >= args.min_recall]
    candidates.sort(key=lambda r: (-r['f1'], -r['yawn_recall'], r['no_yawn_fpr'], r['variant']))
    if candidates:
        show(f'조건 충족 상위 후보 (FPR <= {args.max_fpr:.0%}, Recall >= {args.min_recall:.0%})', candidates[:8])
        save(args.output / 'candidates.csv', candidates)
    else:
        print(f'\nFPR <= {args.max_fpr:.0%}, Recall >= {args.min_recall:.0%} 조건을 충족하는 후보가 없어.')

    best_by_f1 = sorted(results, key=lambda r: (-r['f1'], r['no_yawn_fpr']))[:5]
    show('전체 F1 상위 5개 (참고용)', best_by_f1)
    print(f'\n이미지 수: {len(groups)}; 결과 저장: {output_csv}')
    print('주의: Roboflow 데이터에서 조정한 후보임. 새 미사용 데이터 및 실제 웹캠 이벤트로 검증 후 적용할 것.')


if __name__ == '__main__':
    main()
