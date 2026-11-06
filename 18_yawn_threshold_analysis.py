
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def safe_div(a, b):
    return a / b if b else 0.0


def evaluate(rows, threshold):
    tp = tn = fp = fn = unknown = 0
    roi_fail_yawn = 0
    roi_fail_normal = 0

    for row in rows:
        actual = row["actual"]

        if (
            row["status"] != "OK"
            or not row["positive_probability"]
        ):
            unknown += 1

            if actual == "yawn":
                fn += 1
                roi_fail_yawn += 1
            else:
                roi_fail_normal += 1

            continue

        probability = float(
            row["positive_probability"]
        )

        predicted = (
            "yawn"
            if probability >= threshold
            else "no_yawn"
        )

        if actual == "yawn":
            if predicted == "yawn":
                tp += 1
            else:
                fn += 1
        else:
            if predicted == "yawn":
                fp += 1
            else:
                tn += 1

    total = len(rows)
    valid = total - unknown

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(
        2 * precision * recall,
        precision + recall,
    )

    return {
        "threshold": round(threshold, 2),
        "total": total,
        "valid": valid,
        "unknown": unknown,
        "accuracy": round(
            safe_div(tp + tn, total), 5
        ),
        "detected_accuracy": round(
            safe_div(tp + tn, valid), 5
        ),
        "yawn_recall": round(recall, 5),
        "yawn_precision": round(precision, 5),
        "yawn_f1": round(f1, 5),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn_including_unknown": fn,
        "roi_fail_yawn": roi_fail_yawn,
        "roi_fail_normal": roi_fail_normal,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            ROOT
            / "reports/yawn_roboflow/"
            "external_predictions.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "reports/yawn_roboflow/"
            "threshold_analysis"
        ),
    )

    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(
            f"예측 CSV가 없어: {args.input}"
        )

    with args.input.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    rows = [
        row for row in rows
        if row["task"] == "yawn"
        and row["model"] == "team_resnet18"
    ]

    if not rows:
        raise SystemExit(
            "평가할 하품 예측 결과가 없어."
        )

    results = []

    print()
    print(
        "Threshold | Accuracy | Recall | "
        "Precision | F1 | FP | FN"
    )
    print("-" * 69)

    for i in range(5, 96, 5):
        threshold = i / 100
        result = evaluate(rows, threshold)

        results.append(result)

        print(
            f"{threshold:9.2f} | "
            f"{result['accuracy']:7.2%} | "
            f"{result['yawn_recall']:6.2%} | "
            f"{result['yawn_precision']:9.2%} | "
            f"{result['yawn_f1']:6.2%} | "
            f"{result['fp']:2} | "
            f"{result['fn_including_unknown']:2}"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    dest = (
        args.output
        / "yawn_threshold_sweep.csv"
    )

    with dest.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0]),
        )
        writer.writeheader()
        writer.writerows(results)

    print()
    print("Saved:", dest)
    print(
        "ROI 검출 실패는 UNKNOWN으로 기록하고 "
        "전체 정확도에서는 실패로 반영함."
    )
    print(
        "이 결과는 Roboflow 개발용 데이터 기준. "
        "최종 설정은 새로운 데이터에서 검증 필요."
    )


if __name__ == "__main__":
    main()
