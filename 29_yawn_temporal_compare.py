"""Compare time-aware yawn event filters on existing 28_nitymed_single_video_eval.py CSV.

No new model training, inference or alteration of original outputs.
For objective event-level metrics supply manually reviewed --yawns ranges.
NEVER claim model accuracy from unlabeled candidate intervals.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_IN = ROOT / "reports/nitymed_single_video/P1042778_720/predictions.csv"
DEFAULT_OUT = ROOT / "reports/nitymed_temporal_compare"


def load_samples(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"time_s", "roi_status", "p_yawn"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"필요한 CSV 열: {sorted(required)}, 실제: {reader.fieldnames}")
        rows = []
        for r in reader:
            time = float(r["time_s"])
            status = r["roi_status"].strip().upper()
            prob = (float(r["p_yawn"]) if status == "OK" and r["p_yawn"].strip() else None)
            if prob is not None and (not math.isfinite(prob) or not 0 <= prob <= 1):
                raise ValueError(f"하품 확률이 범위를 벗어남: {prob}")
            rows.append({"time": time, "prob": prob, "roi_status": status})
    rows.sort(key=lambda x: x["time"])
    if not rows:
        raise ValueError("CSV에 샘플이 없어")
    if len(rows) < 2:
        raise ValueError("최소 2개 시간 샘플이 필요해")
    gaps = [rows[i]["time"] - rows[i - 1]["time"] for i in range(1, len(rows))]
    if any(g <= 0 for g in gaps):
        raise ValueError("중복 또는 역순 시간 스탬프 발견")
    step = statistics.median(gaps)
    return rows, step


def parse_intervals(raw: str):
    intervals = []
    if not raw.strip():
        return intervals
    for piece in raw.split(","):
        a, sep, b = piece.strip().partition("-")
        if not sep:
            raise ValueError(f"시간 구간 형태가 잘못됐어: {piece} (예: 31.0-36.0)")
        start, end = float(a), float(b)
        if not (0 <= start < end):
            raise ValueError(f"시작/종료 시간이 잘못됐어: {piece}")
        intervals.append((start, end))
    intervals.sort()
    for prev, cur in zip(intervals, intervals[1:]):
        if cur[0] < prev[1]:
            raise ValueError(f"하품 정답 구간이 겹쳐: {prev}, {cur}")
    return intervals


def calculate_smoothed(rows, step, window):
    vals, smooth = [], []
    for i, r in enumerate(rows):
        if (r["prob"] is None or (i and r["time"] - rows[i - 1]["time"] > step * 1.8)):
            vals.clear()
        if r["prob"] is None:
            smooth.append(None)
        else:
            vals.append(r["prob"])
            if len(vals) > window:
                vals.pop(0)
            smooth.append(statistics.median(vals))
    return smooth


def find_events(rows, prob, step, threshold, min_duration):
    events = []
    start = last = None
    n = 0

    def finish():
        if start is not None:
            end = last + step
            if end - start + 1e-6 >= min_duration:
                events.append((round(start, 3), round(end, 3), n))

    for r, p in zip(rows, prob):
        t = r["time"]
        passed = p is not None and p >= threshold
        if not passed:
            finish()
            start = last = None
            n = 0
        elif start is None:
            start = last = t
            n = 1
        elif t - last <= step * 1.8:
            last = t
            n += 1
        else:
            finish()
            start = last = t
            n = 1
    finish()
    return events


def event_metrics(events, ground_truth, min_overlap):
    def overlap(a, b):
        return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

    matched_e = sum(any(overlap((e[0], e[1]), gt) >= min_overlap for gt in ground_truth)
                    for e in events)
    matched_gt = sum(any(overlap((e[0], e[1]), gt) >= min_overlap for e in events)
                     for gt in ground_truth)
    missed = len(ground_truth) - matched_gt
    false_events = len(events) - matched_e
    recall = matched_gt / len(ground_truth) if ground_truth else None
    precision = matched_e / len(events) if events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if recall and precision else 0.0
    return recall, precision, f1, matched_gt, missed, false_events


def save_csv(path, records):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        wr.writeheader()
        wr.writerows(records)


def main():
    ap = argparse.ArgumentParser(description="NITYMED ResNet18 시간 기반 하품 오탐 필터 비교")
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--yawns", default="", help='사람이 확인한 하품 구간, 예: "30.2-35.6,42-47"')
    ap.add_argument("--full-reviewed", action="store_true", help="영상 전체를 보고 하품 구간 전부 표기했을 때만 지정")
    ap.add_argument("--min-overlap", type=float, default=0.25)
    args = ap.parse_args()
    if not args.input.is_file():
        ap.error(f"CSV 파일이 없어: {args.input}\n먼저 python 28_nitymed_single_video_eval.py 실행")
    if args.min_overlap <= 0:
        ap.error("--min-overlap은 양수여야 해")
    try:
        rows, step = load_samples(args.input)
        gt = parse_intervals(args.yawns)
    except ValueError as e:
        ap.error(str(e))
    if args.full_reviewed and not gt:
        ap.error("--full-reviewed를 사용하려면 반드시 --yawns 정답 구간도 입력해야 해")

    settings = []
    all_events = []
    for window in (1, 3, 5):
        smooth = calculate_smoothed(rows, step, window)
        for threshold in (0.10, 0.15, 0.25, 0.50):
            for min_sec in (0.8, 1.2, 1.6, 2.0):
                events = find_events(rows, smooth, step, threshold, min_sec)
                r, p, f, hit, miss, fp = event_metrics(events, gt, args.min_overlap)
                setting = {
                    "window": window,
                    "threshold": threshold,
                    "min_duration_s": min_sec,
                    "candidate_events": len(events),
                    "actual_yawns_annotated": len(gt),
                    "event_recall": round(r, 5) if r is not None else "",
                    "event_precision": round(p, 5) if gt else "",
                    "event_f1": round(f, 5) if gt else "",
                    "detected_yawns": hit if gt else "",
                    "missed_yawns": miss if gt else "",
                    "other_candidate_events": fp if gt else "",
                    "annotation_complete": int(args.full_reviewed),
                }
                settings.append(setting)
                for a, b, n in events:
                    all_events.append({
                        "window": window, "threshold": threshold,
                        "min_duration_s": min_sec, "start_s": a,
                        "end_s": b, "positive_samples": n,
                    })

    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / "event_sweep.csv", settings)
    save_csv(args.output / "event_candidates.csv", all_events)

    print(f"[CSV] {args.input.resolve()}")
    print(f"[SAMPLES] {len(rows)} | 추론 간격 중앙값 {step:.3f}s | ROI 실패 {sum(r['prob'] is None for r in rows)}")
    print(f"[SETTINGS] {len(settings)}개 (window 3종 x threshold 4종 x min-duration 4종)")
    print("\n기준 설정 (기존 28번: window=3, threshold=0.15, min-duration~1.5s):")
    baseline = next(x for x in settings if x["window"] == 3 and x["threshold"] == 0.15 and x["min_duration_s"] == 1.6)
    print(f"  이벤트 후보 {baseline['candidate_events']}개 (1.6초 기준 근사)")
    if gt:
        title = "[전체 확인 완료]" if args.full_reviewed else "[일부 구간만 확인됨: 참고용]"
        print(f"\n{title} 하품 정답 구간: {gt}")
        if args.full_reviewed:
            rank = sorted(settings, key=lambda x: (-x["event_f1"], x["other_candidate_events"], -x["event_recall"]))
            print("  참고: 아래 순위는 이 영상에만 맞춘 개발용 결과로 새 영상에서 재검증 필요")
            for x in rank[:8]:
                print(f"  win={x['window']} thr={x['threshold']:.2f} dur={x['min_duration_s']:.1f}s "
                      f"event_recall={x['event_recall']:.0%} event_F1={x['event_f1']:.0%} "
                      f"false_candidates={x['other_candidate_events']}")
        else:
            print("  영상 전체 라벨이 아니므로 오탐·이벤트 F1을 신뢰할 수 없어. 전 구간 확인 후 --full-reviewed 사용해.")
    else:
        print("\n[정답 구간 없음] 수치는 모델 이벤트 후보 개수일 뿐 정확도나 재현율이 아님")
        for x in settings:
            if x["window"] == 3 and x["threshold"] in (.10, .15, .50) and x["min_duration_s"] == 1.2:
                print(f"  threshold {x['threshold']:.2f}: 후보 {x['candidate_events']}개 (window 3, 1.2s)")
    print("\nSaved:", args.output.resolve())
    print("주의: 영상 1개에서 고른 설정은 일반화 성능 개선 증거가 아니야.")


if __name__ == "__main__":
    main()
