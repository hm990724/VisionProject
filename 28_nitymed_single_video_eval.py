"""Analyze ONE NITYMED yawn video using the original teammate ResNet18.

Outputs an annotated demonstration video, probability timeline, ROI contact sheet,
per-sample CSV and a short report. Model predictions are NOT ground-truth labels.
No retraining or mutation of existing weights/datasets is performed.

Requirements: cv2, torch, pillow, matplotlib, numpy, existing
24_nitymed_prepare.py + teammate_yawn checkpoints and Face Landmarker model.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
from bisect import bisect_right
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = ROOT / "data/yawn_mp/Yawning/Female/HDTV720/P1042778_720.mp4"
DEFAULT_OUT = ROOT / "reports/nitymed_single_video/P1042778_720"


def load_team_components(folder: Path):
    helper = ROOT / "24_nitymed_prepare.py"
    if not helper.is_file():
        raise FileNotFoundError(f"24번 코드가 필요해: {helper}")
    spec = importlib.util.spec_from_file_location("nitymed_single_helper_24", helper)
    if spec is None or spec.loader is None:
        raise RuntimeError("24번 코드를 불러오지 못했어")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_team(folder)


def smooth_predictions(rows, window: int):
    recent = []
    for row in rows:
        if row["roi_status"] == "OK" and row["p_yawn"] is not None:
            recent.append(row["p_yawn"])
            recent = recent[-window:]
            row["p_smoothed"] = statistics.median(recent)
        else:
            recent.clear()   # don't carry probability through a missing face/ROI
            row["p_smoothed"] = None


def find_events(rows, threshold: float, min_seconds: float, sample_every: float):
    """Group consecutive positive samples, requiring temporal continuity.

    These are *model candidate intervals*, not accuracy/true-yawn events.
    """
    events = []
    start = last = None
    samples = 0

    def flush():
        if start is None:
            return
        estimated_end = last + sample_every
        duration = estimated_end - start
        if duration + 1e-9 >= min_seconds:
            events.append({
                "start_s": round(start, 2),
                "end_s": round(estimated_end, 2),
                "duration_s": round(duration, 2),
                "positive_samples": samples,
            })

    for row in rows:
        sec = row["time_s"]
        probability = row.get("p_smoothed")
        positive = probability is not None and probability >= threshold
        if positive:
            if start is None:
                start = last = sec
                samples = 1
            elif sec - last <= sample_every * 1.85:
                last = sec
                samples += 1
            else:
                flush()
                start = last = sec
                samples = 1
        else:
            flush()
            start = last = None
            samples = 0
    flush()
    return events


def analyze_video(video: Path, team_dir: Path, output: Path, every: float,
                  smooth_window: int, max_side: int):
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    model, extractor, transform, device = load_team_components(team_dir)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        extractor.close()
        raise RuntimeError(f"OpenCV 영상 열기 실패: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps < 1:
        cap.release()
        extractor.close()
        raise RuntimeError("영상 FPS를 읽지 못했어")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames else 0.0

    output.mkdir(parents=True, exist_ok=True)
    rows, examples = [], []
    frame_index = 0
    last_sample_t = -every
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            sec = frame_index / fps
            if sec - last_sample_t + 1e-9 < every:
                frame_index += 1
                continue
            last_sample_t = sec
            h, w = frame.shape[:2]
            if max_side and max(h, w) > max_side:
                scale = max_side / max(h, w)
                frame_in = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            else:
                frame_in = frame
            face = Image.fromarray(cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB))
            roi = None
            error = ""
            p_yawn = None
            status = "ROI_FAIL"
            try:
                result = extractor.extract(face)
                status = "OK" if str(result.get("status", "")).lower() == "ok" else "ROI_FAIL"
                roi = result.get("crop") if status == "OK" else None
                if roi is not None:
                    with torch.inference_mode():
                        batch = transform(roi).unsqueeze(0).to(device)
                        logits = model(batch)
                        p_yawn = float(logits.softmax(1)[0, 1].item())
                else:
                    status = "ROI_FAIL"
            except Exception as exc:
                status = "ERROR"
                error = f"{type(exc).__name__}: {exc}"[:250]
            entry = {"time_s": round(sec, 3), "frame": frame_index,
                     "roi_status": status, "p_yawn": p_yawn, "error": error}
            rows.append(entry)
            if len(examples) < 400:
                examples.append((sec, face.copy(), roi.copy() if roi is not None else None, p_yawn))
            if len(rows) % 20 == 0:
                print(f"  분석 {sec:5.1f}s / 약 {duration:.1f}s | ROI 성공 {sum(r['roi_status']=='OK' for r in rows)}/{len(rows)}")
            frame_index += 1
    finally:
        cap.release()
        extractor.close()

    if not rows:
        raise RuntimeError("읽을 수 있는 영상 프레임이 없어")
    smooth_predictions(rows, smooth_window)
    return rows, examples, {"fps": fps, "video_frames": frame_index,
                            "duration_s": frame_index / fps}


def write_csv(rows, dest: Path, thresholds):
    fieldnames = ["time_s", "frame", "roi_status", "p_yawn", "p_smoothed"]
    fieldnames += [f"candidate_at_{t:.2f}" for t in thresholds]
    fieldnames.append("error")
    with dest.open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            item = {k: r.get(k, "") for k in fieldnames}
            item["p_yawn"] = "" if r["p_yawn"] is None else f"{r['p_yawn']:.6f}"
            item["p_smoothed"] = "" if r["p_smoothed"] is None else f"{r['p_smoothed']:.6f}"
            for t in thresholds:
                item[f"candidate_at_{t:.2f}"] = (
                    "UNKNOWN" if r["p_smoothed"] is None
                    else int(r["p_smoothed"] >= t)
                )
            wr.writerow(item)


def plot_timeline(rows, output: Path, events, threshold: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [r["time_s"] for r in rows]
    raw = [float("nan") if r["p_yawn"] is None else r["p_yawn"] for r in rows]
    smooth = [float("nan") if r["p_smoothed"] is None else r["p_smoothed"] for r in rows]
    fig, ax = plt.subplots(figsize=(13, 4.5), dpi=150)
    ax.plot(xs, raw, color="#8f9eb2", linewidth=1.0, marker=".", markersize=2.4, alpha=0.75, label="Raw p(yawn)")
    ax.plot(xs, smooth, color="#1771bc", linewidth=2.0, label="Smoothed p(yawn)")
    ax.axhline(threshold, color="#d55c29", linestyle="--", linewidth=1.4, label=f"Threshold {threshold:.2f}")
    for j, event in enumerate(events):
        ax.axvspan(event["start_s"], event["end_s"], color="#f4c65e", alpha=.3,
                   label="Candidate event" if j == 0 else None)
    failed = [r["time_s"] for r in rows if r["roi_status"] != "OK"]
    if failed:
        ax.scatter(failed, [0]*len(failed), s=13, color="#bf3a3a", label="ROI unknown", zorder=4)
    ax.set(xlabel="Video time (seconds)", ylabel="Model probability", title="NITYMED: ResNet18 mouth ROI / single-video case study")
    ax.set_ylim(-0.04, 1.05)
    ax.set_xlim(0, max(xs[-1] + .5, 1))
    ax.grid(alpha=.16)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def save_contact_sheet(examples, dest: Path, max_items=12):
    from PIL import Image, ImageOps, ImageDraw, ImageFont
    if not examples:
        return
    if len(examples) > max_items:
        ids = [round(i*(len(examples)-1)/(max_items-1)) for i in range(max_items)]
        examples = [examples[j] for j in ids]
    cw, ch, cols = 340, 226, 3
    nrows = math.ceil(len(examples)/cols)
    sheet = Image.new("RGB", (cw*cols, ch*nrows), "#f0f4f7")
    d = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for k, (sec, face, roi, p) in enumerate(examples):
        x, y = (k % cols)*cw, (k // cols)*ch
        d.rounded_rectangle((x+4,y+4,x+cw-4,y+ch-4), fill="white", radius=10)
        fimg = ImageOps.contain(face, (182, 170))
        sheet.paste(fimg, (x+8+(182-fimg.width)//2, y+38+(170-fimg.height)//2))
        if roi is not None:
            rimg = ImageOps.contain(roi, (137, 128))
            sheet.paste(rimg, (x+195+(137-rimg.width)//2, y+58+(128-rimg.height)//2))
        else:
            d.text((x+204,y+98), "ROI UNKNOWN", fill="#aa4444", font=font)
        label = f"t={sec:.1f}s  p(yawn)=" + (f"{p:.3f}" if p is not None else "UNKNOWN")
        d.text((x+14,y+15), label, fill="#12314c", font=font)
        d.text((x+14,y+205), "Original frame  |  MediaPipe mouth ROI", fill="#43566a", font=font)
    sheet.save(dest, quality=92)


def write_report(output, video: Path, info, rows, events_by_thr, threshold, every):
    good = sum(r["roi_status"] == "OK" for r in rows)
    total = len(rows)
    total_gt = None
    lines = [
        "# NITYMED 단일 영상 정성 검증", "",
        f"- 영상: `{video.name}`",
        f"- 길이: {info['duration_s']:.2f}초 / 원본 FPS: {info['fps']:.2f}",
        f"- ROI 추출: {good}/{total} ({good/total:.1%}); 실패·오류 {total-good}회",
        f"- 분석 간격: 약 {every:.2f}초, 움직이는 중앙값 기준",
        "- 모델: 기존 팀원 ResNet18 + MediaPipe 입 ROI (재학습 없음)",
        "", "## 모델이 하품으로 판단한 연속 구간 (정답 아님)", "",
        "| 판정 임계값 | 지속 조건을 만족한 후보 구간 수 | 시작~종료(초) |",
        "|---:|---:|---|",
    ]
    for t, ev in events_by_thr.items():
        strs = ", ".join(f"{x['start_s']:.1f}–{x['end_s']:.1f}" for x in ev) or "없음"
        lines.append(f"| {t:.2f} | {len(ev)} | {strs} |")
    lines += [
        "", "## 발표 자료에 적을 수 있는 내용", "",
        "- NITYMED 운전자 영상 107개 중 단일 영상에 대해 기존 모델의 입 ROI 추출 및 시간별 하품 확률을 시각화함.",
        "- 임계값에 따른 모델의 하품 의심 구간과 ROI 실패 여부를 확인함.",
        "- 하품 정답 시간 구간을 사람에게 검수받지 않았으므로 **정확도·재현율·F1을 계산하거나 개선됐다고 주장하지 않음**.",
        "- 이 영상은 사례 제시용이다. 107개 영상 전체 정량 검증으로 표현하지 않음.",
        "", "## 시각 자료", "",
        "- `probability_timeline.png`: 하품 확률 추이",
        "- `face_mouth_contact_sheet.jpg`: 원본 얼굴과 추출된 입 ROI",
        "- `annotated_demo.mp4`: 모델 예측 상태를 오버레이한 시연용 영상 (생성 옵션 사용 시)",
        "- `predictions.csv`: 프레임 샘플별 원시 확률·평활화 확률·ROI 성공 여부",
        "", "## 한계", "",
        "- 동영상의 실제 하품 구간을 직접 확인해 정답과 대조해야 감지 성공/실패를 판단할 수 있음.",
        "- 주름/입술 색상과 오류 사이의 인과관계는 본 단일 영상으로 검증되지 않음.",
        "- 실제 운전 중 위험 경고 및 119 연동 성능과 무관한 오프라인 사례 분석임.",
    ]
    (output/"case_study.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def overlay_video(video: Path, dest: Path, rows, examples, threshold: float, events,
                  max_width: int = 960):
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ok, first = cap.read()
    if not ok or not math.isfinite(fps) or fps <= 0:
        cap.release()
        print("[WARN] 데모 영상 렌더링 생략: 읽기 불가")
        return
    h, w = first.shape[:2]
    scale = min(1.0, max_width / w) if max_width else 1.0
    width, height = max(2, int(w*scale)//2*2), max(2, int(h*scale)//2*2)
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    if not writer.isOpened():
        cap.release()
        print("[WARN] 데모 영상 생성 불가: MP4 코덱이 지원되지 않음")
        return
    times = [r["time_s"] for r in rows]
    idx = 0
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_no = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            sec = frame_no / fps
            frame = cv2.resize(frame, (width,height), interpolation=cv2.INTER_AREA)
            idx = max(0, bisect_right(times, sec) - 1)
            row = rows[idx]
            valid = row["p_smoothed"] is not None and sec - row["time_s"] <= 1.5
            p = row["p_smoothed"] if valid else None
            in_event = any(x["start_s"] <= sec < x["end_s"] for x in events)
            status = "YAWN CANDIDATE" if in_event else ("ROI UNKNOWN" if p is None else "MONITORING")
            accent = (38, 126, 239) if in_event else ((60,60,200) if p is None else (91, 211, 163))
            cv2.rectangle(frame, (0,0), (width, 100), (13,27,41), -1)
            cv2.putText(frame, "AURA DRIVE / NITYMED CASE STUDY", (20,31), cv2.FONT_HERSHEY_SIMPLEX, .65, (240,240,240), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{sec:05.1f}s  |  {status}", (20,66), cv2.FONT_HERSHEY_SIMPLEX, .6, accent, 2, cv2.LINE_AA)
            cv2.putText(frame, f"P(yawn): {'--' if p is None else f'{p:.2%}'}  /  Threshold {threshold:.2f}", (20,92), cv2.FONT_HERSHEY_SIMPLEX, .5, (238,238,238), 1, cv2.LINE_AA)
            # Inset the actual MediaPipe crop used for classification; this is
            # a visualization only, and never drawn onto the model's input.
            if idx < len(examples):
                roi = examples[idx][2]
                if roi is not None and width >= 360 and height >= 270:
                    from PIL import Image
                    thumb = cv2.cvtColor(__import__('numpy').array(roi.convert("RGB")), cv2.COLOR_RGB2BGR)
                    tile_w, tile_h = min(174, width//3), min(118, height//3)
                    tile_x, tile_y = width-tile_w-18, height-tile_h-73
                    thumb = cv2.resize(thumb, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                    cv2.rectangle(frame, (tile_x-5,tile_y-22), (tile_x+tile_w+5,tile_y+tile_h+5), (13,27,41), -1)
                    frame[tile_y:tile_y+tile_h, tile_x:tile_x+tile_w] = thumb
                    cv2.putText(frame, "MOUTH ROI", (tile_x,tile_y-6),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, (235,246,247), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (0, height-34), (width,height), (13,27,41), -1)
            cv2.putText(frame, "AI prediction, not medically validated / demo only", (16,height-11), cv2.FONT_HERSHEY_SIMPLEX, .43, (210,224,232), 1, cv2.LINE_AA)
            writer.write(frame)
            frame_no += 1
    finally:
        cap.release()
        writer.release()
    print(f"[SAVED] 시연 영상: {dest}")


def main():
    parser = argparse.ArgumentParser(description="단일 NITYMED 영상 원본 ResNet18 하품 정성 검증")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--team-dir", type=Path, default=ROOT/"teammate_yawn")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--every", type=float, default=0.4, help="추론 샘플 간격(초)")
    parser.add_argument("--smooth-window", type=int, default=3, help="중앙값 평활화 샘플 수")
    parser.add_argument("--threshold", type=float, default=0.15, help="시연 화면 임계값, 0.10/0.15/0.50 추가 비교")
    parser.add_argument("--min-duration", type=float, default=1.5, help="연속 하품 후보 최소 지속 시간(초)")
    parser.add_argument("--max-side", type=int, default=960, help="추론 입력용 원본 프레임 최대 변(기본 960)")
    parser.add_argument("--no-video", action="store_true", help="annotated MP4 렌더링 생략")
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"영상 없음: {args.video}\n  데이터 위치를 확인해 줘")
    if args.every <= 0 or args.smooth_window < 1 or args.min_duration <= 0:
        parser.error("--every / --smooth-window / --min-duration 값은 양수여야 해")
    if not 0 < args.threshold < 1:
        parser.error("--threshold는 0과 1 사이여야 해")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"결과 폴더가 비어 있지 않아 (기존 결과 보호): {args.output}\n  --output으로 새 폴더를 지정해 줘")
    print(f"[VIDEO] {args.video}")
    print("[INFO] 기존 ResNet18 유지 / 재학습 없음 / 모델 예측 ≠ 사람 정답")
    rows, examples, info = analyze_video(args.video, args.team_dir, args.output,
                                         args.every, args.smooth_window, args.max_side)
    thresholds = sorted({0.10, 0.15, 0.50, args.threshold})
    events_by_threshold = {t: find_events(rows, t, args.min_duration, args.every) for t in thresholds}
    write_csv(rows, args.output/"predictions.csv", thresholds)
    plot_timeline(rows, args.output/"probability_timeline.png", events_by_threshold[args.threshold], args.threshold)
    save_contact_sheet(examples, args.output/"face_mouth_contact_sheet.jpg")
    write_report(args.output, args.video, info, rows, events_by_threshold, args.threshold, args.every)
    (args.output/"run_summary.json").write_text(json.dumps({
        "video": str(args.video), "sample_count": len(rows),
        "roi_success_count": sum(r["roi_status"]=="OK" for r in rows),
        "params": {"every": args.every, "threshold": args.threshold,
                   "min_duration": args.min_duration, "smooth_window": args.smooth_window},
        "candidate_events_by_threshold": {f"{k:.2f}": v for k,v in events_by_threshold.items()},
        "warning": "Unlabeled single-video case study; no accuracy claim."
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_video:
        overlay_video(args.video, args.output/"annotated_demo.mp4", rows, examples,
                      args.threshold, events_by_threshold[args.threshold])
    print("\n=== SUMMARY ===")
    print(f"Samples {len(rows)} | ROI OK {sum(r['roi_status']=='OK' for r in rows)}/{len(rows)}")
    for t, ev in events_by_threshold.items():
        print(f"Threshold {t:.2f}: candidate events {len(ev)}", [f"{x['start_s']}-{x['end_s']}s" for x in ev])
    print(f"Saved: {args.output.resolve()}")
    print("NOTE: 감지 이벤트는 정답 검수 전의 모델 후보일 뿐 정확도/재현율이 아님.")


if __name__ == "__main__":
    main()
