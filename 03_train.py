"""3단계: 학습 (YOLO classification, 작업 하나씩)

    python 03_train.py --task eyes
    python 03_train.py --task yawn
    python 03_train.py --task eyes --model yolo26s-cls.pt        # 다른 크기 모델 비교
    python 03_train.py --task eyes --data data/processed_random --name eyes_random   # 누수 방식과 비교

결과: runs/<name>/weights/best.pt, runs/<name>/results.csv (학습·검증 곡선)
"""
from pathlib import Path
import argparse

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["eyes", "yawn"], required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--model", default="yolo26n-cls.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--device", default=None, help="cpu 또는 0(GPU). 비우면 자동")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    data_dir = (ROOT / a.data / a.task).resolve()
    if not (data_dir / "train").exists():
        raise SystemExit(f"[오류] {data_dir / 'train'} 이 없습니다. 02_prepare.py를 먼저 실행하세요.")

    name = a.name or f"{a.task}_{Path(a.model).stem}"
    kw = dict(
        data=str(data_dir), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        patience=7, seed=42, workers=0, plots=True,
        project=str(ROOT / "runs"),  # 절대경로여야 runs/classify/runs/... 중첩이 안 생김
        name=name, exist_ok=True,
    )
    if a.device:
        kw["device"] = a.device

    model = YOLO(a.model)
    model.train(**kw)
    print("\n학습 완료. 가중치:", ROOT / "runs" / name / "weights" / "best.pt")
    print("다음: python 04_evaluate.py --task", a.task, "--weights", f"runs/{name}/weights/best.pt")


if __name__ == "__main__":
    main()
