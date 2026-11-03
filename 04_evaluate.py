"""4단계: 평가 + 과적합 점검

    python 04_evaluate.py --task eyes --weights runs/eyes_yolo26n-cls/weights/best.pt
    python 04_evaluate.py --task eyes --weights ... --extra data/webcam_test/eyes   # 내가 직접 찍은 테스트셋

보는 것
  - train / val / test 정확도 (train만 높고 test가 낮으면 과적합)
  - 클래스별 재현율(recall): 졸음 감지는 Closed, yawn을 놓치는 게 더 위험
  - 혼동행렬 (어떤 클래스를 무엇으로 착각하는지)
--extra 폴더는 <폴더>/<클래스이름>/이미지 구조여야 하고, 학습에 절대 쓰지 않은 데이터여야 한다.
"""
from pathlib import Path
from collections import Counter
import argparse
import csv

from PIL import Image
from ultralytics import YOLO

from common import EXT, to_square


def evaluate(model, split_dir, limit=None):
    conf = Counter()  # (정답, 예측) -> 개수
    for cdir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
        paths = sorted(p for p in cdir.iterdir() if p.suffix.lower() in EXT)
        if limit:
            paths = paths[:limit]
        for p in paths:
            with Image.open(p) as im:
                img = to_square(im)  # 학습 때와 같은 전처리
            r = model.predict(img, imgsz=224, verbose=False)[0]
            conf[(cdir.name, r.names[r.probs.top1])] += 1
    return conf


def report(title, conf, save_to=None):
    labels = sorted({a for a, _ in conf} | {p for _, p in conf})
    total = sum(conf.values())
    correct = sum(v for (a, p), v in conf.items() if a == p)
    print(f"\n[{title}] 정확도 {correct}/{total} = {correct / max(total, 1) * 100:.2f}%")
    print("  정답＼예측".ljust(14) + "".join(l.ljust(10) for l in labels) + "recall")
    for a in labels:
        row_total = sum(conf[(a, p)] for p in labels)
        rec = conf[(a, a)] / row_total * 100 if row_total else float("nan")
        print(f"  {a:12s}" + "".join(str(conf[(a, p)]).ljust(10) for p in labels) + f"{rec:.1f}%")
    if save_to:
        with save_to.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["actual\\pred"] + labels)
            for a in labels:
                w.writerow([a] + [conf[(a, p)] for p in labels])
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["eyes", "yawn"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--extra", default=None, help="직접 찍은 테스트셋 폴더")
    ap.add_argument("--train-limit", type=int, default=300, help="train은 클래스당 이 개수만 샘플 평가")
    a = ap.parse_args()

    model = YOLO(a.weights)
    print("모델 클래스:", model.names)
    out = Path("reports")
    out.mkdir(exist_ok=True)
    tag = Path(a.weights).parent.parent.name

    acc = {}
    for split in ("train", "val", "test"):
        d = Path(a.data) / a.task / split
        if d.exists():
            conf = evaluate(model, d, limit=a.train_limit if split == "train" else None)
            acc[split] = report(f"{split}", conf, out / f"{tag}_{split}_confusion.csv")
    if a.extra:
        acc["extra"] = report("직접 찍은 테스트셋", evaluate(model, Path(a.extra)), out / f"{tag}_extra_confusion.csv")

    print("\n========== 과적합 점검 ==========")
    print({k: f"{v * 100:.1f}%" for k, v in acc.items()})
    if "train" in acc and "val" in acc and acc["train"] - acc["val"] > 0.05:
        print("[주의] train이 val보다 5%p 넘게 높음 -> 과적합 의심 (데이터 추가, 증강, 에폭 축소)")
    if "test" in acc and "extra" in acc and acc["test"] - acc["extra"] > 0.10:
        print("[주의] test는 높은데 직접 찍은 데이터는 훨씬 낮음 -> 데이터 분포 차이(안경, 조명, 카메라 각도) 문제")
    print(f"혼동행렬 CSV 저장 위치: {out.resolve()}")


if __name__ == "__main__":
    main()
