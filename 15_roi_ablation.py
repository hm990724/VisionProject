
"""15_roi_ablation.py - CEW eye preprocessing comparison."""

import argparse
import csv
import importlib
import random
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageOps

from common import to_square
from roi_utils import crop_eyes


ROOT = Path(__file__).resolve().parent
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

MODEL_NAMES = [
    "yolo26n",
    "resnet18",
    "mobilenet_v3_small",
]

VARIANTS = [
    "baseline",
    "tight_85",
    "clahe",
]


def load_models(names):
    loaded = {}
    cnn_module = None

    for name in names:
        if name == "yolo26n":
            from ultralytics import YOLO

            path = (
                ROOT
                / "runs/eyes_yolo26n-cls/weights/best.pt"
            )

            if not path.is_file():
                raise FileNotFoundError(path)

            network = YOLO(str(path))

            closed_idx = next(
                (
                    int(k)
                    for k, value in network.names.items()
                    if str(value).lower() == "closed"
                ),
                None,
            )

            if closed_idx is None:
                raise ValueError(
                    f"Closed 클래스 없음: {network.names}"
                )

            loaded[name] = (
                "yolo",
                network,
                closed_idx,
            )

        else:
            if cnn_module is None:
                cnn_module = importlib.import_module(
                    "08_train_cnn"
                )

            path = (
                ROOT
                / "runs"
                / f"eyes_{name}"
                / "best.pt"
            )

            if not path.is_file():
                raise FileNotFoundError(path)

            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

            if list(checkpoint["classes"]) != [
                "Closed",
                "Open",
            ]:
                raise ValueError(
                    f"클래스 순서 불일치: {path}"
                )

            network = cnn_module.make_model(
                name,
                pretrained=False,
            )

            network.load_state_dict(
                checkpoint["model"]
            )

            network.eval()

            loaded[name] = (
                "cnn",
                network,
                cnn_module.get_transforms(False),
            )

    return loaded


def make_variants(eye):
    """
    기존 MediaPipe ROI를 기준으로
    세 가지 전처리를 생성한다.
    """
    original = eye.convert("RGB")

    width, height = original.size

    # 양쪽 가장자리 7.5%씩 제거
    dx = round(width * 0.075)
    dy = round(height * 0.075)

    if (
        width - 2 * dx >= 8
        and height - 2 * dy >= 8
    ):
        tight = original.crop(
            (
                dx,
                dy,
                width - dx,
                height - dy,
            )
        )
    else:
        tight = original

    # CLAHE: LAB 색 공간의 밝기 채널만 보정
    rgb = np.asarray(original)

    lab = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2LAB,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(4, 4),
    )

    lab[:, :, 0] = clahe.apply(
        lab[:, :, 0]
    )

    enhanced = Image.fromarray(
        cv2.cvtColor(
            lab,
            cv2.COLOR_LAB2RGB,
        )
    )

    return {
        "baseline": original,
        "tight_85": tight,
        "clahe": enhanced,
    }


def predict_closed(model, crops):
    kind, network, setting = model

    probabilities = []

    if kind == "cnn":
        tensors = torch.stack(
            [
                setting(to_square(crop, 224))
                for crop in crops
            ]
        )

        with torch.inference_mode():
            probabilities = (
                network(tensors)
                .softmax(dim=1)[:, 0]
                .tolist()
            )

    else:
        for crop in crops:
            result = network.predict(
                to_square(crop, 224),
                imgsz=224,
                device="cpu",
                verbose=False,
            )[0]

            probabilities.append(
                float(
                    result.probs.data[setting]
                )
            )

    return float(
        np.mean(probabilities)
    )


def save_csv(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[
            (
                row["model"],
                row["variant"],
            )
        ].append(row)

    result = []

    for (model, variant), items in sorted(
        groups.items()
    ):
        valid = [
            row
            for row in items
            if row["status"] == "OK"
        ]

        closed = [
            row
            for row in items
            if row["actual"] == "Closed"
        ]

        opened = [
            row
            for row in items
            if row["actual"] == "Open"
        ]

        correct = sum(
            row["predicted"] == row["actual"]
            for row in valid
        )

        tp = sum(
            row["predicted"] == "Closed"
            for row in closed
        )

        fp = sum(
            row["predicted"] == "Closed"
            for row in opened
        )

        open_correct = sum(
            row["predicted"] == "Open"
            for row in opened
        )

        summary = {
            "model": model,
            "variant": variant,
            "total": len(items),
            "correct": correct,
            "roi_success": len(valid),
            "roi_fail": sum(
                row["status"] == "ROI_FAIL"
                for row in items
            ),
            "errors": sum(
                row["status"] == "ERROR"
                for row in items
            ),
            "overall_accuracy": round(
                correct / len(items),
                5,
            ),
            "detected_accuracy": (
                round(
                    correct / len(valid),
                    5,
                )
                if valid
                else ""
            ),
            "closed_recall": (
                round(
                    tp / len(closed),
                    5,
                )
                if closed
                else ""
            ),
            "open_recall": (
                round(
                    open_correct / len(opened),
                    5,
                )
                if opened
                else ""
            ),
            "open_false_alarm_rate": (
                round(
                    fp / len(opened),
                    5,
                )
                if opened
                else ""
            ),
            "fp_open_as_closed": fp,
            "fn_closed_as_open": sum(
                row["predicted"] == "Open"
                for row in closed
            ),
        }

        result.append(summary)

        print(
            f"{model:20} "
            f"{variant:10} "
            f"Accuracy={summary['overall_accuracy']:.2%} "
            f"Closed Recall={summary['closed_recall']:.2%} "
            f"Open FP Rate={summary['open_false_alarm_rate']:.2%} "
            f"ROI Fail={summary['roi_fail']}"
        )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        default=(
            ROOT / "data/external_face/eyes/cew"
        ),
    )

    parser.add_argument(
        "--model",
        choices=["all"] + MODEL_NAMES,
        default="resnet18",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0이면 전체 이미지 평가",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-side",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--save-examples",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=(
            ROOT / "reports/cew/roi_ablation"
        ),
    )

    args = parser.parse_args()

    if args.limit < 0 or args.max_side < 224:
        parser.error(
            "--limit >= 0, --max-side >= 224"
        )

    names = (
        MODEL_NAMES
        if args.model == "all"
        else [args.model]
    )

    models = load_models(names)

    by_class = {}

    for label in ("Closed", "Open"):
        folder = args.data / label

        if not folder.is_dir():
            parser.error(
                f"폴더 없음: {folder}"
            )

        by_class[label] = sorted(
            p
            for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in EXT
        )

    rng = random.Random(args.seed)

    if args.limit:
        half = args.limit // 2

        selected = {
            label: rng.sample(
                paths,
                min(
                    len(paths),
                    half + (
                        1
                        if label == "Closed"
                        and args.limit % 2
                        else 0
                    ),
                ),
            )
            for label, paths in by_class.items()
        }

    else:
        selected = by_class

    images = sorted(
        [
            (path, label)
            for label, paths in selected.items()
            for path in paths
        ],
        key=lambda item: str(item[0]),
    )

    print(
        f"Images: {len(images)}"
        f" | Models: {names}"
        f" | Variants: {VARIANTS}"
    )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    example_count = 0

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as mesh:

        for index, (path, actual) in enumerate(images):

            try:
                with Image.open(path) as photo:
                    image = ImageOps.exif_transpose(
                        photo
                    ).convert("RGB")

                    if max(image.size) > args.max_side:
                        image.thumbnail(
                            (
                                args.max_side,
                                args.max_side,
                            )
                        )

                eyes = crop_eyes(
                    image,
                    mesh,
                )

                crops = {
                    name: []
                    for name in VARIANTS
                }

                for eye in eyes:
                    variants = make_variants(eye)

                    for name, crop in variants.items():
                        crops[name].append(crop)

                # 실제 모델 입력 ROI 저장
                if (
                    eyes
                    and example_count < args.save_examples
                ):
                    example_count += 1

                    for variant, variant_crops in crops.items():
                        destination = (
                            args.out
                            / "roi_examples"
                            / actual
                            / variant
                        )

                        destination.mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        for n, crop in enumerate(
                            variant_crops
                        ):
                            to_square(
                                crop,
                                224,
                            ).save(
                                destination
                                / f"{index:05d}_{n}.png"
                            )

                for name, model in models.items():
                    for variant in VARIANTS:

                        probability = (
                            predict_closed(
                                model,
                                crops[variant],
                            )
                            if eyes
                            else None
                        )

                        if probability is None:
                            predicted = "UNKNOWN"

                        elif probability >= 0.5:
                            predicted = "Closed"

                        else:
                            predicted = "Open"

                        rows.append({
                            "image": str(path),
                            "actual": actual,
                            "model": name,
                            "variant": variant,
                            "closed_probability": (
                                round(probability, 6)
                                if probability is not None
                                else ""
                            ),
                            "predicted": predicted,
                            "status": (
                                "OK"
                                if eyes
                                else "ROI_FAIL"
                            ),
                            "roi_count": len(eyes),
                            "error": "",
                        })

            except Exception as exc:
                for name in models:
                    for variant in VARIANTS:
                        rows.append({
                            "image": str(path),
                            "actual": actual,
                            "model": name,
                            "variant": variant,
                            "closed_probability": "",
                            "predicted": "UNKNOWN",
                            "status": "ERROR",
                            "roi_count": 0,
                            "error": (
                                f"{type(exc).__name__}: {exc}"
                            ),
                        })

            if (index + 1) % 100 == 0:
                print(
                    f"Processed {index + 1}/{len(images)}"
                )

    if not rows:
        raise SystemExit(
            "평가할 이미지가 없어."
        )

    save_csv(
        args.out / "predictions.csv",
        rows,
    )

    summary = summarize(rows)

    save_csv(
        args.out / "summary.csv",
        summary,
    )

    print()
    print("Saved:", args.out)
    print(
        "전처리 비교 완료. "
        "재학습이나 원본 수정은 하지 않았어."
    )


if __name__ == "__main__":
    main()
