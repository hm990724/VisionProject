
"""12_external_eval.py - External image model evaluation."""

import argparse
import csv
import importlib
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from common import to_square
from roi_utils import crop_eyes


ROOT = Path(__file__).resolve().parent
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CLASSES = {
    "eyes": {"Closed", "Open"},
    "yawn": {"yawn", "no_yawn"},
}


def samples_in(base, task):
    root = base / task
    if not root.is_dir():
        return []

    result = []

    for p in sorted(root.rglob("*")):
        if (
            p.is_file()
            and p.suffix.lower() in EXT
            and p.parent.name in CLASSES[task]
        ):
            rel = p.relative_to(root).parts
            condition = (
                rel[-3] if len(rel) >= 3 else "normal"
            )
            result.append(
                (p, condition, p.parent.name)
            )

    return result


def eye_models():
    result = {}

    weights = (
        ROOT
        / "runs/eyes_yolo26n-cls/weights/best.pt"
    )

    if weights.is_file():
        from ultralytics import YOLO

        model = YOLO(str(weights))

        closed = next(
            (
                int(k)
                for k, v in model.names.items()
                if str(v).lower() == "closed"
            ),
            None,
        )

        if closed is None:
            raise ValueError(
                f"Unexpected YOLO classes: {model.names}"
            )

        result["yolo26n"] = (
            "yolo",
            model,
            closed,
        )
    else:
        print("[SKIP] YOLO26n:", weights)

    cnn_module = importlib.import_module(
        "08_train_cnn"
    )

    for name in (
        "resnet18",
        "mobilenet_v3_small",
    ):
        weights = (
            ROOT
            / "runs"
            / f"eyes_{name}"
            / "best.pt"
        )

        if not weights.is_file():
            print("[SKIP]", name, weights)
            continue

        ckpt = torch.load(
            weights,
            map_location="cpu",
            weights_only=True,
        )

        if list(ckpt["classes"]) != [
            "Closed",
            "Open",
        ]:
            raise ValueError(
                f"Class order mismatch: {weights}"
            )

        model = cnn_module.make_model(
            name,
            pretrained=False,
        )

        model.load_state_dict(
            ckpt["model"]
        )
        model.eval()

        result[name] = (
            "cnn",
            model,
            cnn_module.get_transforms(False),
        )

    return result


def team_yawn(team_folder):
    common = (
        team_folder
        / "03_yawn_common.py"
    )

    weights = (
        team_folder
        / "runs/03_yawn_resnet18/weights/03_best.pt"
    )

    for p in (common, weights):
        if not p.is_file():
            raise FileNotFoundError(
                f"Team file missing: {p}"
            )

    spec = importlib.util.spec_from_file_location(
        "team_yawn_eval",
        common,
    )

    mod = importlib.util.module_from_spec(
        spec
    )

    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    device = mod.choose_device("cpu")

    model, ckpt = mod.load_checkpoint(
        weights,
        device,
    )

    face_file = mod.verify_face_model(
        team_folder / "models/face_landmarker.task",
        ckpt,
    )

    extractor = mod.MouthExtractor(
        face_file,
        mod.CropConfig(
            **ckpt["crop_config"]
        ),
    )

    names = ckpt.get("class_names")

    if (
        names is not None
        and list(names) != ["no_yawn", "yawn"]
    ):
        extractor.close()
        raise ValueError(
            f"Yawn class order mismatch: {names}"
        )

    return (
        model,
        extractor,
        mod.make_transform(training=False),
        device,
    )


def eye_scores(models, crops):
    scores = {}

    for name, (kind, net, settings) in models.items():
        probs = []

        for eye in crops:
            img = to_square(eye, 224)

            if kind == "yolo":
                pred = net.predict(
                    img,
                    imgsz=224,
                    device="cpu",
                    verbose=False,
                )[0]

                prob = float(
                    pred.probs.data[
                        settings
                    ].item()
                )

            else:
                x = (
                    settings(img)
                    .unsqueeze(0)
                )

                with torch.inference_mode():
                    prob = float(
                        net(x)
                        .softmax(1)[0, 0]
                        .item()
                    )

            probs.append(prob)

        scores[name] = float(
            np.mean(probs)
        )

    return scores


def record(
    task,
    cond,
    model,
    path,
    truth,
    prob,
    count,
    status,
    error="",
):
    pos = (
        "Closed"
        if task == "eyes"
        else "yawn"
    )

    neg = (
        "Open"
        if task == "eyes"
        else "no_yawn"
    )

    if status == "ERROR":
        pred = "ERROR"

    elif prob is None:
        pred = "UNKNOWN"

    else:
        pred = (
            pos if prob >= 0.5 else neg
        )

    return {
        "task": task,
        "condition": cond,
        "model": model,
        "image": str(path),
        "actual": truth,
        "predicted": pred,
        "positive_probability": (
            ""
            if prob is None
            else round(prob, 6)
        ),
        "roi_count": count,
        "correct": int(pred == truth),
        "status": status,
        "error": error,
    }


def evaluate(args):
    base = args.data.resolve()
    out = args.output.resolve()
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    tasks = (
        ["eyes", "yawn"]
        if args.task == "both"
        else [args.task]
    )

    if args.eye_input == "cropped":
        print(
            "[INFO] Cropped-eye mode: "
            "classifiers only, "
            "NOT MediaPipe detection."
        )

    for task in tasks:
        entries = samples_in(
            base,
            task,
        )

        print(
            f"[{task}] images: {len(entries)}"
        )

        if not entries:
            continue

        # ==================================
        # Eye model evaluation
        # ==================================

        if task == "eyes":
            import mediapipe as mp

            models = eye_models()

            if not models:
                continue

            with mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
            ) as mesh:

                for i, (path, cond, truth) in enumerate(
                    entries
                ):
                    try:
                        with Image.open(path) as photo:
                            im = ImageOps.exif_transpose(
                                photo
                            ).convert("RGB")

                            if max(im.size) > args.max_side:
                                im.thumbnail(
                                    (
                                        args.max_side,
                                        args.max_side,
                                    )
                                )

                        if args.eye_input == "cropped":
                            crops = [im]

                        else:
                            crops = crop_eyes(
                                im,
                                mesh,
                            )

                        if args.save_roi and crops:
                            dest = (
                                out
                                / "roi"
                                / task
                                / cond
                                / truth
                            )

                            dest.mkdir(
                                parents=True,
                                exist_ok=True,
                            )

                            for j, crop in enumerate(
                                crops
                            ):
                                to_square(
                                    crop,
                                    224,
                                ).save(
                                    dest
                                    / f"{i:05d}_{j}.png"
                                )

                        probs = (
                            eye_scores(
                                models,
                                crops,
                            )
                            if crops
                            else {}
                        )

                        for name in models:
                            if crops:
                                status = (
                                    "PRE_CROPPED"
                                    if args.eye_input == "cropped"
                                    else "OK"
                                )
                            else:
                                status = "ROI_FAIL"

                            rows.append(
                                record(
                                    task,
                                    cond,
                                    name,
                                    path,
                                    truth,
                                    probs.get(name),
                                    len(crops),
                                    status,
                                )
                            )

                    except Exception as exc:
                        for name in models:
                            rows.append(
                                record(
                                    task,
                                    cond,
                                    name,
                                    path,
                                    truth,
                                    None,
                                    0,
                                    "ERROR",
                                    (
                                        f"{type(exc).__name__}: "
                                        f"{exc}"
                                    ),
                                )
                            )

        # ==================================
        # Yawn model evaluation
        # ==================================

        else:
            (
                network,
                extractor,
                transform,
                device,
            ) = team_yawn(
                args.team_dir.resolve()
            )

            try:
                for i, (path, cond, truth) in enumerate(
                    entries
                ):
                    try:
                        with Image.open(path) as photo:
                            im = ImageOps.exif_transpose(
                                photo
                            ).convert("RGB")

                            if max(im.size) > args.max_side:
                                im.thumbnail(
                                    (
                                        args.max_side,
                                        args.max_side,
                                    )
                                )

                        extracted = extractor.extract(
                            im
                        )

                        crop = (
                            extracted.get("crop")
                            if extracted.get("status") == "ok"
                            else None
                        )

                        prob = None

                        if crop is not None:
                            if args.save_roi:
                                dest = (
                                    out
                                    / "roi"
                                    / task
                                    / cond
                                    / truth
                                )

                                dest.mkdir(
                                    parents=True,
                                    exist_ok=True,
                                )

                                crop.save(
                                    dest
                                    / f"{i:05d}.png"
                                )

                            x = (
                                transform(crop)
                                .unsqueeze(0)
                                .to(device)
                            )

                            with torch.inference_mode():
                                prob = float(
                                    network(x)
                                    .softmax(1)[0, 1]
                                    .item()
                                )

                        rows.append(
                            record(
                                task,
                                cond,
                                "team_resnet18",
                                path,
                                truth,
                                prob,
                                int(crop is not None),
                                (
                                    "OK"
                                    if crop is not None
                                    else "ROI_FAIL"
                                ),
                            )
                        )

                    except Exception as exc:
                        rows.append(
                            record(
                                task,
                                cond,
                                "team_resnet18",
                                path,
                                truth,
                                None,
                                0,
                                "ERROR",
                                (
                                    f"{type(exc).__name__}: "
                                    f"{exc}"
                                ),
                            )
                        )

            finally:
                extractor.close()

    if not rows:
        raise SystemExit(
            "No labeled images or usable models. "
            "Check folders and weights."
        )

    # ======================================
    # Save prediction CSV
    # ======================================

    prediction_file = (
        out / "external_predictions.csv"
    )

    with prediction_file.open(
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

    # ======================================
    # Aggregate metrics
    # ======================================

    groups = defaultdict(list)

    for row in rows:
        groups[
            (
                row["task"],
                row["condition"],
                row["model"],
            )
        ].append(row)

        groups[
            (
                row["task"],
                "ALL",
                row["model"],
            )
        ].append(row)

    summaries = []

    for (
        task,
        condition,
        model,
    ), group in sorted(groups.items()):

        pos = (
            "Closed"
            if task == "eyes"
            else "yawn"
        )

        tp = sum(
            x["actual"] == pos
            and x["predicted"] == pos
            for x in group
        )

        fp = sum(
            x["actual"] != pos
            and x["predicted"] == pos
            for x in group
        )

        fn = sum(
            x["actual"] == pos
            and x["predicted"] != pos
            for x in group
        )

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall
            else 0.0
        )

        valid = [
            x
            for x in group
            if x["status"] in (
                "OK",
                "PRE_CROPPED",
            )
        ]

        correct = sum(
            x["correct"]
            for x in group
        )

        summaries.append({
            "task": task,
            "condition": condition,
            "model": model,
            "total": len(group),
            "valid": len(valid),
            "roi_fail": sum(
                x["status"] == "ROI_FAIL"
                for x in group
            ),
            "errors": sum(
                x["status"] == "ERROR"
                for x in group
            ),
            "correct": correct,
            "overall_accuracy": round(
                correct / len(group),
                4,
            ),
            "detected_only_accuracy": (
                round(
                    correct / len(valid),
                    4,
                )
                if valid
                else ""
            ),
            "positive_recall": round(
                recall,
                4,
            ),
            "positive_precision": round(
                precision,
                4,
            ),
            "positive_f1": round(
                f1,
                4,
            ),
            "fp": fp,
            "fn": fn,
        })

    summary_file = (
        out / "external_summary.csv"
    )

    with summary_file.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summaries[0]),
        )

        writer.writeheader()
        writer.writerows(summaries)

    for x in summaries:
        if x["condition"] == "ALL":
            print(
                f"{x['task']:5} | "
                f"{x['model']:20} | "
                f"{x['correct']:3}/{x['total']:3} "
                f"({x['overall_accuracy']:.2%}) | "
                f"ROI/valid "
                f"{x['valid']}/{x['total']}"
            )

    print("Saved:", prediction_file)
    print("Saved:", summary_file)

    print(
        "NOTE: ROI failures are counted as "
        "incorrect. Do not tune on final holdout."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate external images"
    )

    parser.add_argument(
        "--task",
        choices=("eyes", "yawn", "both"),
        default="eyes",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/external",
    )

    parser.add_argument(
        "--team-dir",
        type=Path,
        default=ROOT / "teammate_yawn",
    )

    parser.add_argument(
        "--eye-input",
        choices=("face", "cropped"),
        default="face",
    )

    parser.add_argument(
        "--save-roi",
        action="store_true",
    )

    parser.add_argument(
        "--max-side",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/external",
    )

    args = parser.parse_args()

    if args.max_side < 224:
        parser.error(
            "--max-side must be >= 224"
        )

    evaluate(args)


if __name__ == "__main__":
    main()
