
"""YOLO26n eye detection + teammate ResNet18 yawn detection.

Q / ESC: quit
D: toggle ROI preview
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from common import to_square
from roi_utils import crop_eyes


ROOT = Path(__file__).resolve().parent


def load_team_module(team_dir: Path):
    source = team_dir / "03_yawn_common.py"

    if not source.is_file():
        raise FileNotFoundError(
            f"팀원 공통 코드 없음: {source}"
        )

    spec = importlib.util.spec_from_file_location(
        "team_yawn_common_03", source
    )
    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def show_crop(frame, crop, x, y, name):
    if crop is None:
        return

    preview = crop.convert("RGB")
    preview.thumbnail((135, 90))

    bgr = cv2.cvtColor(
        np.asarray(preview),
        cv2.COLOR_RGB2BGR,
    )

    h, w = bgr.shape[:2]

    if (
        y + h + 20 < frame.shape[0]
        and x + w < frame.shape[1]
    ):
        frame[y:y + h, x:x + w] = bgr

        cv2.putText(
            frame,
            name,
            (x, y + h + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera", type=int, default=0
    )

    parser.add_argument(
        "--team-dir",
        type=Path,
        default=ROOT / "teammate_yawn",
    )

    parser.add_argument(
        "--eyes-weights",
        type=Path,
        default=(
            ROOT
            / "runs/eyes_yolo26n-cls/weights/best.pt"
        ),
    )

    parser.add_argument(
        "--yawn-weights",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--face-model",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--eye-threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--yawn-threshold",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--closed-seconds",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--yawn-seconds",
        type=float,
        default=1.5,
    )

    args = parser.parse_args()

    if args.stride < 1:
        parser.error(
            "--stride는 1 이상이어야 해"
        )

    team_dir = args.team_dir.resolve()

    yawn_weights = (
        args.yawn_weights
        or team_dir
        / "runs/03_yawn_resnet18/weights/03_best.pt"
    )

    face_model = (
        args.face_model
        or team_dir / "models/face_landmarker.task"
    )

    # 가중치 파일 확인
    for path in (
        args.eyes_weights,
        yawn_weights,
    ):
        if not path.is_file():
            parser.error(
                f"가중치 파일 없음: {path}"
            )

    # ----------------------------------------
    # 1. 팀원 하품 ResNet18 불러오기
    # ----------------------------------------

    team = load_team_module(team_dir)

    device = team.choose_device("cpu")

    yawn_net, checkpoint = (
        team.load_checkpoint(
            yawn_weights,
            device,
        )
    )

    verified_face_model = (
        team.verify_face_model(
            face_model,
            checkpoint,
        )
    )

    # 학습 당시 입 추출 설정 사용
    mouth = team.MouthExtractor(
        verified_face_model,
        team.CropConfig(
            **checkpoint["crop_config"]
        ),
    )

    yawn_transform = team.make_transform(
        training=False
    )

    yawn_threshold = (
        float(args.yawn_threshold)
        if args.yawn_threshold is not None
        else float(
            checkpoint.get("threshold", 0.5)
        )
    )

    # ----------------------------------------
    # 2. 기존 YOLO26n 눈 모델 불러오기
    # ----------------------------------------

    eye_net = YOLO(
        str(args.eyes_weights)
    )

    closed_idx = next(
        (
            int(k)
            for k, v in eye_net.names.items()
            if str(v).lower() == "closed"
        ),
        None,
    )

    if closed_idx is None:
        raise ValueError(
            "눈 모델에 Closed 라벨이 없어: "
            f"{eye_net.names}"
        )

    # ----------------------------------------
    # 3. 웹캠 연결
    # ----------------------------------------

    backend = (
        cv2.CAP_DSHOW
        if sys.platform == "win32"
        else cv2.CAP_ANY
    )

    cap = cv2.VideoCapture(
        args.camera,
        backend,
    )

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(
            args.camera
        )

    if not cap.isOpened():
        mouth.close()
        raise RuntimeError(
            "웹캠 열기 실패. --camera 1을 시도해 봐"
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH, 640
    )
    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT, 480
    )

    eye_prob = None
    yawn_prob = None

    closed_since = None
    yawn_since = None

    yawn_latched = False
    yawn_count = 0

    last_infer = None
    fps_samples = deque(maxlen=30)

    frame_no = 0
    show_rois = True

    eye_crops = []
    mouth_crop = None

    print(
        "[READY] Eyes=YOLO26n / Yawn=ResNet18"
    )
    print(
        "Q/ESC: 종료 | D: ROI 미리보기"
    )

    # ----------------------------------------
    # 4. 실시간 추론
    # ----------------------------------------

    try:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as mesh:

            while True:
                start = time.perf_counter()

                ok, frame = cap.read()

                if not ok:
                    print("웹캠 프레임 읽기 실패")
                    break

                frame_no += 1

                frame = cv2.flip(
                    frame, 1
                )

                now = time.monotonic()

                if frame_no % args.stride == 0:

                    rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB,
                    )

                    pil = Image.fromarray(rgb)

                    # -------------------------
                    # 눈 ROI + YOLO26n
                    # -------------------------

                    eye_crops = crop_eyes(
                        pil, mesh
                    )

                    if eye_crops:
                        probabilities = []

                        for crop in eye_crops:
                            prepared = to_square(
                                crop, 224
                            )

                            pred = eye_net.predict(
                                prepared,
                                imgsz=224,
                                device="cpu",
                                verbose=False,
                            )[0]

                            prob = float(
                                pred.probs.data[
                                    closed_idx
                                ]
                            )

                            probabilities.append(
                                prob
                            )

                        eye_prob = (
                            sum(probabilities)
                            / len(probabilities)
                        )
                    else:
                        eye_prob = None

                    # -------------------------
                    # 입 ROI + ResNet18
                    # -------------------------

                    extracted = mouth.extract(
                        pil
                    )

                    mouth_crop = (
                        extracted["crop"]
                        if extracted["status"] == "ok"
                        else None
                    )

                    if mouth_crop is None:
                        # 검출 실패는 정상 판정이 아닌 UNKNOWN
                        yawn_prob = None

                    else:
                        tensor = (
                            yawn_transform(
                                mouth_crop
                            )
                            .unsqueeze(0)
                            .to(device)
                        )

                        with torch.inference_mode():
                            logits = yawn_net(
                                tensor
                            )

                            yawn_prob = float(
                                logits.softmax(
                                    dim=1
                                )[0, 1]
                            )

                    # -------------------------
                    # 연속 시간 처리
                    # -------------------------

                    if (
                        last_infer is not None
                        and now - last_infer > 1.0
                    ):
                        closed_since = None
                        yawn_since = None
                        yawn_latched = False

                    last_infer = now

                    # 눈 감김 지속 시간
                    if (
                        eye_prob is not None
                        and eye_prob
                        >= args.eye_threshold
                    ):
                        if closed_since is None:
                            closed_since = now

                    else:
                        closed_since = None

                    # 하품 지속 시간 및 횟수
                    if (
                        yawn_prob is not None
                        and yawn_prob
                        >= yawn_threshold
                    ):
                        if yawn_since is None:
                            yawn_since = now

                        duration = (
                            now - yawn_since
                        )

                        if (
                            not yawn_latched
                            and duration
                            >= args.yawn_seconds
                        ):
                            yawn_count += 1
                            yawn_latched = True

                    else:
                        yawn_since = None
                        yawn_latched = False

                # --------------------------------
                # 5. 웹캠 UI
                # --------------------------------

                closed_sec = (
                    now - closed_since
                    if closed_since is not None
                    else 0.0
                )

                warning = (
                    closed_sec
                    >= args.closed_seconds
                )

                if eye_prob is None:
                    eye_str = "UNKNOWN"
                else:
                    state = (
                        "CLOSED"
                        if eye_prob
                        >= args.eye_threshold
                        else "OPEN"
                    )
                    eye_str = (
                        f"{state} ({eye_prob:.0%})"
                    )

                if yawn_prob is None:
                    yawn_str = "UNKNOWN"
                else:
                    state = (
                        "YES"
                        if yawn_prob
                        >= yawn_threshold
                        else "NO"
                    )
                    yawn_str = (
                        f"{state} ({yawn_prob:.0%})"
                    )

                def line(
                    msg,
                    y,
                    color=(255, 255, 255),
                ):
                    cv2.putText(
                        frame,
                        msg,
                        (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (0, 0, 0),
                        4,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        frame,
                        msg,
                        (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                line(
                    "EYES: " + eye_str,
                    32,
                )

                line(
                    "YAWN: " + yawn_str,
                    61,
                )

                line(
                    f"Closed: {closed_sec:.1f}s "
                    f"| Yawn candidates: {yawn_count}",
                    90,
                )

                if warning:
                    cv2.rectangle(
                        frame,
                        (2, 2),
                        (
                            frame.shape[1] - 3,
                            frame.shape[0] - 3,
                        ),
                        (0, 0, 255),
                        6,
                    )

                    line(
                        "DROWSINESS WARNING",
                        125,
                        (0, 0, 255),
                    )

                if show_rois:
                    for i, crop in enumerate(
                        eye_crops[:2]
                    ):
                        show_crop(
                            frame,
                            crop,
                            12 + i * 145,
                            165,
                            f"Eye {i + 1}",
                        )

                    show_crop(
                        frame,
                        mouth_crop,
                        12,
                        300,
                        "Mouth ROI",
                    )

                # FPS
                fps_samples.append(
                    time.perf_counter()
                    - start
                )

                fps = (
                    len(fps_samples)
                    / max(
                        sum(fps_samples),
                        1e-9,
                    )
                )

                line(
                    f"FPS: {fps:.1f}"
                    " | Q: quit / D: ROI",
                    frame.shape[0] - 12,
                )

                cv2.imshow(
                    "VisionProject - eye + yawn",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (
                    ord("q"), 27
                ):
                    break

                if key == ord("d"):
                    show_rois = (
                        not show_rois
                    )

    finally:
        mouth.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
