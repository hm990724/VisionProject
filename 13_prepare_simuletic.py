
"""
13_prepare_simuletic.py

Simuletic DMS 데이터셋의 JSON 라벨을 읽고
눈 상태 평가용 Closed / Open 이미지로 분류한다.

원본 데이터는 변경하지 않으며 파일을 복사한다.
Drowsy/Microsleep은 별도 보류한다.
"""

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent

DEFAULT_SOURCE = (
    ROOT
    / "data"
    / "external_source"
    / "Simuletic_DMS_Dataset"
)

DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "external_driver"
    / "eyes"
    / "simuletic"
)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
}

VALID_LABELS = {
    "Closed",
    "Open",
}


def build_image_index(images_dir):
    """
    images/ 아래 모든 하위 폴더를 검색한다.
    파일명을 기준으로 실제 이미지 경로를 찾는다.
    """
    index = defaultdict(list)

    for path in images_dir.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            index[path.name.lower()].append(path)

    return index


def read_json_samples(labels_dir):
    """
    labels/ 아래 JSON 파일을 모두 읽는다.
    """
    for json_path in sorted(labels_dir.rglob("*.json")):
        try:
            with json_path.open(
                "r", encoding="utf-8-sig"
            ) as f:
                data = json.load(f)

        except (OSError, json.JSONDecodeError) as exc:
            print(f"[ERROR] JSON 읽기 실패: {json_path}")
            print(f"        {exc}")
            continue

        samples = data.get("samples", [])

        if not isinstance(samples, list):
            print(f"[SKIP] samples 형식 오류: {json_path}")
            continue

        for sample in samples:
            if isinstance(sample, dict):
                yield json_path, sample


def prepare(source, output, overwrite=False):
    images_dir = source / "images"
    labels_dir = source / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"images 폴더가 없어: {images_dir}"
        )

    if not labels_dir.is_dir():
        raise FileNotFoundError(
            f"labels 폴더가 없어: {labels_dir}"
        )

    # 원본 경로와 출력 경로가 겹치지 않도록 보호
    source = source.resolve()
    output = output.resolve()

    if output == source or source in output.parents:
        raise ValueError(
            "출력 경로를 원본 Simuletic 폴더 안으로 지정하면 안 돼."
        )

    (output / "Closed").mkdir(
        parents=True, exist_ok=True
    )
    (output / "Open").mkdir(
        parents=True, exist_ok=True
    )

    print("=" * 60)
    print("SIMULETIC DATA PREPARATION")
    print("=" * 60)

    print(f"Source : {source}")
    print(f"Output : {output}")

    image_index = build_image_index(images_dir)

    print(
        "발견된 이미지:",
        sum(len(v) for v in image_index.values())
    )

    counts = Counter()
    manifest = []
    seen_destinations = set()

    for json_path, sample in read_json_samples(labels_dir):

        filename = sample.get("image", "")

        attributes = sample.get("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}

        label = attributes.get("eye_state", "")

        if not isinstance(filename, str) or not filename:
            counts["INVALID_SAMPLE"] += 1
            continue

        if not isinstance(label, str):
            label = ""

        # 원본의 video/JSON 이름을 함께 저장
        # 서로 다른 영상의 같은 파일명 충돌 방지
        json_stem = json_path.stem
        image_name = Path(filename).name

        row = {
            "json_file": json_path.name,
            "image": image_name,
            "eye_state": label,
            "timestamp": sample.get("timestamp", ""),
            "perclos": attributes.get("perclos", ""),
            "source_path": "",
            "output_path": "",
            "status": "",
        }

        # Drowsy/Microsleep은 Closed로 임의 변환하지 않음
        if label not in VALID_LABELS:
            counts["EXCLUDED"] += 1
            row["status"] = "EXCLUDED"
            manifest.append(row)
            continue

        matches = image_index.get(
            image_name.lower(), []
        )

        if len(matches) == 0:
            counts["MISSING_IMAGE"] += 1
            row["status"] = "MISSING_IMAGE"
            manifest.append(row)
            continue

        if len(matches) > 1:
            counts["AMBIGUOUS_IMAGE"] += 1
            row["status"] = "AMBIGUOUS_IMAGE"
            manifest.append(row)
            continue

        original = matches[0]

        destination = (
            output
            / label
            / f"{json_stem}__{image_name}"
        )

        row["source_path"] = str(original)
        row["output_path"] = str(destination)

        if destination in seen_destinations:
            counts["DUPLICATE_RECORD"] += 1
            row["status"] = "DUPLICATE_RECORD"
            manifest.append(row)
            continue

        seen_destinations.add(destination)

        if destination.exists() and not overwrite:
            counts["ALREADY_EXISTS"] += 1
            row["status"] = "ALREADY_EXISTS"
            manifest.append(row)
            continue

        try:
            shutil.copy2(original, destination)

            counts[label] += 1
            row["status"] = "COPIED"

        except OSError as exc:
            counts["COPY_ERROR"] += 1
            row["status"] = f"COPY_ERROR: {exc}"

        manifest.append(row)

    # 분류 기록 CSV
    report_dir = ROOT / "reports" / "simuletic"
    report_dir.mkdir(
        parents=True, exist_ok=True
    )

    manifest_path = report_dir / "prepare_manifest.csv"

    columns = [
        "json_file",
        "image",
        "eye_state",
        "timestamp",
        "perclos",
        "source_path",
        "output_path",
        "status",
    ]

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(manifest)

    # 실제 출력 폴더 파일 수 확인
    closed_total = sum(
        1 for p in (output / "Closed").rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    open_total = sum(
        1 for p in (output / "Open").rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print()
    print("=" * 60)
    print("PREPARATION RESULT")
    print("=" * 60)

    print(f"이번 실행 Closed 복사 : {counts['Closed']}장")
    print(f"이번 실행 Open 복사   : {counts['Open']}장")
    print(f"별도 제외             : {counts['EXCLUDED']}장")
    print(f"누락 이미지           : {counts['MISSING_IMAGE']}장")
    print(f"중복 파일명           : {counts['AMBIGUOUS_IMAGE']}장")
    print(f"기존 파일             : {counts['ALREADY_EXISTS']}장")
    print(f"복사 오류             : {counts['COPY_ERROR']}장")

    print()
    print(f"현재 Closed 폴더 : {closed_total}장")
    print(f"현재 Open 폴더   : {open_total}장")

    print()
    print("Manifest:", manifest_path)

    if closed_total == 102 and open_total == 41:
        print()
        print("[OK] 예상한 143장 분류 완료!")
    else:
        print()
        print("[CHECK] 예상 개수와 차이가 있어.")
        print("Manifest의 status 항목을 확인해 줘.")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Simuletic eye dataset"
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 출력 파일 덮어쓰기",
    )

    args = parser.parse_args()

    prepare(
        args.source,
        args.output,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
