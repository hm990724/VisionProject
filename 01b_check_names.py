"""파일 이름에 사람/영상 번호 같은 단서가 있는지 확인

    python 01b_check_names.py --data data/raw

클래스 폴더(Closed, Open, no_yawn, yawn)마다
  - 파일 이름 샘플
  - 이름 패턴(숫자를 #으로 바꾼 형태)별 개수
  - 이름 끝의 번호를 뗀 '접두어'가 몇 종류인지, 그 접두어가 train과 test에 동시에 있는지
를 보여준다. 접두어가 사람이나 영상 번호라면 그 단위로 묶어서 분할해야 한다.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import re

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": "train", "training": "train", "val": "val", "valid": "val",
          "validation": "val", "test": "test", "testing": "test"}


def split_of(path, root):
    for part in path.relative_to(root).parts[:-1]:
        if part.lower() in SPLITS:
            return SPLITS[part.lower()]
    return "none"


def pattern(stem):
    return re.sub(r"\d+", "#", stem)


def prefix(stem):
    """이름 끝의 숫자·구분자를 떼어낸 앞부분. 예: 'person3_frame120' -> 'person3_frame'"""
    return re.sub(r"[\d_\-\s\(\)]+$", "", stem) or stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--samples", type=int, default=8)
    a = ap.parse_args()

    root = Path(a.data).resolve()
    if not root.exists():
        raise SystemExit(f"[오류] 폴더가 없습니다: {root}")

    by_class = defaultdict(list)
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXT:
            by_class[p.parent.name].append(p)

    for cls, paths in sorted(by_class.items()):
        print(f"\n==================== {cls} ({len(paths)}장) ====================")
        step = max(1, len(paths) // a.samples)
        print("이름 샘플:", [p.name for p in paths[::step][: a.samples]])

        pats = Counter(pattern(p.stem) for p in paths)
        print(f"이름 패턴 {len(pats)}종류, 상위 5개:")
        for pat, n in pats.most_common(5):
            print(f"   {n:5d}장  {pat}")

        pre_splits = defaultdict(set)
        pre_count = Counter()
        for p in paths:
            k = prefix(p.stem)
            pre_count[k] += 1
            pre_splits[k].add(split_of(p, root))
        cross = [k for k, s in pre_splits.items() if len(s - {"none"}) > 1]
        print(f"접두어 {len(pre_count)}종류, 상위 5개:")
        for k, n in pre_count.most_common(5):
            print(f"   {n:5d}장  {k!r}  split={sorted(pre_splits[k])}")
        print(f"train과 test에 동시에 나오는 접두어: {len(cross)}종류")

    print("\n[해석 가이드]")
    print(" - 접두어가 수십 종류 이하이고 한 접두어에 수십 장씩 있으면 사람/영상 단위 이름일 가능성이 높다.")
    print(" - 접두어가 장수와 거의 같으면(예: 1448장에 1400종류) 이름에는 단서가 없다. 번호만 있는 경우다.")
    print(" - 단서가 있으면: python 02_prepare.py --group-regex \"정규식\"  (예: ^(.*?)[_\\-]?\\d+$)")


if __name__ == "__main__":
    main()
