"""Fast, human-verified NITYMED video labeling for 24_nitymed_prepare.py.

Run:
    streamlit run 26_nitymed_review_fast.py

Review each original video, type the observed yawn intervals (seconds), and
apply labels to its sampled frames in one save. No automatic pseudo-labeling.
Original labels are preserved by default; backs up manifest before edits.
"""
from __future__ import annotations

import csv
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "reports/nitymed_full_review/annotation_manifest.csv"
LABELS = {"yawn", "no_yawn"}

st.set_page_config(
    page_title="NITYMED | 빠른 영상 라벨링",
    page_icon="🎞️",
    layout="wide",
)


# ---------- Safe CSV updates ----------
def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def persist_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    backup = path.with_name("annotation_manifest.before_fast_review.csv")
    if not backup.exists():
        shutil.copy2(path, backup)
    tmp = path.with_name("annotation_manifest.fast_review_tmp.csv")
    with tmp.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)


def intervals_from_text(text: str) -> list[tuple[float, float]]:
    """Accept '4.5-8, 12-14.5' as inclusive second ranges."""
    cleaned = text.strip()
    if not cleaned:
        return []
    out = []
    parts = re.split(r"[,;\n]+", cleaned)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        pattern = r"^\s*(\d+(?:\.\d+)?)\s*[-~–—]\s*(\d+(?:\.\d+)?)\s*$"
        match = re.match(pattern, part)
        if not match:
            raise ValueError(f"시간 구간 형식이 잘못됐어: {part!r}. 예) 5-9, 13.5-16")
        a, b = map(float, match.groups())
        if b <= a:
            raise ValueError(f"끝 시간이 시작보다 커야 해: {part!r}")
        out.append((a, b))
    out.sort()
    for prev, cur in zip(out, out[1:]):
        if cur[0] < prev[1]:
            raise ValueError("서로 겹치는 하품 구간이 있어. 구간을 합쳐줘.")
    return out


def classify_sec(seconds: float, intervals: list[tuple[float, float]], guard: float) -> str:
    """Classify only from *human-described* intervals. Guard boundaries as unsure."""
    for start, end in intervals:
        if start <= seconds <= end:
            if seconds - start < guard or end - seconds < guard:
                return "hold"
            return "yawn"
        if (start - guard) <= seconds < start or end < seconds <= (end + guard):
            return "hold"
    return "no_yawn"


def is_valid_roi(row: dict) -> bool:
    return str(row.get("roi_status", "")).lower() == "ok" and bool(row.get("mouth_roi"))


def is_confirmed(row: dict) -> bool:
    return row.get("confirmed") == "1" and row.get("label") in LABELS


def picture_path(base: Path, row: dict, col: str) -> Path | None:
    rel = row.get(col, "")
    if not rel:
        return None
    p = Path(rel)
    return p if p.is_absolute() else base / p


# ---------- UI ----------
st.title("NITYMED · 영상 단위 빠른 라벨링")
st.caption(
    "영상 하나에서 하품 구간을 눈으로 확인하면, 14개 샘플을 한 번에 라벨링할 수 있어. "
    "기존 AI 확률은 참고만 하고 정답으로 자동 사용하지 않아."
)

manifest_text = st.sidebar.text_input("라벨 CSV", value=str(DEFAULT_MANIFEST))
manifest = Path(manifest_text).expanduser()
if not manifest.is_file():
    st.error(f"파일이 없어: {manifest}")
    st.code("python 24_nitymed_prepare.py prepare --source data/yawn_mp/Yawning "
            "--team-dir teammate_yawn --output reports/nitymed_full_review")
    st.stop()

rows = load_manifest(manifest)
if not rows:
    st.error("CSV에 이미지 정보가 없어.")
    st.stop()

by_video: dict[str, list[int]] = {}
for ix, rec in enumerate(rows):
    by_video.setdefault(rec["video_id"], []).append(ix)
video_ids = list(by_video)
name_of = {v: Path(rows[by_video[v][0]]["video"]).name for v in video_ids}

def reviewed_count(v: str) -> int:
    return sum(is_confirmed(rows[i]) for i in by_video[v])

all_confirmed = sum(is_confirmed(r) for r in rows)
counts = Counter(r.get("label") for r in rows if is_confirmed(r))
st.sidebar.metric("전체 확정", f"{all_confirmed} / {len(rows)}")
st.sidebar.write(f"하품 {counts['yawn']}장 · 비하품 {counts['no_yawn']}장")
st.sidebar.progress(all_confirmed / len(rows))

v = st.sidebar.selectbox(
    "영상 선택",
    video_ids,
    format_func=lambda x: f"{name_of[x]} · {reviewed_count(x)}/{len(by_video[x])}장",
)
indices = by_video[v]
records = [rows[ix] for ix in indices]
original_video = Path(records[0]["video"])
max_time = max(float(r["time_seconds"]) for r in records)

left, right = st.columns([1.08, 1], gap="large")
with left:
    st.subheader(f"원본 영상 · {name_of[v]}")
    if original_video.is_file():
        st.video(str(original_video))
    else:
        st.warning(f"영상 원본 경로를 찾지 못했어: {original_video}")
    st.caption(
        "하품 시작부터 끝까지 실제 영상을 확인해. "
        "Yawning 폴더에 있어도 영상 전체가 하품은 아니야."
    )

with right:
    st.subheader("프레임 미리보기 · 모델 예측과 정답 구분")
    montage_candidates = sorted((manifest.parent / "contact_sheets").glob(f"*_{v}.jpg"))
    if montage_candidates:
        st.image(str(montage_candidates[0]), use_container_width=True)
    else:
        st.info("Contact sheet를 찾지 못했어. 아래 프레임 표를 참고해.")
    st.caption("미리보기의 P(yawn)은 기존 모델 출력일 뿐 정답 라벨이 아니야.")

st.divider()
st.subheader("① 하품 구간 지정 → ② 한 번에 확정")
subject_default = next((r.get("subject_id", "") for r in records if r.get("subject_id", "")), "")
subject = st.text_input(
    "인물 ID (동일 인물이 여러 영상에 나오면 같은 값 입력)",
    value=subject_default,
    key=f"subject_{v}",
    placeholder="예: person_003",
)
st.caption(
    "한 영상의 인물 ID는 통일해줘. 인물 ID가 없으면 학습/평가를 영상 단위로 나누게 돼."
)

range_input = st.text_input(
    "영상에서 직접 본 하품 구간 (초)",
    key=f"ranges_{v}",
    placeholder="예: 5-9   또는   5-9, 13-16",
    help="시작·끝 시간을 원본 영상을 보면서 입력. 여러 번 하품하면 쉼표로 구분.",
)

with st.expander("안전한 라벨링 설정", expanded=True):
    guard = st.slider(
        "시작·끝 경계 보류 폭(초)",
        0.0, 1.5, 0.4, 0.1,
        help="하품 시작과 끝에 가까운 샘플은 강제로 하품/비하품으로 확정하지 않음.",
    )
    checked_all = st.checkbox(
        "원본 영상 전체를 확인했고, 지정 구간 바깥은 비하품임을 확인했어",
        key=f"watched_{v}",
    )
    no_yawn_video = st.checkbox(
        "이 영상에는 하품이 전혀 없어 (전체를 보고 확인함)",
        key=f"none_{v}",
    )
    overwrite = st.checkbox(
        "기존 확정 라벨도 덮어쓰기 (주의)",
        value=False,
        key=f"overwrite_{v}",
    )

intervals: list[tuple[float, float]] = []
parse_error = ""
try:
    intervals = intervals_from_text(range_input)
except ValueError as exc:
    parse_error = str(exc)
    st.error(parse_error)
if intervals and no_yawn_video:
    parse_error = "하품 구간과 '하품 없음'을 동시에 선택할 수 없어."
    st.error(parse_error)

preview = []
for idx in indices:
    r = rows[idx]
    sec = float(r["time_seconds"])
    proposal = (
        "hold" if not is_valid_roi(r)
        else classify_sec(sec, [] if no_yawn_video else intervals, guard)
    )
    preview.append({
        "샘플": r.get("frame_id", ""),
        "시각(초)": round(sec, 2),
        "제안": {"yawn": "하품", "no_yawn": "비하품", "hold": "보류"}[proposal],
        "기존확정": r["label"] if is_confirmed(r) else "-",
        "ROI": r.get("roi_status", ""),
        "기존P(yawn)": r.get("p_yawn", ""),
    })

st.dataframe(preview, use_container_width=True, hide_index=True)
st.info(
    "입 ROI가 없는 사진·하품 시작/끝 경계의 애매한 프레임은 자동 보류해. "
    "여러 하품 구간을 지정할 수 있어."
)

can_save = checked_all and not parse_error and bool(intervals or no_yawn_video)
if st.button("확인한 영상의 프레임 라벨 한 번에 저장", type="primary", disabled=not can_save):
    changed = 0
    skipped = 0
    for ix in indices:
        r = rows[ix]
        if is_confirmed(r) and not overwrite:
            skipped += 1
            continue
        sec = float(r["time_seconds"])
        classification = (
            "hold" if not is_valid_roi(r)
            else classify_sec(sec, [] if no_yawn_video else intervals, guard)
        )
        if classification == "hold":
            if not is_confirmed(r):
                r["label"] = ""
                r["confirmed"] = "0"
        else:
            r["label"] = classification
            r["confirmed"] = "1"
            if subject.strip():
                r["subject_id"] = subject.strip()
            changed += 1
    persist_manifest(manifest, rows)
    st.success(f"저장 완료! 새로 확정 {changed}장, 기존 라벨 보존 {skipped}장")
    st.rerun()

st.divider()
with st.expander("특정 프레임만 예외로 수정하기", expanded=False):
    st.caption("영상을 보고 하품 경계나 애매한 장면만 개별 수정하면 돼. 대부분은 위 일괄 저장으로 충분해.")
    row_options = {f"{rows[i]['time_seconds']}초 | {rows[i].get('frame_id','')}번": i for i in indices}
    selected_name = st.selectbox("수정할 프레임", list(row_options), key=f"manual_{v}")
    ri = row_options[selected_name]
    r = rows[ri]
    pic1, pic2 = st.columns(2)
    for col, key in ((pic1, "face_image"), (pic2, "mouth_roi")):
        p = picture_path(manifest.parent, r, key)
        with col:
            if p is not None and p.is_file():
                st.image(str(p), use_container_width=True)
            else:
                st.warning(f"{key} 없음")
    label = st.selectbox(
        "눈으로 확인한 정답",
        ["보류", "하품", "비하품"],
        key=f"manual_label_{v}_{ri}",
    )
    if st.button("이 프레임만 수정 저장", key=f"manual_save_{v}"):
        new_label = {"보류": "", "하품": "yawn", "비하품": "no_yawn"}[label]
        if new_label and not is_valid_roi(r):
            st.error("ROI 추출 실패 프레임은 학습 이미지로 확정할 수 없어.")
        else:
            r["label"] = new_label
            r["confirmed"] = "1" if new_label else "0"
            if subject.strip():
                r["subject_id"] = subject.strip()
            persist_manifest(manifest, rows)
            st.success("저장 완료")
            st.rerun()

st.caption(
    "라벨 확정은 사람이 직접 영상을 확인한 뒤에만 실행해. "
    "동일 인물/영상은 train·val·test에서 분리해야 하고, 프레임 1,400장이 독립 이벤트 1,400개를 뜻하지는 않아."
)
