"""5단계: Streamlit 데모 (일단 이미지 한 장 분류. 영상/웹캠/여러 명은 이 다음 단계)

    streamlit run 05_app.py
"""
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO

from common import to_square

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="졸음 징후 분류", layout="centered")


@st.cache_resource
def load(path: str):
    return YOLO(path)


st.title("졸음 징후 이미지 분류")
weights = sorted(ROOT.glob("runs/*/weights/best.pt"))
if not weights:
    st.warning("학습된 모델이 없습니다. 03_train.py를 먼저 실행하세요.")
    st.stop()

choice = st.selectbox("모델 선택", weights, format_func=lambda p: p.parent.parent.name)
up = st.file_uploader("눈 또는 입 주변 이미지", type=["jpg", "jpeg", "png", "bmp", "webp"])

if up:
    img = to_square(Image.open(up))  # 학습 때와 같은 전처리
    st.image(img, caption="모델 입력 이미지(전처리 후)", width=224)
    r = load(str(choice)).predict(img, imgsz=224, verbose=False)[0]
    st.subheader(f"예측: {r.names[r.probs.top1]}  ({float(r.probs.top1conf) * 100:.1f}%)")
    st.bar_chart({r.names[i]: float(v) for i, v in enumerate(r.probs.data.tolist())})
    st.caption("실험용 데모이며 실제 졸음 상태나 운전 안전을 보장하지 않습니다.")
