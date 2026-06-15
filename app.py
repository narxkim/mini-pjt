# ==================================================
# Streamlit YOLO 추론 앱
# 목적: 새로운 이미지에서 user_character / user_id 탐지 확인
# ==================================================
import streamlit as st
import pandas as pd
import cv2
import numpy as np

from pathlib import Path
from PIL import Image
from ultralytics import YOLO



st.set_page_config(
    page_title="NS Character Detection",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).parent

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "ns_yolov8s_960"
    / "weights"
    / "best.pt"
)

# ==================================================
# 모델 로드
# Streamlit은 위젯 조작 시 전체 코드가 다시 실행되므로
# 모델은 캐시로 한 번만 로드
# ==================================================

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))

model = load_model()

# ==================================================
# 화면 구성
# ==================================================

st.title("YOLOv8 캐릭터 · 닉네임 탐지 테스트")
st.write("새로운 게임 이미지를 업로드하면 user_character와 user_id 탐지 결과를 확인합니다.")

with st.sidebar:
    st.header("설정")
    conf_value = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )

uploaded_file = st.file_uploader(
    "이미지 업로드",
    type=["jpg", "jpeg", "png"]
)

# ==================================================
# 이미지 추론
# ==================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("원본 이미지")
        st.image(image, width="stretch")

    results = model.predict(
        source=image_np,
        imgsz=960,
        conf=conf_value,
        device=0,
        verbose=False
    )

    result = results[0]

    # 박스가 그려진 결과 이미지
    plotted_img = result.plot()
    plotted_img = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("탐지 결과")
        st.image(plotted_img, width="stretch")

    # ==================================================
    # 탐지 결과 표 생성
    # ==================================================

    rows = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        rows.append({
            "class": class_name,
            "confidence": round(confidence, 4),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

    st.subheader("탐지 박스 정보")

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True)

        st.metric("탐지 객체 수", len(rows))
        st.write("클래스별 개수")
        st.write(df["class"].value_counts())
    else:
        st.warning("탐지된 객체가 없습니다. Confidence 값을 낮춰보세요.")