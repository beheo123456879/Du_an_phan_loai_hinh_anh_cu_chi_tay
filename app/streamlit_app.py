"""Giao diện Streamlit (chạy trên máy cá nhân có webcam).

    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import pandas as pd
import streamlit as st

from app.predictor import GesturePredictor, draw_result
from app.rps_game import RPSGame
from src.config import VI_NAMES
from src.gradcam import overlay_cam

st.set_page_config(page_title="Nhận diện cử chỉ tay", layout="wide")
st.title("✋ Phân loại cử chỉ tay theo thời gian thực")

with st.sidebar:
    ckpt = st.text_input("Checkpoint", "outputs/mobilenet_v2_finetune/best.pt")
    mode = st.radio("Chế độ", ["Nhận diện", "Oẳn tù tì"])
    show_cam = st.checkbox("Hiển thị Grad-CAM", value=False)
    thresh = st.slider("Ngưỡng tin cậy", 0.0, 1.0, 0.6, 0.05)
    window = st.slider("Số frame làm mượt", 1, 15, 7)
    run = st.checkbox("Bật camera", value=False)


@st.cache_resource
def get_predictor(path, window):
    return GesturePredictor(path, smooth_window=window)


if "game" not in st.session_state:
    st.session_state.game = RPSGame()
game = st.session_state.game

if mode == "Oẳn tù tì" and st.button("🎮 Chơi (đếm ngược 3 giây)"):
    game.start()

col_cam, col_info = st.columns([3, 2])
frame_ph = col_cam.empty()
label_ph = col_info.empty()
cam_ph = col_info.empty()
chart_ph = col_info.empty()
game_ph = col_cam.empty()

if run:
    predictor = get_predictor(ckpt, window)
    predictor.conf_thresh = thresh
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                st.error("Không đọc được webcam")
                break
            frame = cv2.flip(frame, 1)
            res = predictor.process(frame, with_cam=show_cam)
            frame_ph.image(cv2.cvtColor(draw_result(frame, res), cv2.COLOR_BGR2RGB), channels="RGB")

            if res is None:
                label_ph.info("Không thấy bàn tay")
            else:
                name = VI_NAMES.get(res["label"], "Không chắc chắn")
                label_ph.success(f"**{name}** ({res['label']}) — độ tin cậy {res['conf']:.1%}")
                chart_ph.bar_chart(pd.Series(res["probs"], index=predictor.classes))
                if show_cam and res["cam"] is not None:
                    cam_ph.image(overlay_cam(cv2.resize(res["crop"], (224, 224)), res["cam"]),
                                 caption="Grad-CAM", width=224)

            if mode == "Oẳn tù tì":
                msg = game.update(res["label"] if res else None)
                game_ph.markdown(f"### {msg}\n{game.score_text()}")
    finally:
        cap.release()
else:
    frame_ph.info("Tick 'Bật camera' ở thanh bên để bắt đầu")
