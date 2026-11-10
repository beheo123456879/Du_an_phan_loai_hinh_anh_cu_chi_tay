"""Demo webcam bằng cửa sổ OpenCV (chạy trên máy cá nhân, không chạy trên Kaggle).

    python -m app.demo_opencv --ckpt outputs/mobilenet_v2_finetune/best.pt

Phím: q = thoát | m = đổi chế độ (nhận diện / oẳn tù tì) | g = bật/tắt Grad-CAM | SPACE = chơi
"""
import argparse
import time

import cv2

from app.predictor import GesturePredictor, draw_result
from app.rps_game import RPSGame
from src.gradcam import overlay_cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--window", type=int, default=7, help="số frame để làm mượt")
    ap.add_argument("--thresh", type=float, default=0.6)
    args = ap.parse_args()

    predictor = GesturePredictor(args.ckpt, args.window, args.thresh)
    game = RPSGame()
    mode, show_cam = "classify", False
    cap = cv2.VideoCapture(args.camera)
    prev = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = predictor.process(frame, with_cam=show_cam)
        draw_result(frame, res)

        if show_cam and res is not None and res["cam"] is not None:
            vis = cv2.cvtColor(overlay_cam(cv2.resize(res["crop"], (224, 224)), res["cam"]), cv2.COLOR_RGB2BGR)
            vis = cv2.resize(vis, (180, 180))
            frame[10:190, frame.shape[1] - 190:frame.shape[1] - 10] = vis

        if mode == "rps":
            text = game.update(res["label"] if res else None)
            cv2.putText(frame, text, (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, game.score_text(), (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        now = time.perf_counter()
        fps = 1 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"FPS {fps:.0f} | mode: {mode} | cam: {'on' if show_cam else 'off'}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Hand Gesture Classification", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            mode = "rps" if mode == "classify" else "classify"
        if key == ord("g"):
            show_cam = not show_cam
        if key == ord(" ") and mode == "rps":
            game.start()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
