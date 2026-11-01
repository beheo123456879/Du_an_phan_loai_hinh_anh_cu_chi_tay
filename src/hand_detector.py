"""MediaPipe Hands: CHỈ dùng để định vị bàn tay, việc phân loại do mô hình của mình làm."""
import mediapipe as mp

from src.data.crop import square_box


class HandDetector:
    def __init__(self, max_hands=1, min_conf=0.5, static=False, margin=0.25):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=static, max_num_hands=max_hands,
            min_detection_confidence=min_conf, min_tracking_confidence=0.5,
        )
        self.margin = margin

    def detect(self, rgb):
        """rgb: ndarray HxWx3 uint8 (RGB). Trả về list khung (x1, y1, x2, y2) vuông, đã mở rộng biên."""
        h, w = rgb.shape[:2]
        res = self.hands.process(rgb)
        boxes = []
        for lm in res.multi_hand_landmarks or []:
            xs = [p.x * w for p in lm.landmark]
            ys = [p.y * h for p in lm.landmark]
            boxes.append(square_box(min(xs), min(ys), max(xs), max(ys), w, h, self.margin))
        return boxes

    def close(self):
        self.hands.close()
