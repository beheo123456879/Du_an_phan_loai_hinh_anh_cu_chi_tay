"""MediaPipe định vị tay -> cắt vùng tay -> mô hình phân loại -> làm mượt qua nhiều frame."""
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image

from src.data.transforms import get_eval_transforms
from src.gradcam import GradCAM
from src.hand_detector import HandDetector
from src.models import get_gradcam_layer, load_checkpoint


class GesturePredictor:
    def __init__(self, ckpt_path, smooth_window=7, conf_thresh=0.6, margin=0.25):
        self.model, ckpt = load_checkpoint(ckpt_path, "cpu")
        self.model_name = ckpt["model_name"]
        self.classes = ckpt["classes"]
        self.tf = get_eval_transforms(ckpt["img_size"])
        self.detector = HandDetector(margin=margin)
        self.buffer = deque(maxlen=smooth_window)
        self.conf_thresh = conf_thresh
        self._gradcam = None

    @property
    def gradcam(self):
        if self._gradcam is None:
            self._gradcam = GradCAM(self.model, get_gradcam_layer(self.model, self.model_name))
        return self._gradcam

    def process(self, frame_bgr, with_cam=False):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes = self.detector.detect(rgb)
        if not boxes:
            self.buffer.clear()
            return None
        x1, y1, x2, y2 = boxes[0]
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        x = self.tf(Image.fromarray(crop)).unsqueeze(0)

        cam = None
        if with_cam:
            cam, _, probs = self.gradcam(x)
        else:
            with torch.no_grad():
                probs = torch.softmax(self.model(x), 1)[0].numpy()

        self.buffer.append(probs)
        avg = np.mean(self.buffer, axis=0)
        idx = int(avg.argmax())
        label = self.classes[idx] if avg[idx] >= self.conf_thresh else "unknown"
        return {"box": (x1, y1, x2, y2), "label": label, "conf": float(avg[idx]),
                "probs": avg, "crop": crop, "cam": cam}


def draw_result(frame, res):
    if res is None:
        return frame
    x1, y1, x2, y2 = res["box"]
    color = (0, 200, 0) if res["label"] != "unknown" else (0, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{res['label']} {res['conf']:.2f}", (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame
