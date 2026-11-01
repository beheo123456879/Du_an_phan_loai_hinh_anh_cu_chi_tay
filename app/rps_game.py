"""Oẳn tù tì người vs máy: fist = búa, palm = bao, peace = kéo."""
import math
import random
import time

from src.config import RPS_MAP

MOVES = ["rock", "paper", "scissors"]
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
MOVE_TEXT = {"rock": "Bua", "paper": "Bao", "scissors": "Keo"}  # OpenCV không vẽ được dấu tiếng Việt


def judge(player, computer):
    if player == computer:
        return "draw"
    return "win" if BEATS[player] == computer else "lose"


class RPSGame:
    def __init__(self, countdown=3.0):
        self.countdown = countdown
        self.score = {"win": 0, "lose": 0, "draw": 0}
        self.state = "idle"
        self.t0 = None
        self.message = "Nhan SPACE de choi"

    def start(self):
        self.state = "countdown"
        self.t0 = time.time()

    def update(self, gesture_label):
        """Gọi mỗi frame với nhãn cử chỉ hiện tại; trả về dòng chữ cần hiển thị."""
        if self.state != "countdown":
            return self.message
        remain = self.countdown - (time.time() - self.t0)
        if remain > 0:
            return f"Oan tu ti... {math.ceil(remain)}"

        player = RPS_MAP.get(gesture_label)
        if player is None:
            self.message = "Khong nhan ra Bua/Keo/Bao - SPACE de thu lai"
        else:
            computer = random.choice(MOVES)
            result = judge(player, computer)
            self.score[result] += 1
            verdict = {"win": "BAN THANG!", "lose": "MAY THANG!", "draw": "HOA"}[result]
            self.message = f"Ban: {MOVE_TEXT[player]} - May: {MOVE_TEXT[computer]} => {verdict}"
        self.state = "result"
        return self.message

    def score_text(self):
        return f"Thang {self.score['win']} | Thua {self.score['lose']} | Hoa {self.score['draw']}"
