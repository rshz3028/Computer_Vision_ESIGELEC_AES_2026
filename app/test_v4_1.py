import cv2
import mediapipe as mp
import numpy as np

import os
import csv
import time
import math
import random
import datetime

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------- helpers ----------------
def clampi(v, lo, hi):
    return max(lo, min(hi, v))

def now_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def sanitize_nickname(s: str) -> str:
    s = s.replace("\n", "").replace("\r", "").replace(",", "").replace("\t", "")
    s = s.strip()
    if not s:
        s = "player"
    return s[:20]

def resources_dir_path():
    return os.environ.get("RESOURCES_DIR", "resources")

def score_file_path():
    return os.path.join(resources_dir_path(), "scores.csv")

def ensure_resources_dir_exists():
    os.makedirs(resources_dir_path(), exist_ok=True)

def append_score(name: str, score: int):
    ensure_resources_dir_exists()
    try:
        with open(score_file_path(), "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([name, score, now_timestamp()])
    except Exception as e:
        print("WARN: Could not write scores file:", score_file_path(), e)

def load_scores_sorted():
    entries = []
    try:
        with open(score_file_path(), "r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            for row in r:
                if len(row) < 2:
                    continue
                name = row[0]
                try:
                    sc = int(row[1])
                except:
                    continue
                t = row[2] if len(row) >= 3 else ""
                entries.append((name, sc, t))
    except FileNotFoundError:
        return []
    except Exception as e:
        print("WARN: Could not read scores file:", score_file_path(), e)
        return []
    entries.sort(key=lambda x: x[1], reverse=True)
    return entries

def beep_pop():
    try:
        import winsound
        winsound.Beep(1200, 60)
    except:
        pass

def draw_text_outlined(img, text, org, scale, thickness, fg, outline):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, outline, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA)

def dist2(ax, ay, bx, by):
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy

# ---------------- balloons ----------------
def bright_color():
    while True:
        b = random.randint(140, 255)
        g = random.randint(140, 255)
        r = random.randint(140, 255)
        if b + g + r >= 600:
            return (b, g, r)

class Balloon:
    __slots__ = ("x", "y", "vx", "vy", "r", "color")
    def __init__(self, x, y, vx, vy, r, color):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.r = int(r)
        self.color = color

def random_balloon(W, H):
    r = random.randint(10, 26)
    x = random.randint(r, W - r)
    y = random.randint(r, H - r)
    ang = random.uniform(0.0, 2.0 * math.pi)
    sp = random.uniform(2.0, 5.0)
    vx = math.cos(ang) * sp
    vy = math.sin(ang) * sp
    return Balloon(x, y, vx, vy, r, bright_color())

def out_of_bounds(b: Balloon, W, H):
    return (b.x + b.r < 0) or (b.x - b.r > W) or (b.y + b.r < 0) or (b.y - b.r > H)

# ---------------- index fingertip (MediaPipe Tasks) ----------------
class IndexTipTracker:
    def __init__(self, model_path, max_num_hands=1):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.tip_id = 8  # INDEX_FINGER_TIP

    def get_index_tip(self, frame_bgr, mirror: bool):
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = int(time.time() * 1000)
        res = self.landmarker.detect_for_video(mp_image, ts_ms)

        if not res.hand_landmarks:
            return False, (-1, -1)

        lm = res.hand_landmarks[0][self.tip_id]
        x = int(round(lm.x * W))
        y = int(round(lm.y * H))

        if mirror:
            x = W - 1 - x

        x = clampi(x, 0, W - 1)
        y = clampi(y, 0, H - 1)
        return True, (x, y)

    def close(self):
        self.landmarker.close()

# ---------------- main ----------------
def main():
    nickname = sanitize_nickname(input("Enter nickname: "))

    # IMPORTANT: model file path
    MODEL_PATH = r"D:/MSc/Sem 2/Computer Vision/models/hand_landmarker.task"
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Missing model file:", MODEL_PATH)
        print("Download it to that path, then run again.")
        return

    W, H = 640, 480
    mirror_display = True

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    cv2.namedWindow("Game", cv2.WINDOW_AUTOSIZE)

    tracker = IndexTipTracker(model_path=MODEL_PATH, max_num_hands=1)

    game_seconds = 60.0
    t0 = time.time()

    score = 0
    balloons = []
    max_balloons = 18
    spawn_prob = 0.25

    cursor_color = (0, 255, 0)
    cursor_r = 8
    brush_r = 6

    paint = np.zeros((H, W, 3), dtype=np.uint8)
    has_prev = False
    prev_pt = (0, 0)

    has_smooth = False
    smooth_x, smooth_y = 0.0, 0.0
    alpha = 0.75

    bg_color = [230, 230, 230]

    while True:
        elapsed = time.time() - t0
        remaining = game_seconds - elapsed
        if remaining <= 0:
            break

        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame = cv2.resize(frame, (W, H))
        if mirror_display:
            frame = cv2.flip(frame, 1)

        if len(balloons) < max_balloons and random.random() < spawn_prob:
            balloons.append(random_balloon(W, H))

        for b in balloons:
            b.x += b.vx
            b.y += b.vy

        balloons = [b for b in balloons if not out_of_bounds(b, W, H)]

        # frame already flipped -> mirror=False
        has_c, (cx, cy) = tracker.get_index_tip(frame, mirror=False)

        ui = np.full((H, W, 3), bg_color, dtype=np.uint8)

        if has_c:
            if not has_smooth:
                smooth_x, smooth_y = float(cx), float(cy)
                has_smooth = True
            else:
                smooth_x = alpha * smooth_x + (1.0 - alpha) * float(cx)
                smooth_y = alpha * smooth_y + (1.0 - alpha) * float(cy)

            sx = clampi(int(round(smooth_x)), 0, W - 1)
            sy = clampi(int(round(smooth_y)), 0, H - 1)
            spt = (sx, sy)

            hit = -1
            for i, b in enumerate(balloons):
                rr = float(b.r + cursor_r)
                if dist2(sx, sy, b.x, b.y) <= rr * rr:
                    hit = i
                    break

            if hit >= 0:
                beep_pop()
                popped_color = balloons[hit].color
                balloons.pop(hit)
                score += 1

                bg_color = [int(popped_color[0]), int(popped_color[1]), int(popped_color[2])]
                paint[:] = 0
                has_prev = False
            else:
                if not has_prev:
                    cv2.circle(paint, spt, brush_r, cursor_color, -1, cv2.LINE_AA)
                else:
                    cv2.line(paint, prev_pt, spt, cursor_color, brush_r * 2, cv2.LINE_AA)
                prev_pt = spt
                has_prev = True

            cv2.circle(ui, spt, cursor_r, cursor_color, -1, cv2.LINE_AA)
            cv2.circle(ui, spt, cursor_r, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            has_prev = False
            has_smooth = False

        for b in balloons:
            p = (int(round(b.x)), int(round(b.y)))
            cv2.circle(ui, p, b.r, b.color, -1, cv2.LINE_AA)
            cv2.circle(ui, p, b.r, (255, 255, 255), 2, cv2.LINE_AA)

        draw_text_outlined(ui, f"Player: {nickname}", (10, 30), 0.8, 2, (0, 0, 0), (255, 255, 255))
        draw_text_outlined(ui, f"Score: {score}", (10, 60), 0.9, 2, (0, 0, 0), (255, 255, 255))
        draw_text_outlined(ui, f"Time: {int(max(0.0, remaining))}", (10, 90), 0.9, 2, (0, 0, 0), (255, 255, 255))

        out = cv2.addWeighted(ui, 0.85, paint, 0.95, 0.0)

        cv2.imshow("Game", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    append_score(nickname, score)
    scores = load_scores_sorted()

    final = np.full((H, W, 3), bg_color, dtype=np.uint8)

    draw_text_outlined(final, "TIME UP!", (20, 60), 1.3, 3, (0, 0, 0), (255, 255, 255))
    draw_text_outlined(final, f"SCORE: {score}", (20, 110), 1.0, 2, (0, 0, 0), (255, 255, 255))
    draw_text_outlined(final, "SCOREBOARD (Top 10)", (20, 170), 0.9, 2, (0, 0, 0), (255, 255, 255))

    y = 210
    for i, (name, sc, _) in enumerate(scores[:10]):
        line = f"{i+1}. {name} - {sc}"
        draw_text_outlined(final, line, (20, y), 0.8, 2, (0, 0, 0), (255, 255, 255))
        y += 32

    draw_text_outlined(final, "Press any key to exit", (20, H - 20), 0.7, 2, (0, 0, 0), (255, 255, 255))

    cv2.imshow("Game", final)
    cv2.waitKey(0)

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
