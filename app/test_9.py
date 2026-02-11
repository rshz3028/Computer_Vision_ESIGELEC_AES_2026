import os
import time
import cv2
import numpy as np

import pygame
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ------------------ PATHS ------------------
MODEL_PATH = r"D:/MSc/Sem 2/Computer Vision/models/hand_landmarker.task"
SAMPLES_DIR = r"D:/MSc/Sem 2/Computer Vision/app/piano_samples"
WINDOW_NAME = "Virtual Piano (Real Samples, Top) - q quit"

# ------------------ PIANO (1 OCTAVE chromatic) ------------------
# File naming convention in SAMPLES_DIR:
# C4.wav Cs4.wav D4.wav Ds4.wav E4.wav F4.wav Fs4.wav G4.wav Gs4.wav A4.wav As4.wav B4.wav
WHITE_NOTES = ["C4", "D4", "E4", "F4", "G4", "A4", "B4"]
BLACK_NOTES = ["Cs4", "Ds4", "Fs4", "Gs4", "As4"]

# Black keys appear after these white indices: C, D, (skip E), F, G, A
BLACK_AFTER_WHITE_INDEX = [0, 1, 3, 4, 5]

# Keys at TOP
KEY_AREA_Y0_FRAC = 0.02
KEY_AREA_Y1_FRAC = 0.33
BLACK_KEY_HEIGHT_FRAC = 0.60
BLACK_KEY_WIDTH_FRAC = 0.62

# Press detection (TOP keys): move UP across press line
PRESS_LINE_FRAC_IN_KEYBED = 0.72
MIN_UP_VEL = 0.008
VEL_SMOOTH = 0.35
KEY_COOLDOWN_S = 0.18

# Fingers used
FINGER_TIPS = {"index": 8, "middle": 12}

DRAW_DEBUG = True

# ------------------ AUDIO (pygame samples) ------------------
def init_audio():
    # These settings are usually stable and low-latency
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=256)
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.set_num_channels(32)

def load_samples(samples_dir: str):
    # Map note name -> pygame Sound
    # White notes use C4..B4
    # Black notes use Cs4..As4
    notes = WHITE_NOTES + BLACK_NOTES
    sounds = {}
    missing = []
    for n in notes:
        path = os.path.join(samples_dir, f"{n}.wav")
        if not os.path.exists(path):
            missing.append(f"{n}.wav")
            continue
        sounds[n] = pygame.mixer.Sound(path)
    return sounds, missing

def play_note(sounds, note: str, volume: float = 0.9):
    s = sounds.get(note)
    if s is None:
        return
    s.set_volume(volume)
    s.play()  # non-blocking

# ------------------ GEOMETRY ------------------
def point_in_rect(px, py, rect):
    x0, y0, x1, y1 = rect
    return (x0 <= px <= x1) and (y0 <= py <= y1)

def build_piano_keys(w: int, h: int):
    y0 = int(h * KEY_AREA_Y0_FRAC)
    y1 = int(h * KEY_AREA_Y1_FRAC)
    white_h = y1 - y0
    white_w = w / 7.0

    white = []
    for i, note in enumerate(WHITE_NOTES):
        x0 = int(round(i * white_w))
        x1 = int(round((i + 1) * white_w)) - 1
        white.append({"note": note, "is_black": False, "rect": (x0, y0, x1, y1)})

    black = []
    bw = int(round(white_w * BLACK_KEY_WIDTH_FRAC))
    bh = int(round(white_h * BLACK_KEY_HEIGHT_FRAC))
    by1 = y0 + bh

    for bi, after_wi in enumerate(BLACK_AFTER_WHITE_INDEX):
        left = white[after_wi]["rect"]
        right = white[after_wi + 1]["rect"]
        cx = (left[2] + right[0]) // 2
        x0 = int(cx - bw // 2)
        x1 = int(cx + bw // 2)
        x0 = max(0, x0)
        x1 = min(w - 1, x1)
        black.append({"note": BLACK_NOTES[bi], "is_black": True, "rect": (x0, y0, x1, by1)})

    press_y = int(y0 + (y1 - y0) * PRESS_LINE_FRAC_IN_KEYBED)
    return black, white, press_y

def draw_piano(frame, black, white, press_y, active_notes):
    # White keys
    for k in white:
        x0, y0, x1, y1 = k["rect"]
        active = (k["note"] in active_notes)
        if active:
            cv2.rectangle(frame, (x0, y0), (x1, y1), (230, 230, 230), -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)
        cv2.putText(frame, k["note"], (x0 + 8, y0 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0) if active else (255, 255, 255), 2)

    # Black keys
    for k in black:
        x0, y0, x1, y1 = k["rect"]
        active = (k["note"] in active_notes)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (90, 90, 90) if active else (20, 20, 20), -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 1)
        cv2.putText(frame, k["note"], (x0 + 6, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Press line
    cv2.line(frame, (0, press_y), (frame.shape[1] - 1, press_y), (255, 255, 255), 2)
    cv2.putText(frame, "press = move UP across line", (10, press_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

# ------------------ MAIN ------------------
def main():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Missing model file: {MODEL_PATH}")

    init_audio()
    sounds, missing = load_samples(SAMPLES_DIR)
    if missing:
        print("Missing sample files in:", SAMPLES_DIR)
        for m in missing:
            print("  -", m)
        print("\nPut WAVs named like C4.wav, Cs4.wav, ..., B4.wav into that folder.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    black = white = None
    press_y = None

    # finger_state[fid] = {"prev_y": float, "vy": float, "prev_below": bool, "last_play": {note: t}}
    finger_state = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if black is None:
            black, white, press_y = build_piano_keys(w, h)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        now = time.time()
        active_notes = set()

        tips = []
        if result.hand_landmarks:
            for hand_i, lm_list in enumerate(result.hand_landmarks):
                for name, tip_id in FINGER_TIPS.items():
                    lm = lm_list[tip_id]
                    x_px = int(lm.x * w)
                    y_px = int(lm.y * h)
                    fid = f"h{hand_i}_{name}"
                    tips.append((fid, x_px, y_px, float(lm.y)))

        seen = set()

        for fid, x, y, y_norm in tips:
            seen.add(fid)
            st = finger_state.get(fid)
            if st is None:
                st = {"prev_y": y_norm, "vy": 0.0, "prev_below": True, "last_play": {}}
                finger_state[fid] = st

            dy = y_norm - st["prev_y"]      # negative when moving UP
            st["vy"] = (1.0 - VEL_SMOOTH) * st["vy"] + VEL_SMOOTH * dy
            st["prev_y"] = y_norm

            below = (y > press_y)

            # hit-test: black first then white
            hit_note = None
            for k in black:
                if point_in_rect(x, y, k["rect"]):
                    hit_note = k["note"]
                    break
            if hit_note is None:
                for k in white:
                    if point_in_rect(x, y, k["rect"]):
                        hit_note = k["note"]
                        break

            if hit_note is not None:
                active_notes.add(hit_note)

                crossed_up = (st["prev_below"] is True) and (below is False)
                enough_up = (st["vy"] <= -MIN_UP_VEL)

                if crossed_up and enough_up:
                    last_t = st["last_play"].get(hit_note, 0.0)
                    if (now - last_t) >= KEY_COOLDOWN_S:
                        st["last_play"][hit_note] = now
                        play_note(sounds, hit_note, volume=0.9)

            st["prev_below"] = below

            if DRAW_DEBUG:
                cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)

        # cleanup
        if not seen:
            finger_state.clear()
        else:
            for fid in list(finger_state.keys()):
                if fid not in seen:
                    del finger_state[fid]

        draw_piano(frame, black, white, press_y, active_notes)

        if DRAW_DEBUG:
            cv2.putText(frame, "Real samples via pygame | q quit",
                        (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()
