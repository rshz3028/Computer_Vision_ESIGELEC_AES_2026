import cv2
import os
import time
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# -------- Settings --------
IMAGE_DIR = r"D:/MSc/Sem 2/Computer Vision/resources/parsing"   # <-- change this if needed
MODEL_PATH = r"D:/MSc/Sem 2/Computer Vision/resources/models/hand_landmarker.task"
MAX_IMAGES = 10
WINDOW_NAME = "Swipe Gallery (press q to quit)"

COOLDOWN_S = 0.6            # min time between swipes
HISTORY_LEN = 6             # frames used for swipe detection
SWIPE_MIN_DX = 0.25         # fraction of frame width traveled to count as swipe
SWIPE_MAX_DT = 0.35         # seconds max for that movement window
# -------------------------


def load_images(folder, max_images=10):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]
    files = files[:max_images]
    if not files:
        raise RuntimeError(f"No images found in: {folder}")
    imgs = []
    for p in files:
        img = cv2.imread(p)
        if img is None:
            continue
        imgs.append((p, img))
    if not imgs:
        raise RuntimeError("Images found but none could be read by OpenCV.")
    return imgs


def fit_to_window(img, w, h):
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def main():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            "Missing model file:\n"
            f"  {MODEL_PATH}\n\n"
            "Download it (PowerShell):\n"
            "  iwr -Uri \"https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task\" "
            f"-OutFile \"{MODEL_PATH}\"\n"
        )

    images = load_images(IMAGE_DIR, MAX_IMAGES)
    idx = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # Create HandLandmarker (Tasks API)
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    # Track fingertip history: list of (t, x_px, y_px)
    hist = []
    last_swipe_t = 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        # MediaPipe Tasks expects RGB mp.Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # timestamp must be increasing (ms)
        ts_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        tip = None
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            lm = result.hand_landmarks[0]
            tip_lm = lm[8]  # index fingertip
            tip = (int(tip_lm.x * fw), int(tip_lm.y * fh))
            cv2.circle(frame, tip, 10, (0, 255, 0), -1)

        # Update history
        now = time.time()
        if tip is not None:
            hist.append((now, tip[0], tip[1]))
            if len(hist) > HISTORY_LEN:
                hist.pop(0)
        else:
            hist.clear()

        # Swipe detection
        swipe = None
        if len(hist) >= 2 and (now - last_swipe_t) >= COOLDOWN_S:
            t0, x0, y0 = hist[0]
            t1, x1, y1 = hist[-1]
            dt = t1 - t0
            dx = x1 - x0

            if 0 < dt <= SWIPE_MAX_DT:
                if abs(dx) >= int(SWIPE_MIN_DX * fw):
                    swipe = "RIGHT" if dx > 0 else "LEFT"

        if swipe:
            if swipe == "LEFT":
                idx = (idx + 1) % len(images)
            else:
                idx = (idx - 1) % len(images)
            last_swipe_t = now
            hist.clear()

        # Render: current image full, camera preview small corner
        disp_w, disp_h = fw, fh
        canvas = fit_to_window(images[idx][1], disp_w, disp_h)

        preview = cv2.resize(frame, (int(disp_w * 0.28), int(disp_h * 0.28)))
        ph, pw = preview.shape[:2]
        canvas[10:10 + ph, 10:10 + pw] = preview

        # UI text
        path = os.path.basename(images[idx][0])
        cv2.putText(canvas, f"{idx + 1}/{len(images)}  {path}", (10, disp_h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if swipe:
            cv2.putText(canvas, f"SWIPE {swipe}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        cv2.imshow(WINDOW_NAME, canvas)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
