import cv2
import os
import time
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# -------- Settings --------
IMAGE_DIR = r"D:/MSc/Sem 2/Computer Vision/resources/parsing"
MODEL_PATH = r"D:/MSc/Sem 2/Computer Vision/resources/models/hand_landmarker.task"
MAX_IMAGES = 10
WINDOW_NAME = "Swipe Gallery (q to quit)"

COOLDOWN_S = 0.6
HISTORY_LEN = 6
SWIPE_MIN_DX = 0.25
SWIPE_MAX_DT = 0.35

# Pinch zoom
ZOOM_MIN = 1.0
ZOOM_MAX = 4.0
PINCH_SMOOTH = 0.25       # 0..1 higher = faster response
PINCH_DEADZONE = 0.012    # normalized distance change (relative) to ignore jitter
ZOOM_SENS = 2.2           # higher = more zoom per pinch change
PINCH_COOLDOWN_S = 0.05   # small cooldown between zoom updates
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


def render_zoomed_to_window(img, out_w, out_h, zoom, center_xy=None):
    """
    zoom=1.0 -> fit whole image to window
    zoom>1.0 -> crop around center and scale up

    center_xy: (cx, cy) in normalized [0..1] relative to the *window* / frame.
               We map it to the image coordinates to choose crop center.
    """
    ih, iw = img.shape[:2]

    # First, compute the "fit" scale to fill window without cropping (letterbox on black)
    fit_scale = min(out_w / iw, out_h / ih)
    fit_w, fit_h = int(iw * fit_scale), int(ih * fit_scale)

    # We will zoom by cropping *in original image space*.
    # Effective zoom relative to the fitted display:
    z = float(np.clip(zoom, ZOOM_MIN, ZOOM_MAX))

    # Determine crop size in image pixels.
    # When z=1, crop is full image; when z=2, crop is half width/height, etc.
    crop_w = int(iw / z)
    crop_h = int(ih / z)
    crop_w = max(20, min(crop_w, iw))
    crop_h = max(20, min(crop_h, ih))

    # Crop center
    if center_xy is None:
        cx_img, cy_img = iw // 2, ih // 2
    else:
        cxn, cyn = center_xy
        cxn = float(np.clip(cxn, 0.0, 1.0))
        cyn = float(np.clip(cyn, 0.0, 1.0))
        cx_img = int(cxn * iw)
        cy_img = int(cyn * ih)

    x0 = int(np.clip(cx_img - crop_w // 2, 0, iw - crop_w))
    y0 = int(np.clip(cy_img - crop_h // 2, 0, ih - crop_h))
    crop = img[y0:y0 + crop_h, x0:x0 + crop_w]

    # Scale crop up to fit size, then letterbox into window
    scaled = cv2.resize(crop, (fit_w, fit_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    ox = (out_w - fit_w) // 2
    oy = (out_h - fit_h) // 2
    canvas[oy:oy + fit_h, ox:ox + fit_w] = scaled
    return canvas


def main():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Missing model file: {MODEL_PATH}")

    images = load_images(IMAGE_DIR, MAX_IMAGES)
    idx = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

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

    # Swipe history (time, x_px, y_px)
    hist = []
    last_swipe_t = 0.0

    # Zoom state
    zoom = 1.0
    pinch_ref = None        # reference distance (normalized)
    pinch_smooth = None     # smoothed pinch distance
    last_pinch_t = 0.0
    zoom_center = None      # normalized (0..1, 0..1) from pinch midpoint

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        tip = None
        pinch_dist = None
        pinch_mid = None

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            lm = result.hand_landmarks[0]

            # Index fingertip for swipe (8)
            idx_tip = lm[8]
            tip = (int(idx_tip.x * fw), int(idx_tip.y * fh))
            cv2.circle(frame, tip, 10, (0, 255, 0), -1)

            # Pinch: thumb tip (4) and index tip (8)
            th = lm[4]
            ix = lm[8]
            dx = (ix.x - th.x)
            dy = (ix.y - th.y)
            pinch_dist = (dx * dx + dy * dy) ** 0.5  # normalized (0..~1)

            pinch_mid = ((ix.x + th.x) * 0.5, (ix.y + th.y) * 0.5)

            # draw pinch points
            cv2.circle(frame, (int(th.x * fw), int(th.y * fh)), 8, (255, 0, 0), -1)
            cv2.circle(frame, (int(ix.x * fw), int(ix.y * fh)), 8, (255, 0, 0), -1)
            cv2.line(frame,
                     (int(th.x * fw), int(th.y * fh)),
                     (int(ix.x * fw), int(ix.y * fh)),
                     (255, 0, 0), 2)

        now = time.time()

        # ---- Swipe history update ----
        if tip is not None:
            hist.append((now, tip[0], tip[1]))
            if len(hist) > HISTORY_LEN:
                hist.pop(0)
        else:
            hist.clear()

        # ---- Swipe detection ----
        swipe = None
        if len(hist) >= 2 and (now - last_swipe_t) >= COOLDOWN_S:
            t0, x0, y0 = hist[0]
            t1, x1, y1 = hist[-1]
            dt = t1 - t0
            dx_px = x1 - x0
            if 0 < dt <= SWIPE_MAX_DT and abs(dx_px) >= int(SWIPE_MIN_DX * fw):
                swipe = "RIGHT" if dx_px > 0 else "LEFT"

        if swipe:
            if swipe == "LEFT":
                idx = (idx + 1) % len(images)
            else:
                idx = (idx - 1) % len(images)
            last_swipe_t = now
            hist.clear()
            # reset zoom when switching image (optional; comment out if you want zoom to persist)
            zoom = 1.0
            pinch_ref = None
            pinch_smooth = None
            zoom_center = None

        # ---- Pinch zoom update ----
        # We only update zoom if pinch_dist exists (hand present)
        if pinch_dist is None:
            pinch_ref = None
            pinch_smooth = None
        else:
            # smooth pinch distance to reduce jitter
            if pinch_smooth is None:
                pinch_smooth = pinch_dist
            else:
                pinch_smooth = (1.0 - PINCH_SMOOTH) * pinch_smooth + PINCH_SMOOTH * pinch_dist

            # if first time pinching, set reference
            if pinch_ref is None:
                pinch_ref = pinch_smooth

            # Only update zoom at a small rate limit
            if (now - last_pinch_t) >= PINCH_COOLDOWN_S:
                d = pinch_smooth - pinch_ref  # positive when fingers separate
                # deadzone to ignore tiny tremor
                if abs(d) >= PINCH_DEADZONE:
                    # convert change into zoom factor increment
                    zoom_delta = d * ZOOM_SENS
                    zoom = float(np.clip(zoom + zoom_delta, ZOOM_MIN, ZOOM_MAX))

                    # update reference gradually so continuous pinch keeps working
                    pinch_ref = pinch_smooth

                    # zoom around pinch midpoint
                    if pinch_mid is not None:
                        zoom_center = (pinch_mid[0], pinch_mid[1])

                last_pinch_t = now

        # ---- Render current image with zoom ----
        disp_w, disp_h = fw, fh
        canvas = render_zoomed_to_window(images[idx][1], disp_w, disp_h, zoom, zoom_center)

        # camera preview
        preview = cv2.resize(frame, (int(disp_w * 0.28), int(disp_h * 0.28)))
        ph, pw = preview.shape[:2]
        canvas[10:10 + ph, 10:10 + pw] = preview

        # UI text
        path = os.path.basename(images[idx][0])
        cv2.putText(canvas, f"{idx + 1}/{len(images)}  {path}", (10, disp_h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f"zoom: {zoom:.2f}x", (10, disp_h - 45),
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
