import time
import math
import cv2
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# -------------------- PATHS / SETTINGS --------------------
MODEL_PATH = r"D:/MSc/Sem 2/Computer Vision/resources/models/hand_landmarker.task"
CAM_INDEX = 0

WINDOW_NAME = "Hand Mouse (q quit, p pause)"

# Cursor smoothing (0..1). Higher = follows faster, lower = smoother.
SMOOTH_SLOW = 0.12      # steady hand smoothing
SMOOTH_FAST = 0.55      # when you move quickly
FAST_DIST_PX = 120      # start speeding up after this many pixels of error
MAX_STEP_PX = 260
DEADZONE_PX = 5
ACCEL = 1.7

DRAG_HOLD_S = 0.35      # pinch held longer than this => drag
CLICK_COOLDOWN_S = 0.20

# Map camera space -> screen space
MARGIN = 0.02  # normalized margin on each side (0..0.3)

# ---- Pinch detection (IMPROVED) ----
# We use a ratio: dist(thumb_tip, index_tip) / hand_scale
# This adapts to hand size & camera distance.
PINCH_ON = 0.22
PINCH_OFF = 0.28

# Click timing (for "tap" pinch)
CLICK_MIN_HOLD_S = 0.05   # debounce accidental micro-pinches
CLICK_MAX_HOLD_S = 0.90   # pinch held less than this => click

# Drag mode: if True, pinch hold becomes click+drag (mouseDown while pinched)
ENABLE_DRAG = True

# Fist detection (pause) threshold (hysteresis):
FIST_ON = 0.10
FIST_OFF = 0.13

DRAW_DEBUG = True
# ----------------------------------------------------------


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def norm_dist(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)


def fist_score(lm):
    """
    Simple fist score:
    Average fingertip distance (8,12,16,20) to wrist (0).
    Smaller => more fist-like.
    """
    wrist = lm[0]
    tips = [lm[8], lm[12], lm[16], lm[20]]
    return sum(norm_dist(t, wrist) for t in tips) / len(tips)


def apply_margin(x, y, margin=MARGIN):
    # allow slight overshoot, then clamp & remap
    x = clamp(x, -0.05, 1.05)
    y = clamp(y, -0.05, 1.05)

    x = clamp(x, margin, 1.0 - margin)
    y = clamp(y, margin, 1.0 - margin)

    x = (x - margin) / (1.0 - 2 * margin)
    y = (y - margin) / (1.0 - 2 * margin)
    return x, y


# -------- Improved pinch helpers --------
def hand_scale(lm):
    # Stable reference length: wrist (0) to middle MCP (9)
    return max(1e-6, norm_dist(lm[0], lm[9]))


def pinch_ratio(lm):
    # Scale-normalized pinch distance
    return norm_dist(lm[4], lm[8]) / hand_scale(lm)


def thumb_is_actively_pinching(lm):
    """
    Extra intent check to reduce false pinches:
    Thumb tip (4) should be closer to index tip (8) than thumb IP (3) is.
    That implies the thumb is folding in towards the index.
    """
    d_tip = norm_dist(lm[4], lm[8])
    d_ip = norm_dist(lm[3], lm[8])
    return d_tip < d_ip
# ----------------------------------------


def main():
    pyautogui.FAILSAFE = False

    screen_w, screen_h = pyautogui.size()

    cap = cv2.VideoCapture(CAM_INDEX)
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

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # smoothed cursor position
    sx, sy = screen_w / 2, screen_h / 2

    # pinch state
    pinched = False
    pinch_start_t = 0.0
    dragging = False
    last_click_t = 0.0

    # pause state
    manual_pause = False
    fist_state = False  # hysteresis internal

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000)

        result = landmarker.detect_for_video(mp_image, ts_ms)

        tip_px = None
        pinch_d = None
        pinch_ok = False
        fscore = None

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            lm = result.hand_landmarks[0]

            # index tip controls cursor
            idx_tip = lm[8]
            x_norm, y_norm = apply_margin(float(idx_tip.x), float(idx_tip.y), MARGIN)

            target_x = x_norm * screen_w
            target_y = y_norm * screen_h

            # smooth mouse-like motion
            dx = target_x - sx
            dy = target_y - sy
            dist = math.hypot(dx, dy)

            if dist < DEADZONE_PX:
                dx = dy = 0.0
            else:
                # acceleration
                scale = (dist / 140.0) ** (ACCEL - 1.0)
                dx *= scale
                dy *= scale

                # clamp step
                step = math.hypot(dx, dy)
                if step > MAX_STEP_PX:
                    s = MAX_STEP_PX / step
                    dx *= s
                    dy *= s

            t = min(1.0, dist / FAST_DIST_PX)
            smooth = SMOOTH_SLOW + (SMOOTH_FAST - SMOOTH_SLOW) * t

            sx += dx * smooth
            sy += dy * smooth

            tip_px = (int(idx_tip.x * w), int(idx_tip.y * h))

            # ---- Improved pinch detection (ratio + intent check) ----
            pinch_d = pinch_ratio(lm)                # normalized distance
            pinch_ok = thumb_is_actively_pinching(lm)
            # --------------------------------------------------------

            # fist detection (score + hysteresis)
            fscore = fist_score(lm)
            if (not fist_state) and (fscore <= FIST_ON):
                fist_state = True
            elif fist_state and (fscore >= FIST_OFF):
                fist_state = False

        paused = manual_pause or fist_state
        now = time.time()

        # Cursor move
        if (tip_px is not None) and (not paused):
            pyautogui.moveTo(int(sx), int(sy), duration=0)

        # If paused or no hand: release drag and reset pinch
        if paused or (pinch_d is None):
            if dragging:
                pyautogui.mouseUp()
                dragging = False
            pinched = False
        else:
            # pinch hysteresis transitions
            if (not pinched) and pinch_ok and (pinch_d <= PINCH_ON):
                pinched = True
                pinch_start_t = now

            elif pinched and ((pinch_d >= PINCH_OFF) or (not pinch_ok)):
                # pinch released (or thumb intent lost): decide click vs end-drag
                held = now - pinch_start_t
                pinched = False

                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                else:
                    # click only if it was a real quick pinch
                    if (CLICK_MIN_HOLD_S <= held <= CLICK_MAX_HOLD_S) and ((now - last_click_t) >= CLICK_COOLDOWN_S):
                        pyautogui.click()
                        last_click_t = now

            # if still pinched: possibly start drag after hold time
            if pinched and ENABLE_DRAG and (not dragging):
                if (now - pinch_start_t) >= DRAG_HOLD_S:
                    pyautogui.mouseDown()
                    dragging = True

        # Debug UI
        if DRAW_DEBUG:
            if tip_px is not None:
                cv2.circle(frame, tip_px, 10, (0, 255, 0), -1)

            status = "PAUSED" if paused else "LIVE"
            cv2.putText(frame, f"{status}  manual:{manual_pause} fist:{fist_state}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if pinch_d is not None:
                cv2.putText(frame, f"pinch_ratio: {pinch_d:.3f} ok:{pinch_ok} pinched:{pinched} drag:{dragging}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if fscore is not None:
                cv2.putText(frame, f"fist_score: {fscore:.3f}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            mx0, my0 = int(MARGIN * w), int(MARGIN * h)
            mx1, my1 = int((1 - MARGIN) * w), int((1 - MARGIN) * h)
            cv2.rectangle(frame, (mx0, my0), (mx1, my1), (255, 255, 255), 1)

        cv2.imshow(WINDOW_NAME, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('p'):
            manual_pause = not manual_pause
            if manual_pause and dragging:
                pyautogui.mouseUp()
                dragging = False

    # cleanup
    if dragging:
        pyautogui.mouseUp()
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
