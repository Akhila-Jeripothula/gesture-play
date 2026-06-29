"""
subway_hand_controller.py
Play Subway Surfers (or any arrow-key game) with hand gestures.

Requirements:
  pip install opencv-python mediapipe pyautogui numpy

Run:
  python subway_hand_controller.py

How to use:
  1. Run the script.
  2. Hold your hand steady in front of the camera for ~2 seconds while it calibrates.
  3. Click the game window so it has keyboard focus.
  4. The script will auto-start sending keys. Move your hand left/right/up/down to control.
  5. Press 'q' in the camera window to quit, or 'r' to recalibrate.
"""

import time
import math
import cv2
import numpy as np
import pyautogui
import mediapipe as mp

# ---------------- CONFIG (tweak these if needed) ----------------
CAM_INDEX = 0                # webcam device (0 is usually default)
CALIBRATION_SECONDS = 2.0    # hold still to calibrate neutral hand position
AUTO_START_COUNTDOWN = 4.0   # seconds after calibration to click game and prepare
THRESHOLD_X_FRAC = 0.12      # fraction of frame width to trigger left/right
THRESHOLD_Y_FRAC = 0.12      # fraction of frame height to trigger up/down
COOLDOWN_SEC = 0.35          # cooldown between repeated same actions
DRAW_DEBUG = True            # draw landmarks & boxes on camera feed
# ----------------------------------------------------------------

pyautogui.FAILSAFE = False  # allow move without triggering fail-safe

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check CAM_INDEX and camera permissions.")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

TH_X_PX = int(W * THRESHOLD_X_FRAC)
TH_Y_PX = int(H * THRESHOLD_Y_FRAC)

def get_index_tip_xy(landmarks, w, h):
    """Return pixel (x,y) of index finger tip (landmark 8)."""
    lm = landmarks.landmark[8]
    return int(lm.x * w), int(lm.y * h)

def get_hand_center(landmarks, w, h):
    """Approximate hand center by averaging all landmark points."""
    xs = [lm.x * w for lm in landmarks.landmark]
    ys = [lm.y * h for lm in landmarks.landmark]
    return int(np.mean(xs)), int(np.mean(ys))

def press_key(key):
    """Safe wrapper to press a single key."""
    try:
        pyautogui.press(key)
    except Exception as e:
        print("pyautogui error:", e)

def draw_ui(frame, neutral, index_pt, status):
    if neutral is not None:
        # neutral point and threshold box
        cv2.circle(frame, neutral, 6, (0, 255, 0), -1)
        cv2.rectangle(
            frame,
            (neutral[0] - TH_X_PX, neutral[1] - TH_Y_PX),
            (neutral[0] + TH_X_PX, neutral[1] + TH_Y_PX),
            (0, 255, 0), 2
        )
    if index_pt is not None:
        cv2.circle(frame, index_pt, 6, (255, 255, 0), -1)
        if neutral is not None:
            cv2.line(frame, neutral, index_pt, (255, 255, 255), 1)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

print("Starting camera. Please hold your hand steady in front of the camera for calibration.")

# main state
neutral = None
samples = []
collecting = True
calib_start = time.time()
countdown_end = None
control_enabled = False
last_action_time = {"left":0,"right":0,"up":0,"down":0}
status_text = "Calibrating... hold your hand steady"

while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera read failed")
        break

    frame = cv2.flip(frame, 1)  # mirror
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    index_pt = None
    hand_center = None

    if results.multi_hand_landmarks:
        # use the first detected hand
        landmarks = results.multi_hand_landmarks[0]
        index_pt = get_index_tip_xy(landmarks, W, H)
        hand_center = get_hand_center(landmarks, W, H)
        if DRAW_DEBUG:
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Calibration phase
    if collecting:
        if hand_center is not None:
            samples.append(hand_center)
        if time.time() - calib_start >= CALIBRATION_SECONDS:
            if samples:
                neutral = tuple(np.mean(samples, axis=0).astype(int))
                status_text = f"Calibrated. Click game window. Auto-start in {int(AUTO_START_COUNTDOWN)}s"
                countdown_end = time.time() + AUTO_START_COUNTDOWN
            else:
                status_text = "Calibration failed (no hand). Press 'r' to retry."
            collecting = False
    else:
        # countdown to auto-start (gives you time to click the game window)
        if not control_enabled and countdown_end is not None:
            left = int(math.ceil(countdown_end - time.time()))
            if left > 0:
                status_text = f"Click inside the game — controls start in {left}s"
            else:
                control_enabled = True
                status_text = "Controls ON — move your hand to play!"
        # if controls enabled, detect gestures
        if control_enabled and hand_center is not None and neutral is not None:
            dx = hand_center[0] - neutral[0]
            dy = hand_center[1] - neutral[1]
            action = None
            # choose the dominant axis to avoid double triggers
            if abs(dx) >= abs(dy) and abs(dx) > TH_X_PX:
                action = "right" if dx > 0 else "left"
            elif abs(dy) > TH_Y_PX:
                action = "down" if dy > 0 else "up"

            now = time.time()
            if action and (now - last_action_time[action] >= COOLDOWN_SEC):
                # send keyboard press
                press_key({"left":"left","right":"right","up":"up","down":"down"}[action])
                last_action_time[action] = now
                status_text = f"Sent: {action.upper()}"
        elif not control_enabled:
            # waiting for countdown or for you to focus the game window
            pass

    if DRAW_DEBUG:
        draw_ui(frame, neutral, index_pt, status_text)

    cv2.imshow("Subway Hand Controller - Press Q to quit, R to recalibrate", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):          # quit
        break
    elif key == ord('r'):        # recalibrate
        samples = []
        collecting = True
        calib_start = time.time()
        countdown_end = None
        control_enabled = False
        neutral = None
        status_text = "Recalibrating... hold your hand steady"

# cleanup
cap.release()
cv2.destroyAllWindows()
