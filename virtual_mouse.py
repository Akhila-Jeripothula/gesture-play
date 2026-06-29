import cv2
import mediapipe as mp
import pyautogui

# Initialize hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_w, screen_h = pyautogui.size()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    frame_h, frame_w, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of index finger tip (landmark 8)
            index_finger = hand_landmarks.landmark[8]
            x = int(index_finger.x * frame_w)
            y = int(index_finger.y * frame_h)

            # Move mouse (map camera coords to screen coords)
            screen_x = int(index_finger.x * screen_w)
            screen_y = int(index_finger.y * screen_h)
            pyautogui.moveTo(screen_x, screen_y)

            # Check if thumb (landmark 4) is near index finger (landmark 8) -> Click
            thumb = hand_landmarks.landmark[4]
            thumb_x = int(thumb.x * frame_w)
            thumb_y = int(thumb.y * frame_h)

            # If distance between index and thumb is small = click
            if abs(x - thumb_x) < 30 and abs(y - thumb_y) < 30:
                pyautogui.click()
                pyautogui.sleep(1)

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
