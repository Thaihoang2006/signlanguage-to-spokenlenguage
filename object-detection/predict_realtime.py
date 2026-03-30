import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import joblib
import os
import time

# Load model đã train
model = joblib.load("hand_model.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Mở webcam
cap = cv2.VideoCapture(0)

# Biến điều khiển auto đọc
current_label = ""
candidate_label = ""
candidate_start_time = 0
last_spoken_label = ""

speak_hold_time = 0.8   # giữ label ổn định 1 giây thì đọc
no_hand_reset_time = 1.0  # mất tay 1 giây thì cho phép đọc lại
last_hand_seen_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    detected_label = ""

    if result.multi_hand_landmarks:
        last_hand_seen_time = time.time()

        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Lấy dữ liệu keypoints
            row = []
            for lm in hand_landmarks.landmark:
                row.append(lm.x)
                row.append(lm.y)

            # Dự đoán
            prediction = model.predict([row])
            detected_label = prediction[0]

    current_label = detected_label
    now = time.time()

    # ===== LOGIC AUTO ĐỌC =====
    if current_label != "":
        # Nếu label mới xuất hiện
        if current_label != candidate_label:
            candidate_label = current_label
            candidate_start_time = now

        # Nếu label giữ ổn định đủ lâu và chưa đọc
        elif (now - candidate_start_time >= speak_hold_time) and (current_label != last_spoken_label):
            print("Auto speaking:", current_label)

            command = f'''PowerShell -Command "Add-Type -AssemblyName System.Speech; \
$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; \
$speak.Speak('{current_label}');"'''
            os.system(command)

            last_spoken_label = current_label

    else:
        # Nếu không thấy tay trong 1 khoảng thời gian -> reset để giơ lại sẽ đọc lại
        if now - last_hand_seen_time >= no_hand_reset_time:
            candidate_label = ""
            candidate_start_time = 0
            last_spoken_label = ""

    # ===== HIỂN THỊ =====
    cv2.putText(frame, f"Prediction: {current_label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, "Auto Speak ON | Q to quit", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Realtime Prediction", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()