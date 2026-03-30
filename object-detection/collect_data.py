import cv2
import mediapipe as mp
import csv
import os

# Nhãn của ký hiệu muốn lưu
label = "b"

# Tạo file CSV nếu chưa có
file_name = "hand_data.csv"
file_exists = os.path.isfile(file_name)

# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Label: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press S to save | Q to quit", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Collect Hand Data", frame)

    key = cv2.waitKey(1) & 0xFF

    # Nhấn S để lưu dữ liệu
    if key == ord('s'):
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                row = [label]
                for lm in hand_landmarks.landmark:
                    row.append(lm.x)
                    row.append(lm.y)

                with open(file_name, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                print("Saved one sample!")

    # Nhấn Q để thoát
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()