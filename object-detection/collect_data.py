import cv2
import mediapipe as mp
import csv
import os
import time

# ===== PIL để hiển thị tiếng Việt =====
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ===== LABEL =====
label = "e"

# FILE CSV 
file_name = "hand_data.csv"
file_exists = os.path.isfile(file_name)

# MEDIAPIPE
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


last_save_time = 0
save_delay = 0.3

#
font_path = "C:/Windows/Fonts/arial.ttf"

def draw_text(frame, text, position, size=40, color=(0,255,0)):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, size)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

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

    #HIỂN THỊ TIẾNG VIỆT 
    frame = draw_text(frame, f"Label: {label}", (10, 20), 40)
    frame = draw_text(frame, "Nhấn S để lưu | Q để thoát", (10, 70), 30, (255,255,0))

    cv2.imshow("Collect Hand Data", frame)

    key = cv2.waitKey(1) & 0xFF

    #SAVE DATA
    if key == ord('s'):
        if result.multi_hand_landmarks:
            if time.time() - last_save_time > save_delay:

                for hand_landmarks in result.multi_hand_landmarks:

                    # ===== NORMALIZE =====
                    base_x = hand_landmarks.landmark[0].x
                    base_y = hand_landmarks.landmark[0].y

                    row = [label]

                    for lm in hand_landmarks.landmark:
                        row.append(lm.x - base_x)
                        row.append(lm.y - base_y)

                    with open(file_name, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)

                print("đã lưu 1 mẫu!")
                last_save_time = time.time()

    # ===== QUIT =====
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()