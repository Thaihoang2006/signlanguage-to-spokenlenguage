import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import joblib
import time
import threading
import os

from gtts import gTTS
from playsound import playsound

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ===== FONT =====
font_path = "C:/Windows/Fonts/tahoma.ttf"

def draw_text(frame, text, position, size=40, color=(255,255,255)):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, size)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# ===== TTS =====
is_speaking = False

def speak(text):
    global is_speaking

    def run():
        global is_speaking
        try:
            is_speaking = True
            filename = "voice.mp3"

            tts = gTTS(text=text, lang='vi')
            tts.save(filename)

            playsound(filename)

            if os.path.exists(filename):
                os.remove(filename)

        except:
            pass

        is_speaking = False

    if not is_speaking:
        threading.Thread(target=run).start()

# ===== LOAD MODEL =====
model = joblib.load("hand_model.pkl")

# ===== MEDIAPIPE =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# ===== CAMERA =====
cap = cv2.VideoCapture(0)

# ===== STATE =====
current_label = ""
candidate_label = ""
candidate_start_time = 0
last_spoken_label = ""

speak_hold_time = 0.8
no_hand_reset_time = 1.0
last_hand_seen_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    detected_label = ""

    if result.multi_hand_landmarks:
        last_hand_seen_time = time.time()

        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ===== NORMALIZE =====
            row = []
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y

            for lm in hand_landmarks.landmark:
                row.append(lm.x - base_x)
                row.append(lm.y - base_y)

            prediction = model.predict([row])
            detected_label = prediction[0]

    current_label = detected_label
    now = time.time()

    status_text = "Chưa nhận diện"
    status_color = (200, 200, 200)

    # ===== AUTO SPEAK =====
    if current_label != "":
        status_text = "Đang nhận diện"
        status_color = (0, 255, 0)

        if current_label != candidate_label:
            candidate_label = current_label
            candidate_start_time = now

        elif now - candidate_start_time >= speak_hold_time:
            if current_label != last_spoken_label:
                speak(current_label)
                last_spoken_label = current_label
                status_text = "Đang phát âm"
                status_color = (0, 200, 255)

    else:
        if now - last_hand_seen_time >= no_hand_reset_time:
            candidate_label = ""
            candidate_start_time = 0
            last_spoken_label = ""

    # ===== UI PANEL =====
    overlay = frame.copy()

    # box nền
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # ===== TEXT =====
    frame = draw_text(frame, "Hand Sign Recognition", (10, 5), 30)
    frame = draw_text(frame, f" {current_label}", (10, 40), 50, (0,255,0))

    frame = draw_text(frame, status_text, (10, 85), 28, status_color)

    # hint
    frame = draw_text(frame, "Nhấn Q để thoát", (w-250, 85), 25, (200,200,200))

    cv2.imshow("AI Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()