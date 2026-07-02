import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

points = []  
current_path = []
drawing = False
sx, sy = 0, 0

print("التعليمات:")
print("- اضغط 's' (Shift البديل) للبدء/التوقف عن الرسم.")
print("- اضغط 'Space' لمسح الشاشة.")
print("- اضغط 'q' للخروج.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    overlay = frame.copy()
    overlay[:] = (2, 6, 10) 
    canvas = cv2.addWeighted(frame, 0.15, overlay, 0.85, 0)

    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            ind = hand_lms.landmark[8]
            rx, ry = int(ind.x * w), int(ind.y * h)

            sx += (rx - sx) * 0.45
            sy += (ry - sy) * 0.45
            pos = (int(sx), int(sy))

            mp_draw.draw_landmarks(canvas, hand_lms, mp_hands.HAND_CONNECTIONS)

            color = (255, 255, 255) if drawing else (97, 208, 245) # BGR
            cv2.circle(canvas, pos, 6, color, -1)

            if drawing:
                current_path.append(pos)

    for path in points:
        for i in range(1, len(path)):
            cv2.line(canvas, path[i-1], path[i], (97, 208, 245), 12)
            cv2.line(canvas, path[i-1], path[i], (150, 230, 255), 4) 

    cv2.imshow('Ramadan Pen - Python', canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'): 
        drawing = not drawing
        if drawing:
            current_path = []
            points.append(current_path)
    elif key == ord(' '):
        points = []
    elif key == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
