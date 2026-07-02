import cv2
import mediapipe as mp
import numpy as np

# إعداد Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# إعداد الكاميرا
cap = cv2.VideoCapture(0)

# متغيرات الرسم
points = []  # لتخزين جميع المسارات
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

    # معالجة الصورة (مرآة + تحويل ألوان)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # إنشاء خلفية داكنة رمضانية (تأثير 0.85 opacity في الكود الأصلي)
    overlay = frame.copy()
    overlay[:] = (2, 6, 10) # لون قريب من الأسود المذكور
    canvas = cv2.addWeighted(frame, 0.15, overlay, 0.85, 0)

    # التحقق من وجود يد
    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            # تتبع طرف السبابة (Landmark 8)
            ind = hand_lms.landmark[8]
            rx, ry = int(ind.x * w), int(ind.y * h)

            # تنعيم الحركة (Smoothing)
            sx += (rx - sx) * 0.45
            sy += (ry - sy) * 0.45
            pos = (int(sx), int(sy))

            # رسم الهيكل العظمي لليد (شفاف قليلاً)
            mp_draw.draw_landmarks(canvas, hand_lms, mp_hands.HAND_CONNECTIONS)

            # مؤشر الرسم
            color = (255, 255, 255) if drawing else (97, 208, 245) # BGR
            cv2.circle(canvas, pos, 6, color, -1)

            if drawing:
                current_path.append(pos)

    # رسم جميع المسارات المخزنة
    for path in points:
        for i in range(1, len(path)):
            # تأثير التوهج (رسم خط عريض باهت خلف الخط الأصلي)
            cv2.line(canvas, path[i-1], path[i], (97, 208, 245), 12) # التوهج
            cv2.line(canvas, path[i-1], path[i], (150, 230, 255), 4) # الخط الأساسي

    cv2.imshow('Ramadan Pen - Python', canvas)

    # التحكم بلوحة المفاتيح
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'): # بديل المفتاح Shift
        drawing = not drawing
        if drawing:
            current_path = []
            points.append(current_path)
    elif key == ord(' '): # مسح الشاشة
        points = []
    elif key == ord('q'): # خروج
        break

cap.release()
cv2.destroyAllWindows()
