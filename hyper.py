import cv2
import numpy as np
from ultralytics import YOLO
import time

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

prev_positions = {}

HYPER_THRESHOLD = 50
ALERT_DURATION = 5
alert_until_time = 0

# ⭐ Standing stability counters
standing_frames = 0
sitting_frames = 0
STANDING_CONFIRM_FRAMES = 5

def get_center(box):
    x1,y1,x2,y2 = box
    return int((x1+x2)/2), int((y1+y2)/2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    standing_detected_this_frame = False

    results = model.track(frame, persist=True, classes=[0], verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, id in zip(boxes, ids):
            x1,y1,x2,y2 = map(int, box)
            cx, cy = get_center(box)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            # --- STANDING DETECTION ---
            height = y2 - y1
            width  = x2 - x1
            ratio = height / width

            if ratio > 2.2:   # slightly relaxed threshold
                standing_detected_this_frame = True
                cv2.putText(frame,"STANDING!",
                            (x1,y2+25),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

            # --- HYPER MOVEMENT ---
            if id in prev_positions:
                px, py = prev_positions[id]
                distance = np.sqrt((cx-px)**2 + (cy-py)**2)

                if distance > HYPER_THRESHOLD:
                    alert_until_time = time.time() + ALERT_DURATION
                    cv2.putText(frame,"HYPER MOVEMENT!",
                                (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

            prev_positions[id] = (cx, cy)

    # ⭐ STANDING STABILITY LOGIC
    if standing_detected_this_frame:
        standing_frames += 1
        sitting_frames = 0
    else:
        sitting_frames += 1
        if sitting_frames > STANDING_CONFIRM_FRAMES:
            standing_frames = 0

    # 🚨 Hyper movement alert
    if time.time() < alert_until_time:
        cv2.putText(frame,"ALERT: Suspicious Activity",
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    # 🚨 Stable standing alert
    if standing_frames >= STANDING_CONFIRM_FRAMES:
        cv2.putText(frame,"ALERT: STUDENT STANDING",
                    (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    cv2.imshow("Hyper Movement Detector", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
